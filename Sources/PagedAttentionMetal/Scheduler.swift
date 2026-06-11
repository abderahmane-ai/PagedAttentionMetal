import Foundation
import os
import os.lock

/// The lifecycle status of a sequence managed by the scheduler.
public enum SequenceStatus: Sendable {
    /// The sequence is queued and waiting to be scheduled.
    case waiting
    /// The sequence is actively being processed.
    case running
    /// The sequence has been preempted (paused) due to memory pressure.
    case paused
    /// The sequence has finished generating all requested tokens.
    case completed
}

public struct GenerationRequest: Sendable {
    public let id: Int
    public let promptTokenCount: Int
    public let maxNewTokens: Int

    public init(id: Int, promptTokenCount: Int, maxNewTokens: Int) {
        self.id = id
        self.promptTokenCount = promptTokenCount
        self.maxNewTokens = maxNewTokens
    }
}

/// Represents a single generation sequence tracked by the continuous batching scheduler.
public final class SchedulerSequence: @unchecked Sendable {
    /// The unique identifier for this sequence.
    public let id: Int
    /// The current lifecycle status of the sequence.
    public var status: SequenceStatus = .waiting
    /// The number of tokens generated so far during the decode phase.
    public var numGeneratedTokens: Int = 0
    /// The maximum number of new tokens to generate.
    public let maxNewTokens: Int
    /// The number of tokens in the initial prompt.
    public let promptLength: Int
    /// The total sequence length (prompt + generated tokens).
    public var currentLength: Int { promptLength + numGeneratedTokens }
    /// Whether the sequence has finished generating all requested tokens.
    public var isDone: Bool { numGeneratedTokens >= maxNewTokens }
    /// The order in which this sequence arrived (used for fair scheduling).
    public let arrivalOrder: Int

    /// Creates a new scheduler sequence.
    public init(id: Int, promptLength: Int, maxNewTokens: Int, arrivalOrder: Int) {
        self.id = id
        self.promptLength = promptLength
        self.maxNewTokens = maxNewTokens
        self.arrivalOrder = arrivalOrder
    }
}

/// Scheduler implementing continuous batching for paged attention.
///
/// Manages the lifecycle of sequences (waiting → running → paused/completed)
/// and constructs batches for each step while respecting memory pressure.
public final class ContinuousBatchingScheduler: @unchecked Sendable {
    private static let log = OSLog(subsystem: "com.pagedattentionmetal", category: "scheduler")

    /// All sequences currently tracked by the scheduler, keyed by ID.
    public private(set) var sequences: [Int: SchedulerSequence] = [:]
    /// The maximum number of sequences that can be included in a single batch.
    public let maxBatchSize: Int
    /// The maximum total number of sequences the scheduler will accept.
    public let maxSequences: Int

    private var nextSequenceId: Int = 0
    private var arrivalCounter: Int = 0
    private let lock = OSAllocatedUnfairLock()

    /// Creates a new continuous batching scheduler.
    public init(maxBatchSize: Int, maxSequences: Int = 128) {
        self.maxBatchSize = maxBatchSize
        self.maxSequences = maxSequences
    }

    /// Adds a new generation request to the scheduler.
    /// - Parameters:
    ///   - promptTokenCount: The number of tokens in the input prompt.
    ///   - maxNewTokens: The maximum number of tokens to generate.
    /// - Returns: The unique sequence ID, or -1 if the scheduler is full.
    @discardableResult
    public func addRequest(promptTokenCount: Int, maxNewTokens: Int) -> Int {
        lock.lock()
        defer { lock.unlock() }
        guard sequences.count < maxSequences else { return -1 }
        let id = nextSequenceId
        nextSequenceId += 1
        let arrivalOrder = arrivalCounter
        arrivalCounter += 1
        let seq = SchedulerSequence(id: id, promptLength: promptTokenCount, maxNewTokens: maxNewTokens, arrivalOrder: arrivalOrder)
        sequences[id] = seq
        return id
    }

    /// Marks a sequence as completed.
    /// - Parameter id: The sequence ID.
    public func completeSequence(id: Int) {
        lock.lock()
        defer { lock.unlock() }
        sequences[id]?.status = .completed
    }

    /// Constructs the next batch of sequences to process.
    ///
    /// Priority order: waiting → running → paused. The batch is filled up to `maxBatchSize`.
    /// - Returns: An array of sequences to execute in this step.
    public func step() -> [SchedulerSequence] {
        lock.lock()
        defer { lock.unlock() }

        let waiting = sequences.values.filter { $0.status == .waiting }.sorted { $0.arrivalOrder < $1.arrivalOrder }
        let running = sequences.values.filter { $0.status == .running && !$0.isDone }.sorted { $0.arrivalOrder < $1.arrivalOrder }
        let paused = sequences.values.filter { $0.status == .paused }.sorted { $0.arrivalOrder < $1.arrivalOrder }

        var batch: [SchedulerSequence] = []

        let waitingSlots = min(waiting.count, maxBatchSize)
        let waitingToRun = waiting.prefix(waitingSlots)
        for seq in waitingToRun {
            seq.status = .running
            batch.append(seq)
            os_log(.debug, log: Self.log, "Scheduler: sequence %d promoted waiting→running", seq.id)
        }

        let remainingAfterWaiting = maxBatchSize - batch.count
        if remainingAfterWaiting > 0 {
            let runningToAdd = running.prefix(remainingAfterWaiting)
            batch.append(contentsOf: runningToAdd)
        }

        let remainingAfterRunning = maxBatchSize - batch.count
        if remainingAfterRunning > 0 {
            let pausedToResume = paused.prefix(remainingAfterRunning)
            for seq in pausedToResume {
                seq.status = .running
                batch.append(seq)
                os_log(.debug, log: Self.log, "Scheduler: sequence %d resumed paused→running", seq.id)
            }
        }

        os_log(.debug, log: Self.log, "Scheduler step: batch size %d (waiting=%d, running=%d, paused=%d)", batch.count, waiting.count, running.count, paused.count)
        return batch
    }

    /// Records that a sequence has completed its prefill phase.
    /// - Parameter id: The sequence ID.
    public func markPrefilled(id: Int) {
        lock.withLock {}
    }

    /// Records that a sequence has generated one additional token during decode.
    /// - Parameter id: The sequence ID.
    public func markDecoded(id: Int) {
        lock.lock()
        defer { lock.unlock() }
        sequences[id]?.numGeneratedTokens += 1
    }

    /// Preempts a running sequence, moving it to the paused state.
    /// - Parameter id: The sequence ID.
    public func preemptSequence(id: Int) {
        lock.lock()
        defer { lock.unlock() }
        sequences[id]?.status = .paused
    }

    /// Preempts running sequences (starting from the most recently arrived) until the required blocks are available.
    /// - Parameters:
    ///   - requiredBlocks: The number of free blocks needed.
    ///   - availableBlocks: Autoclosure returning the current number of free blocks.
    /// - Returns: The IDs of any sequences that were preempted.
    @discardableResult
    public func handleMemoryPressure(requiredBlocks: Int, availableBlocks: @autoclosure () -> Int) -> [Int] {
        lock.lock()
        defer { lock.unlock() }
        var preempted: [Int] = []
        while availableBlocks() < requiredBlocks {
            let candidates = sequences.values
                .filter { $0.status == .running }
                .sorted { $0.arrivalOrder > $1.arrivalOrder }
            guard let victim = candidates.first else { break }
            victim.status = .paused
            preempted.append(victim.id)
            os_log(.error, log: Self.log, "Preempted sequence %d (needed %d blocks, got %d available)", victim.id, requiredBlocks, availableBlocks())
        }
        if !preempted.isEmpty {
            os_log(.error, log: Self.log, "Memory pressure: preempted %d sequences", preempted.count)
        }
        return preempted
    }

    /// The number of sequences currently in the waiting state.
    public var waitingCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return sequences.values.filter { $0.status == .waiting }.count
    }

    /// The number of sequences currently in the running state.
    public var runningCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return sequences.values.filter { $0.status == .running }.count
    }

    /// The number of sequences that have completed generation.
    public var completedCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return sequences.values.filter { $0.status == .completed }.count
    }

    /// The total number of sequences that have been handled (including completed ones).
    public var totalSequencesProcessed: Int {
        lock.lock()
        defer { lock.unlock() }
        return sequences.count
    }
}
