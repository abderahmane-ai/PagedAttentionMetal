import Foundation
import os
import os.lock

public enum SequenceStatus: Sendable {
    case waiting
    case running
    case paused
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

public final class SchedulerSequence: @unchecked Sendable {
    public let id: Int
    public var status: SequenceStatus = .waiting
    public var numGeneratedTokens: Int = 0
    public let maxNewTokens: Int
    public let promptLength: Int
    public var currentLength: Int { promptLength + numGeneratedTokens }
    public var isDone: Bool { numGeneratedTokens >= maxNewTokens }
    public let arrivalOrder: Int

    public init(id: Int, promptLength: Int, maxNewTokens: Int, arrivalOrder: Int) {
        self.id = id
        self.promptLength = promptLength
        self.maxNewTokens = maxNewTokens
        self.arrivalOrder = arrivalOrder
    }
}

public final class ContinuousBatchingScheduler: @unchecked Sendable {
    private static let log = OSLog(subsystem: "com.pagedattentionmetal", category: "scheduler")

    public private(set) var sequences: [Int: SchedulerSequence] = [:]
    public let maxBatchSize: Int
    public let maxSequences: Int

    private var nextSequenceId: Int = 0
    private var arrivalCounter: Int = 0
    private let lock = OSAllocatedUnfairLock()

    public init(maxBatchSize: Int, maxSequences: Int = 128) {
        self.maxBatchSize = maxBatchSize
        self.maxSequences = maxSequences
    }

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

    public func completeSequence(id: Int) {
        lock.lock()
        defer { lock.unlock() }
        sequences[id]?.status = .completed
    }

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

    public func markPrefilled(id: Int) {
        lock.lock()
        defer { lock.unlock() }
        sequences[id]?.status = .running
    }

    public func markDecoded(id: Int) {
        lock.lock()
        defer { lock.unlock() }
        sequences[id]?.numGeneratedTokens += 1
    }

    public func preemptSequence(id: Int) {
        lock.lock()
        defer { lock.unlock() }
        sequences[id]?.status = .paused
    }

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

    public var waitingCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return sequences.values.filter { $0.status == .waiting }.count
    }

    public var runningCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return sequences.values.filter { $0.status == .running }.count
    }

    public var completedCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return sequences.values.filter { $0.status == .completed }.count
    }

    public var totalSequencesProcessed: Int {
        lock.lock()
        defer { lock.unlock() }
        return sequences.count
    }
}
