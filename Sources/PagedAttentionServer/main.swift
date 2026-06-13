import Foundation
import Metal
import MLX
import MLXLMCommon
import MLXHuggingFace
import HuggingFace
import Hub
import Tokenizers
import NIOCore
import NIOPosix
import NIOHTTP1
import PagedAttentionMetal
import PagedAttentionMLXSupport

final class RateLimiter: @unchecked Sendable {
    private let limit: Int
    private let windowSize: TimeInterval
    private let lock = NSLock()
    private var clientRequests: [String: [Date]] = [:]

    init(limit: Int, windowSize: TimeInterval = 60.0) {
        self.limit = limit
        self.windowSize = windowSize
    }

    func isAllowed(clientKey: String) -> Bool {
        lock.lock()
        defer { lock.unlock() }

        let now = Date()
        let cutoff = now.addingTimeInterval(-windowSize)

        var requests = clientRequests[clientKey] ?? []
        requests = requests.filter { $0 > cutoff }

        if requests.count < limit {
            requests.append(now)
            clientRequests[clientKey] = requests
            return true
        } else {
            clientRequests[clientKey] = requests
            return false
        }
    }
}

final class ServerMetrics: @unchecked Sendable {
    private let lock = NSLock()
    private(set) var totalRequests: Int = 0
    private(set) var activeRequests: Int = 0
    private(set) var unauthorizedRequests: Int = 0
    private(set) var rateLimitedRequests: Int = 0
    private(set) var errorRequests: Int = 0

    func incrementTotal() {
        lock.lock()
        totalRequests += 1
        activeRequests += 1
        lock.unlock()
    }

    func decrementActive() {
        lock.lock()
        activeRequests = max(0, activeRequests - 1)
        lock.unlock()
    }

    func incrementUnauthorized() {
        lock.lock()
        unauthorizedRequests += 1
        lock.unlock()
    }

    func incrementRateLimited() {
        lock.lock()
        rateLimitedRequests += 1
        lock.unlock()
    }

    func incrementError() {
        lock.lock()
        errorRequests += 1
        lock.unlock()
    }
}

final class CompletionHandler: @unchecked Sendable, ChannelInboundHandler {
    typealias InboundIn = HTTPServerRequestPart
    typealias OutboundOut = HTTPServerResponsePart

    private let inference: PagedAttentionInference
    private let generator: PagedAttentionGenerator
    private let tokenizer: MLXLMCommon.Tokenizer
    private let rateLimiter: RateLimiter?
    private let apiKey: String?
    private let metrics: ServerMetrics
    private var requestState: [ObjectIdentifier: (head: HTTPRequestHead, body: Data)] = [:]
    private let stateLock = NSLock()

    init(
        inference: PagedAttentionInference,
        generator: PagedAttentionGenerator,
        tokenizer: MLXLMCommon.Tokenizer,
        rateLimiter: RateLimiter?,
        apiKey: String?,
        metrics: ServerMetrics
    ) {
        self.inference = inference
        self.generator = generator
        self.tokenizer = tokenizer
        self.rateLimiter = rateLimiter
        self.apiKey = apiKey
        self.metrics = metrics
    }

    func channelRead(context: ChannelHandlerContext, data: NIOAny) {
        let channelID = ObjectIdentifier(context.channel)
        let reqPart = unwrapInboundIn(data)
        switch reqPart {
        case .head(let head):
            stateLock.lock()
            requestState[channelID] = (head, Data())
            stateLock.unlock()
        case .body(var buffer):
            stateLock.lock()
            if let read = buffer.readBytes(length: buffer.readableBytes) {
                requestState[channelID]?.body.append(contentsOf: read)
            }
            stateLock.unlock()
        case .end:
            stateLock.lock()
            let state = requestState.removeValue(forKey: channelID)
            stateLock.unlock()
            guard let (head, body) = state else { return }
            handleRequest(context: context, head: head, body: body)
        }
    }

    func channelInactive(context: ChannelHandlerContext) {
        let channelID = ObjectIdentifier(context.channel)
        stateLock.lock()
        requestState.removeValue(forKey: channelID)
        stateLock.unlock()
        context.fireChannelInactive()
    }

    private func handleRequest(context: ChannelHandlerContext, head: HTTPRequestHead, body: Data) {
        metrics.incrementTotal()
        switch (head.method, head.uri) {
        case (.POST, "/v1/completions"):
            var isAuthorized = false
            if let apiKey {
                if let authHeader = head.headers.first(name: "Authorization"),
                   authHeader.starts(with: "Bearer ") {
                    let token = authHeader.dropFirst("Bearer ".count)
                    if token == apiKey {
                        isAuthorized = true
                    }
                }
            } else {
                isAuthorized = true
            }

            if !isAuthorized {
                metrics.incrementUnauthorized()
                metrics.decrementActive()
                sendJSON(context: context, status: .unauthorized, dict: ["error": "Unauthorized: Invalid or missing API key"])
                return
            }

            let clientKey = context.channel.remoteAddress?.ipAddress ?? "unknown"
            if let rateLimiter, !rateLimiter.isAllowed(clientKey: clientKey) {
                metrics.incrementRateLimited()
                metrics.decrementActive()
                sendJSON(context: context, status: .tooManyRequests, dict: ["error": "Too Many Requests: Rate limit exceeded"])
                return
            }

            handleCompletions(context: context, body: body)
        case (.GET, "/health"):
            metrics.decrementActive()
            sendJSON(context: context, status: .ok, dict: ["status": "ok"])
        case (.GET, "/metrics"), (.GET, "/stats"):
            metrics.decrementActive()
            handleMetrics(context: context)
        default:
            metrics.decrementActive()
            sendJSON(context: context, status: .notFound, dict: ["error": "not found"])
        }
    }

    private func handleMetrics(context: ChannelHandlerContext) {
        let stats = inference.cache.memoryStats()
        let evictionStr: String
        switch stats.evictionPolicy {
        case .fifo: evictionStr = "fifo"
        case .lifo: evictionStr = "lifo"
        case .bestFit: evictionStr = "bestFit"
        }

        let dict: [String: Any] = [
            "server": [
                "total_requests": metrics.totalRequests,
                "active_requests": metrics.activeRequests,
                "unauthorized_requests": metrics.unauthorizedRequests,
                "rate_limited_requests": metrics.rateLimitedRequests,
                "error_requests": metrics.errorRequests
            ],
            "kv_cache": [
                "total_blocks": stats.totalBlocks,
                "used_blocks": stats.usedBlocks,
                "free_blocks": stats.freeBlocks,
                "active_sequences": stats.activeSequences,
                "total_memory_bytes": stats.totalMemoryBytes,
                "used_memory_bytes": stats.usedMemoryBytes,
                "partially_filled_blocks": stats.partiallyFilledBlocks,
                "fragmentation_ratio": stats.fragmentationRatio,
                "prefix_cache_hits": stats.prefixCacheHits,
                "prefix_cache_entries": stats.prefixCacheEntries,
                "shared_blocks": stats.sharedBlocks,
                "eviction_policy": evictionStr
            ]
        ]
        sendJSON(context: context, status: .ok, dict: dict)
    }

    private func handleCompletions(context: ChannelHandlerContext, body: Data) {
        do {
            guard let json = try JSONSerialization.jsonObject(with: body) as? [String: Any],
                  let prompt = json["prompt"] as? String else {
                metrics.incrementError()
                metrics.decrementActive()
                sendJSON(context: context, status: .badRequest, dict: ["error": "prompt required"])
                return
            }

            let maxTokens   = json["max_tokens"]  as? Int    ?? 256
            let temperature = json["temperature"] as? Float  ?? 1.0
            let topP        = json["top_p"]       as? Float  ?? 1.0
            let topK        = json["top_k"]       as? Int    ?? 50
            let stream      = json["stream"]      as? Bool   ?? false

            guard maxTokens >= 0 else {
                metrics.incrementError()
                metrics.decrementActive()
                sendJSON(context: context, status: .badRequest, dict: ["error": "max_tokens must be non-negative"])
                return
            }
            guard (0...2).contains(temperature) else {
                metrics.incrementError()
                metrics.decrementActive()
                sendJSON(context: context, status: .badRequest, dict: ["error": "temperature must be between 0 and 2"])
                return
            }
            guard (0...1).contains(topP) else {
                metrics.incrementError()
                metrics.decrementActive()
                sendJSON(context: context, status: .badRequest, dict: ["error": "top_p must be between 0 and 1"])
                return
            }
            guard topK > 0 else {
                metrics.incrementError()
                metrics.decrementActive()
                sendJSON(context: context, status: .badRequest, dict: ["error": "top_k must be positive"])
                return
            }

            let config = SamplingConfig(temperature: temperature, topK: topK, topP: topP)
            let eventLoop = context.eventLoop

            final class SendableWriter: @unchecked Sendable {
                let ctx: ChannelHandlerContext
                let handler: CompletionHandler
                init(ctx: ChannelHandlerContext, handler: CompletionHandler) {
                    self.ctx = ctx; self.handler = handler
                }
                func sendJSON(_ status: HTTPResponseStatus, _ dict: Any) {
                    self.ctx.eventLoop.execute {
                        var head = HTTPResponseHead(version: .http1_1, status: status)
                        head.headers.add(name: "Content-Type", value: "application/json")
                        head.headers.add(name: "Access-Control-Allow-Origin", value: "*")
                        self.ctx.write(self.handler.wrapOutboundOut(.head(head)), promise: nil)
                        if let data = try? JSONSerialization.data(withJSONObject: dict),
                           let str = String(data: data, encoding: .utf8) {
                            let buf = self.ctx.channel.allocator.buffer(string: str)
                            self.ctx.write(self.handler.wrapOutboundOut(.body(.byteBuffer(buf))), promise: nil)
                        }
                        self.ctx.writeAndFlush(self.handler.wrapOutboundOut(.end(nil)), promise: nil)
                    }
                }
                func sendEvent(_ line: String) {
                    self.ctx.eventLoop.execute {
                        let buf = self.ctx.channel.allocator.buffer(string: line)
                        self.ctx.writeAndFlush(self.handler.wrapOutboundOut(.body(.byteBuffer(buf))), promise: nil)
                    }
                }
                func sendEnd() {
                    self.ctx.eventLoop.execute {
                        self.ctx.writeAndFlush(self.handler.wrapOutboundOut(.end(nil)), promise: nil)
                    }
                }
            }
            let writer = SendableWriter(ctx: context, handler: self)

            if stream {
                var respHead = HTTPResponseHead(version: .http1_1, status: .ok)
                respHead.headers.add(name: "Content-Type", value: "text/event-stream")
                respHead.headers.add(name: "Cache-Control", value: "no-cache")
                respHead.headers.add(name: "Connection", value: "keep-alive")
                respHead.headers.add(name: "Access-Control-Allow-Origin", value: "*")
                context.write(wrapOutboundOut(.head(respHead)), promise: nil)

                eventLoop.execute {
                    Task {
                        var count = 0
                        do {
                            let pTokens = self.tokenizer.encode(text: prompt, addSpecialTokens: true)
                            let s = try self.generator.generate(promptTokens: pTokens, maxNewTokens: maxTokens, samplingConfig: config)
                            for try await token in s {
                                count += 1
                                let text = self.tokenizer.decode(tokenIds: [token], skipSpecialTokens: true)
                                let escaped = text.replacingOccurrences(of: "\\", with: "\\\\")
                                    .replacingOccurrences(of: "\"", with: "\\\"")
                                    .replacingOccurrences(of: "\n", with: "\\n")
                                    .replacingOccurrences(of: "\r", with: "\\r")
                                    .replacingOccurrences(of: "\t", with: "\\t")
                                let line = "data: {\"choices\":[{\"text\":\"\(escaped)\",\"index\":0}],\"usage\":{\"completion_tokens\":\(count)}}\n\n"
                                writer.sendEvent(line)
                            }
                            writer.sendEvent("data: [DONE]\n\n")
                        } catch {
                            self.metrics.incrementError()
                            let msg = String(describing: error)
                                .replacingOccurrences(of: "\\", with: "\\\\")
                                .replacingOccurrences(of: "\"", with: "\\\"")
                                .replacingOccurrences(of: "\n", with: "\\n")
                                .replacingOccurrences(of: "\r", with: "\\r")
                                .replacingOccurrences(of: "\t", with: "\\t")
                            writer.sendEvent("data: {\"error\":\"\(msg)\"}\n\n")
                        }
                        self.metrics.decrementActive()
                        writer.sendEnd()
                    }
                }
            } else {
                eventLoop.execute {
                    Task {
                        do {
                            let pTokens = self.tokenizer.encode(text: prompt, addSpecialTokens: true)
                            let s = try self.generator.generate(promptTokens: pTokens, maxNewTokens: maxTokens, samplingConfig: config)
                            var tokens: [Int] = []
                            for try await token in s {
                                tokens.append(token)
                            }
                            let text = self.tokenizer.decode(tokenIds: tokens, skipSpecialTokens: true)
                            writer.sendJSON(.ok, [
                                "choices": [["text": text, "index": 0]],
                                "usage": ["completion_tokens": tokens.count]
                            ])
                        } catch {
                            self.metrics.incrementError()
                            writer.sendJSON(.internalServerError, ["error": "\(error)"])
                        }
                        self.metrics.decrementActive()
                    }
                }
            }
        } catch {
            metrics.incrementError()
            metrics.decrementActive()
            sendJSON(context: context, status: .internalServerError, dict: ["error": "\(error)"])
        }
    }

    private func sendJSON(context: ChannelHandlerContext, status: HTTPResponseStatus, dict: Any) {
        var head = HTTPResponseHead(version: .http1_1, status: status)
        head.headers.add(name: "Content-Type", value: "application/json")
        head.headers.add(name: "Access-Control-Allow-Origin", value: "*")
        context.write(wrapOutboundOut(.head(head)), promise: nil)
        if let data = try? JSONSerialization.data(withJSONObject: dict),
           let str = String(data: data, encoding: .utf8) {
            let buf = context.channel.allocator.buffer(string: str)
            context.write(wrapOutboundOut(.body(.byteBuffer(buf))), promise: nil)
        }
        context.writeAndFlush(wrapOutboundOut(.end(nil)), promise: nil)
    }
}

final class DummyModelAdapter: @unchecked Sendable, ModelAdapterProtocol {
    let hiddenSize: Int
    let vocabSize: Int
    let numLayers: Int
    let layerSpecs: [PagedLayerSpec]
    let device: MTLDevice

    init(layerSpec: PagedLayerSpec, vocabSize: Int, hiddenSize: Int) throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw PagedAttentionError.deviceInitializationFailed
        }
        self.hiddenSize = hiddenSize
        self.vocabSize = vocabSize
        self.numLayers = 1
        self.layerSpecs = [layerSpec]
        self.device = device
    }

    func embed(tokens: [Int]) throws -> MLXArray {
        var values = [Float](repeating: 0, count: tokens.count * hiddenSize)
        for (tokenIndex, token) in tokens.enumerated() {
            for dim in 0..<hiddenSize {
                values[tokenIndex * hiddenSize + dim] = Float((token + dim) % 127) / 127.0 - 0.5
            }
        }
        let data = Data(bytes: values, count: values.count * MemoryLayout<Float>.stride)
        return MLXArray(data, [tokens.count, hiddenSize], dtype: .float32)
    }

    func projectQKV(hidden: MLXArray, layer: Int, offset: Int) throws -> (q: MTLBuffer, k: MTLBuffer, v: MTLBuffer) {
        let spec = layerSpecs[layer]
        let seqLen = hidden.shape[0]
        let qSize = seqLen * spec.numHeads * spec.headDim * spec.dataType.byteWidth
        let kvSize = seqLen * spec.numKVHeads * spec.headDim * spec.dataType.byteWidth
        guard let qBuf = device.makeBuffer(length: qSize, options: .storageModeShared),
              let kBuf = device.makeBuffer(length: kvSize, options: .storageModeShared),
              let vBuf = device.makeBuffer(length: kvSize, options: .storageModeShared) else {
            throw PagedAttentionError.commandEncodingFailed("dummy QKV buffers")
        }
        switch spec.dataType {
        case .float16:
            fillHalfBuffer(qBuf, elements: seqLen * spec.numHeads * spec.headDim, seed: 11)
            fillHalfBuffer(kBuf, elements: seqLen * spec.numKVHeads * spec.headDim, seed: 23)
            fillHalfBuffer(vBuf, elements: seqLen * spec.numKVHeads * spec.headDim, seed: 37)
        case .float32:
            fillFloatBuffer(qBuf, elements: seqLen * spec.numHeads * spec.headDim, seed: 11)
            fillFloatBuffer(kBuf, elements: seqLen * spec.numKVHeads * spec.headDim, seed: 23)
            fillFloatBuffer(vBuf, elements: seqLen * spec.numKVHeads * spec.headDim, seed: 37)
        case .float8:
            fillFloat16BackedBuffer(qBuf, elements: seqLen * spec.numHeads * spec.headDim, seed: 11)
            fillHalfBuffer(kBuf, elements: seqLen * spec.numKVHeads * spec.headDim, seed: 23)
            fillHalfBuffer(vBuf, elements: seqLen * spec.numKVHeads * spec.headDim, seed: 37)
        }
        return (qBuf, kBuf, vBuf)
    }

    private func deterministic(_ index: Int, seed: Int) -> Float {
        let x = Float((index * 17 + seed * 31) % 2000)
        return (x - 1000) * 0.001
    }

    private func fillFloatBuffer(_ buffer: MTLBuffer, elements: Int, seed: Int) {
        let ptr = buffer.contents().assumingMemoryBound(to: Float.self)
        for i in 0..<elements {
            ptr[i] = deterministic(i, seed: seed)
        }
    }

    private func fillFloat16BackedBuffer(_ buffer: MTLBuffer, elements: Int, seed: Int) {
        let ptr = buffer.contents().assumingMemoryBound(to: Float16.self)
        for i in 0..<elements {
            ptr[i] = Float16(deterministic(i, seed: seed))
        }
    }

    private func fillHalfBuffer(_ buffer: MTLBuffer, elements: Int, seed: Int) {
        fillFloat16BackedBuffer(buffer, elements: elements, seed: seed)
    }

    func applyAttentionOutput(hidden: MLXArray, attentionFloats: [Float], layer: Int) throws -> MLXArray {
        hidden
    }

    func projectOutput(hidden: MLXArray) throws -> MLXArray {
        let seqLen = hidden.shape[0]
        var logits = [Float](repeating: 0, count: seqLen * vocabSize)
        for row in 0..<seqLen {
            let offset = row * vocabSize
            for token in 0..<vocabSize {
                let value = Float((hidden.shape[1] + token + row) % 97) / 97.0 - 0.5
                logits[offset + token] = value
            }
        }
        let data = Data(bytes: logits, count: logits.count * MemoryLayout<Float>.stride)
        return MLXArray(data, [seqLen, vocabSize], dtype: DType.float32)
    }
}

@main
struct PagedAttentionServer {
    static func main() async throws {
        let device = MTLCreateSystemDefaultDevice()!

        let portEnv = ProcessInfo.processInfo.environment["PORT"] ?? ProcessInfo.processInfo.environment["SERVER_PORT"] ?? "8080"
        let port = Int(portEnv) ?? 8080

        let apiKey = ProcessInfo.processInfo.environment["PAGED_ATTENTION_API_KEY"]

        let rateLimitEnv = ProcessInfo.processInfo.environment["RATE_LIMIT_RPM"] ?? "60"
        let rateLimit = Int(rateLimitEnv) ?? 60
        let rateLimiter = rateLimit > 0 ? RateLimiter(limit: rateLimit) : nil

        let sharedMetrics = ServerMetrics()

        var adapter: ModelAdapterProtocol
        var tokenizer: MLXLMCommon.Tokenizer

        if CommandLine.arguments.contains("--dummy") {
            let layerSpec = PagedLayerSpec(headDim: 64, numHeads: 8, numKVHeads: 2, blockSize: 16, dataType: .float16)
            adapter = try DummyModelAdapter(layerSpec: layerSpec, vocabSize: 1000, hiddenSize: 512)
            tokenizer = PassthroughTokenizer()
            print("Using dummy model adapter (--dummy mode)")
        } else {
            let cfg = ModelConfiguration(id: "mlx-community/Llama-3.2-1B-Instruct-4bit")
            print("Loading model: \(cfg.name)...")
            let container = try await #huggingFaceLoadModelContainer(configuration: cfg) { progress in
                print("Download: \(Int(progress.fractionCompleted * 100))%")
            }
            adapter = try await LlamaModelAdapter(container: container)
            tokenizer = await container.perform { context in
                context.tokenizer
            }
            print("Model loaded: \(cfg.name)")
        }

        let maxBlocks = 2048
        let blockSize = adapter.layerSpecs.first?.blockSize ?? 16
        let inference = try PagedAttentionInference(
            device: device,
            maxBlocks: maxBlocks,
            blockSize: blockSize,
            layerSpecs: adapter.layerSpecs,
            maxBatchSize: 16,
            maxSequences: 128
        )
        let generator = PagedAttentionGenerator(inference: inference, modelAdapter: adapter)
        let capturedTokenizer = tokenizer

        let group = MultiThreadedEventLoopGroup(numberOfThreads: System.coreCount)
        let bootstrap = ServerBootstrap(group: group)
            .serverChannelOption(ChannelOptions.backlog, value: 256)
            .serverChannelOption(ChannelOptions.socketOption(.so_reuseaddr), value: 1)
            .childChannelInitializer { channel in
                let handler = CompletionHandler(
                    inference: inference,
                    generator: generator,
                    tokenizer: capturedTokenizer,
                    rateLimiter: rateLimiter,
                    apiKey: apiKey,
                    metrics: sharedMetrics
                )
                return channel.pipeline.configureHTTPServerPipeline().flatMap {
                    channel.pipeline.addHandler(handler)
                }
            }
            .childChannelOption(ChannelOptions.socketOption(.so_reuseaddr), value: 1)
            .childChannelOption(ChannelOptions.maxMessagesPerRead, value: 1)

        let channel = try await bootstrap.bind(host: "0.0.0.0", port: port).get()
        print("PagedAttention server listening on http://localhost:\(port)")
        print("  POST /v1/completions")
        print("  GET  /health")
        print("  GET  /metrics")
        if let apiKey {
            print("  Enforcing token authentication (API Key configured)")
        }
        if rateLimit > 0 {
            print("  Rate Limiting active: \(rateLimit) requests per minute per IP")
        }
        try await channel.closeFuture.get()
    }
}

private struct PassthroughTokenizer: MLXLMCommon.Tokenizer {
    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        Array(text.utf8).map { Int($0) }
    }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        String(tokenIds.compactMap { UnicodeScalar($0).map(Character.init) })
    }
    func convertTokenToId(_ token: String) -> Int? { nil }
    func convertIdToToken(_ id: Int) -> String? { nil }
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }
    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [] }
}
