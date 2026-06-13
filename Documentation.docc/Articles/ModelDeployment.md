# Model Deployment Guide

Deploy PagedAttentionMetal in production environments.

## Deployment Options

### 1. Embedded in Application

Bundle the library directly in your macOS/iOS app:

```swift
import Metal
import PagedAttentionMetal

class ModelManager {
    let engine: PagedAttentionEngine
    let cacheManager: BatchKVCacheManager
    let scheduler: ContinuousBatchingScheduler
    let layerSpecs: [PagedLayerSpec]
    
    init() throws {
        let device = MTLCreateSystemDefaultDevice()!
        self.engine = try PagedAttentionEngine()
        
        self.layerSpecs = (0..<32).map { _ in
            PagedLayerSpec(
                headDim: 128,
                numHeads: 32,
                numKVHeads: 8,
                blockSize: 16,
                dataType: .float16
            )
        }
        
        let firstSpec = layerSpecs[0]
        self.cacheManager = try BatchKVCacheManager(
            device: device,
            maxBatchSize: 32,
            maxSequenceBlocks: 1024,
            maxBlocks: 8192,
            blockSize: firstSpec.blockSize,
            headDim: firstSpec.headDim,
            numKVHeads: firstSpec.numKVHeads,
            dataType: firstSpec.dataType
        )
        
        self.scheduler = ContinuousBatchingScheduler(
            maxBatchSize: 32,
            maxSequences: 128
        )
    }
}
```

### 2. HTTP Server (PagedAttentionServer)

Run as a standalone inference server:

```bash
# Build
swift build -c release --product PagedAttentionServer

# Run in dummy mode (ideal for testing API integrations)
.build/release/PagedAttentionServer --dummy

# Run with a loaded model path
.build/release/PagedAttentionServer
```

#### Server Configuration Environment Variables

- `PORT` / `SERVER_PORT` - Port to listen on (default: `8080`)
- `PAGED_ATTENTION_API_KEY` - Optional key to enforce token authentication (`Authorization: Bearer <key>`)
- `RATE_LIMIT_RPM` - Rate Limit in requests per minute per IP (default: `60`)
- `MAX_BATCH_SIZE` - Maximum batch size (default: `16`)
- `MAX_BLOCKS` - Max physical cache blocks (default: `2048`)

### 3. Docker Deployment

```dockerfile
# Multi-stage build included in repository
docker build -t pagedattention:latest .
docker run -d -p 8080:8080 \
  -e PAGED_ATTENTION_API_KEY="your-secret-token" \
  -e RATE_LIMIT_RPM=60 \
  pagedattention:latest
```

---

## Memory Planning

| Model Size | Precision | KV Cache (8K ctx) | Recommended RAM |
|------------|-----------|-------------------|-----------------|
| 7B         | FP16      | ~2.5 GB           | 8 GB            |
| 7B         | FP8       | ~1.3 GB           | 6 GB            |
| 13B        | FP16      | ~4.5 GB           | 16 GB           |
| 13B        | FP8       | ~2.3 GB           | 8 GB            |
| 32B        | FP16      | ~11 GB            | 32 GB           |
| 32B        | FP8       | ~5.5 GB           | 16 GB           |

Calculate KV cache:
```
KV Cache = 2 * num_layers * num_kv_heads * head_dim * max_seq_len * batch_size * bytes_per_element
```

---

## Performance Tuning

### Block Size Selection

```swift
// Smaller blocks = better memory utilization, more metadata overhead
// Larger blocks = less metadata, potential fragmentation
let blockSize: Int = 16  // Standard default for PagedAttention
```

### Precision Selection

```swift
// float32: Highest accuracy, slowest, most memory
// float16: Balanced (recommended default)
// float8: Fastest, least memory (quantized)
let dataType: PagedAttentionDataType = .float16
```

---

## Monitoring and Metrics

Query the `/metrics` endpoint to monitor server and cache stats:

```bash
curl http://localhost:8080/metrics
```

This returns JSON metrics:
- `server`: Total processed requests, current active requests, unauthorized requests, rate-limited requests, and errors.
- `kv_cache`: Memory usage, allocated blocks, active sequences, block fragmentation ratio, and prefix cache hit rate.

---

## Security Checklist

- [x] Enable API key authentication (`PAGED_ATTENTION_API_KEY`)
- [x] Configure rate limiting (`RATE_LIMIT_RPM`)
- [ ] Enable TLS/SSL (HTTPS) in front of the server using a reverse proxy (e.g. Nginx, Envoy, or Cloudflare)