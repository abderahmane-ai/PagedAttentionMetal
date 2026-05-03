# Benchmark Methodology

## Hardware Configuration

All benchmarks were conducted on:
- **Device:** Apple M4 (2024)
- **Memory:** 11 GB unified memory
- **OS:** macOS 14.0+
- **Swift:** 6.0
- **Metal:** 3.0

## Benchmark Suite

### 1. Prefill Benchmark
**File:** `PrefillBenchmark.swift`

Tests attention computation for initial prompt processing.

**Configuration:**
- Sequence lengths: 256, 512, 1024, 2048 tokens
- Head dimension: 128
- Number of heads: 8 (Q) / 2 (KV) - GQA configuration
- Block size: 16
- Data types: FP32, FP16
- Iterations: 100 (with 10 warmup)

**Metrics:**
- Average latency (ms)
- Throughput (tokens/sec)
- Memory bandwidth utilization

### 2. Decode Benchmark
**File:** `DecodeBenchmark.swift`

Tests single-token generation (the LLM hot path).

**Configuration:**
- Batch sizes: 1, 4, 8, 16
- Context length: 1024 tokens
- Head dimension: 128
- Number of heads: 8 (Q) / 2 (KV)
- Block size: 16
- Data types: FP32, FP16
- Iterations: 100 (with 10 warmup)

**Metrics:**
- Average latency (ms)
- Throughput (tokens/sec)
- Batch efficiency

### 3. Memory Benchmark
**File:** `MemoryBenchmark.swift`

Compares memory usage between contiguous and paged KV cache.

**Configuration:**
- Sequence lengths: 512, 1024, 2048, 4096 tokens
- Fragmentation scenarios: 25%, 50%, 75%
- Block size: 16

**Metrics:**
- Memory allocated (MB)
- Memory wasted (fragmentation)
- Memory savings vs contiguous

### 4. Precision Benchmark
**File:** `PrecisionBenchmark.swift`

Validates FP16 accuracy against FP32 reference.

**Configuration:**
- Sequence length: 1024 tokens
- Head dimension: 128
- Number of heads: 8 (Q) / 2 (KV)
- Block size: 16

**Metrics:**
- Maximum absolute error
- Mean absolute error
- Speedup (FP16 vs FP32)
- Memory reduction

## Running Benchmarks

### Quick Benchmark (30 seconds)
```bash
swift run Benchmarks -c release
```

### Full Benchmark Suite (2-3 minutes)
```bash
swift run Benchmarks -c release --comprehensive
```

### Individual Benchmarks
```bash
swift run Benchmarks -c release --prefill
swift run Benchmarks -c release --decode
swift run Benchmarks -c release --memory
swift run Benchmarks -c release --precision
```

## Interpreting Results

### Prefill Performance
- **Good:** >20M tokens/sec (FP16)
- **Excellent:** >30M tokens/sec (FP16)

### Decode Performance
- **Good:** >800 tokens/sec (batch=8, FP16)
- **Excellent:** >1200 tokens/sec (batch=8, FP16)

### FP16 Speedup
- **Expected:** 1.5-2.0x vs FP32
- **Achieved:** 1.89x (M4)

### Memory Savings
- **Paged vs Contiguous:** 20-40% with fragmentation
- **FP16 vs FP32:** 2.0x exact

## Comparison Methodology

### vs MLX Native Attention
To compare against MLX's built-in attention:

1. Run MLX benchmark:
```bash
cd MLXDemo
swift run MLXDemo -c release
```

2. Compare metrics:
- Prefill latency
- Decode latency
- Memory usage

### vs vLLM (Python)
For cross-platform comparison:

1. Install vLLM:
```bash
pip install vllm
```

2. Run equivalent workload:
```python
from vllm import LLM
llm = LLM(model="meta-llama/Llama-3.2-1B")
# Measure throughput
```

3. Compare:
- Tokens/sec (prefill)
- Tokens/sec (decode)
- Memory efficiency

## Reproducing Published Results

### M4 Results (Published)
```
Prefill (1024 tok, FP16): 0.03ms → 30.7M tokens/sec
Decode (batch=8, FP16): 6.60ms → 1,212 tokens/sec
FP16 Speedup: 1.89x
```

**To reproduce:**
```bash
swift run Benchmarks -c release
```

Expected variance: ±5% due to thermal throttling and background processes.

## Benchmark Data

Results are saved to:
- `Benchmarks/Results/benchmark_YYYY-MM-DDTHH:MM:SSZ.json`
- `Benchmarks/Results/benchmark_YYYY-MM-DDTHH:MM:SSZ.csv`

## Notes

- All benchmarks use release builds (`-c release`)
- Warmup iterations prevent cold-start bias
- Metal API validation is disabled for accurate timing
- Results may vary based on thermal state and system load
- Close other GPU-intensive applications before benchmarking

## Contributing

To add new benchmarks:
1. Create new file in `Benchmarks/Sources/`
2. Follow existing benchmark structure
3. Add to `main.swift` benchmark suite
4. Update this README with methodology
