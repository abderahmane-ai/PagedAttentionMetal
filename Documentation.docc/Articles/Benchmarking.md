# Benchmarking Guide

Measure and optimize PagedAttentionMetal performance.

## Built-in Benchmarks

Run the included benchmark executable:

```bash
# Run all benchmarks
swift run -c release Benchmarks all

# Run specific benchmark command
swift run -c release Benchmarks prefill --seq-len 1024
swift run -c release Benchmarks decode --batch-size 8
swift run -c release Benchmarks memory
```

### Benchmark Commands:
- `all` - Runs all prefill, decode, memory, and precision comparative benchmarks.
- `prefill` - Measures prefill performance across various sequence lengths.
- `decode` - Measures decode step performance under batch size configurations.
- `memory` - Gauges KV cache overhead and prefix cache savings.
- `precision` - Measures output correctness/precision comparison (FP32 vs FP16 vs FP8).

---

## Performance Tuning Checklist

- [ ] Use FP16 instead of FP32 (2x memory savings, ~2x GPU speedup)
- [ ] Enable FP8 (float8) on supported layouts for 4x memory savings and accelerated throughput.
- [ ] Utilize prefix cache mapping to reuse prompt prefixes across requests.
- [ ] Maximize batch sizes within memory boundaries.
- [ ] Use continuous batching for high server concurrency.