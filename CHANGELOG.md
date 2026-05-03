# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2026-05-03

### 🚨 BREAKING CHANGES
- **Removed `forward()` method** — replaced with purpose-built `prefill()`, `decode()`, `appendToCache()`, and `backward()` methods
- **API redesign** — all methods now require explicit `blockSize` parameter (no longer hardcoded to 16)
- **Metal kernels** — removed `constant uint BLOCK_SIZE = 16`, now runtime-configurable

### Added
- **Runtime-configurable block size:** Pass any block size (8, 16, 32, 64, etc.) at dispatch time
- **Causal masking:** All attention kernels now support `causal` flag for autoregressive generation
- **Batch decode kernel:** `paged_decode_single` optimized for single-token generation across multiple sequences
- **KV cache append kernels:** `kv_cache_append` for GPU-side cache writes (zero CPU memcpy)
- **BatchKVCacheManager:** Manages 2D block tables `[batchSize × maxSequenceBlocks]` for batch processing
- **Dynamic head_dim support:** Removed 128-dimension limit via threadgroup memory allocation
- **Prefill method:** `prefill()` with causal masking and configurable block size
- **Decode method:** `decode()` for batch single-token generation
- **AppendToCache method:** `appendToCache()` for GPU-side KV cache writes

### Changed
- **Phase1 kernels:** Now use threadgroup memory for `acc_o` instead of hardcoded `float acc_o[128]` array
- **Backward kernel:** Upgraded to 3D grid with full MHA/GQA support (`num_heads`/`num_kv_heads` parameters)
- **Split-K phase1:** Threadgroup memory sized dynamically based on `head_dim`
- **All kernels:** Accept `block_size` as runtime parameter instead of compile-time constant

### Fixed
- **head_dim > 128 crash:** Eliminated hardcoded 128-element stack arrays
- **MHA backward pass:** Now correctly handles multiple heads with 3D grid dispatch
- **Block size flexibility:** Any block size now supported without shader recompilation

## [1.0.0] - 2026-05-03

### Added
- **Multi-Head Attention (MHA):** Fully upgraded compute kernels to process multiple attention heads simultaneously using 3D threadgroup grids (`gid.z`).
- **Grouped-Query Attention (GQA):** Integrated native GQA/MQA support via dynamic KV-to-Query head indexing, matching modern architectures (e.g., LLaMA 3, Mistral).
- **FP16 Mixed Precision:** Implemented specialized `_f16` Metal kernels that halve memory bandwidth consumption (2x speedup) while computing internal Softmax/dot-products in 32-bit `float` for mathematical stability.
- **Dynamic KV Cache Manager:** Added `KVCacheManager.swift`, a GPU memory orchestrator that automatically tracks, allocates, and frees physical KV blocks for virtual LLM sequences in O(1) time.
- **SIMD FP16 Helpers:** Added custom `simd_dot_product_f16` vector instructions, utilizing `float4` to extract maximum ALU performance out of 16-bit cache reads.
- **Public API Scaling:** Expanded `PagedAttentionEngine.forward()` to accept `numHeads`, `numKVHeads`, and `dataType` arguments for direct precision and architectural configuration.
- **Verification Tests:** Added rigorous `runMultiHeadPagedAttention()`, `runGroupedQueryAttention()`, and `runKVCacheManagerTest()` CPU/GPU validations.
- **ExampleApp Benchmarks:** Added an explicit FP32 vs FP16 latency benchmark to the ExampleApp executable, demonstrating near theoretical-maximum speedups.

### Changed
- **Threadgroup Stability:** Hardened `paged_attention_single` kernel by unifying vector declarations to `uint3` to guarantee compilation stability on Apple Silicon.
- **Engine Memory Allocation:** Scaled all Split-K intermediate buffers by `numHeads` for proper Z-axis memory segmentation, and ensured intermediate precision safety by locking threadgroup memory maps to Float32 sizes.
- **Swift Concurrency:** Enforced strict `.v6` Swift language modes across all data structures and test arrays, updating raw memory allocations to `withUnsafeBytes`.

### Removed
- **Dead Code Eradication:** Purged orphaned `paged_attention_half` and `kv_cache_copy` kernels, along with legacy global `PipelineCache` aliases.
- **SPM Boilerplate:** Removed all auto-generated template comments from `Package.swift`.
- **Ghost Tests:** Deleted empty scaffolding code inside `ExampleApp/Tests`.

## [0.1.0] - 2026-05-02

### Added
- **Initial Release:** Production-ready Paged Attention Metal implementation.
- **V2 Split-K Map-Reduce Kernel:** Supports massive context window sequences by bypassing local threadgroup memory limits.
- **Hardware-Accelerated Atomics:** Replaced standard CAS spinlocks with Metal 3 `atomic_fetch_add_explicit` for contention-free gradients.
- **Vectorized Math:** Implemented `simd_float4` operations for maximal memory-read throughput on Apple 128-bit ALUs.
- **Swift API (`PagedAttentionEngine`):** High-level abstraction for dynamic runtime kernel selection.
- **JIT Compiler Bypass:** Added dynamic runtime linking of `.metal` assets to prevent SPM `.metallib` corruption.
- **Mathematical Benchmark Harness:** Fully integrated CPU vs. GPU validation suite ensuring $1.5 \times 10^{-7}$ precision.
- **DocC Documentation:** Added comprehensive inline Apple-standard API documentation.
- **CI/CD Actions:** Added GitHub Actions workflows for automated verification on macOS runners.
