# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-05-03

### Added
- **Multi-Head Attention (MHA):** Fully upgraded compute kernels to process multiple attention heads simultaneously using 3D threadgroup grids (`gid.z`).
- **Grouped-Query Attention (GQA):** Integrated native GQA/MQA support via dynamic KV-to-Query head indexing, matching modern architectures (e.g., LLaMA 3, Mistral).
- **Public API Scaling:** Expanded `PagedAttentionEngine.forward()` to accept `numHeads` and `numKVHeads` arguments for direct configuration.
- **Verification Tests:** Added rigorous `runMultiHeadPagedAttention()` and `runGroupedQueryAttention()` CPU/GPU mathematical validations.

### Changed
- **Threadgroup Stability:** Hardened `paged_attention_single` kernel by unifying vector declarations to `uint3` to guarantee compilation stability on Apple Silicon.
- **Engine Memory Allocation:** Scaled all Split-K intermediate buffers by `numHeads` for proper Z-axis memory segmentation.
- **Swift Concurrency:** Enforced strict `.v6` Swift language modes across all data structures and test arrays.

### Removed
- **Dead Code Eradication:** Purged orphaned `paged_attention_half` and `kv_cache_copy` kernels.
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
