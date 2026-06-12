# PagedAttentionMetal — Agent Guide

## Project Overview
High-performance paged attention engine for Apple M-series GPUs (Metal 3.1+). Implements PagedAttention (virtual memory for transformer KV caches) with FlashAttention-style online softmax, GQA/MQA support, sliding window, and hardware simdgroup_matrix MMA operations.

## Key Files
- `Sources/PagedAttentionMetal/kernels.metal` (2097 lines) — All GPU kernels
- `Sources/PagedAttentionMetal/PagedAttentionEngine.swift` (1803 lines) — CPU dispatch, pipeline management
- `Sources/PagedAttentionMetal/PagedAttentionTypes.swift` (432 lines) — Data types, errors, request structs
- `Tests/PagedAttentionMetalTests/PagedAttentionMetalTests.swift` (1020 lines) — 20 tests
- `Benchmarks/Sources/` — Prefill, decode, memory, precision benchmarks
- `Sources/MLXDemo/` — MLX comparison benchmark

## Architecture
- **Kernels**: FlashAttention prefill/decode (device-direct reads, cooperative dot product), paged single-pass/tiled/FP8 prefill, decode, backward, KV cache append, fused append+prefill
- **Engine**: `PagedAttentionEngine` class with `withRetry` logic, `CommandBufferManager` for triple-buffering, automatic kernel selection based on headDim, dataType, seqLen
- **Dispatch**: `prefillFlash` for headDim≤64 (f16/f32), tiled for long sequences, single-pass for short sequences, decode with flash decode for headDim≤128
- **Pipeline**: All 20+ pipelines compiled at engine init from kernels.metal source
- **Build**: `swift build && swift test` / `swift run -c release Benchmarks all`

## Current State (Pre-MMA)
- FlashAttention prefill uses per-thread cooperative `simd_shuffle_xor` dot product (32× less compute than sequential, but still scalar FMAs)
- Decode kernels extended to headDim=128 with cooperative dot
- GQA supported across all kernels
- FP8 quantization kernels present but separate
- ~2.3× slower than MLX at seqLen=4096; remaining gap is lack of hardware MMA

## Hardware Constraints (M4 / Apple7+)
- 20 GPU cores, 1024 max threads/TG, 64 KB tgmem
- Supports `simdgroup_half8x8` and `simdgroup_float8x8` (64 FMAs/instruction vs ~4 scalar → 16× ALU improvement)
- `simdgroup_load` loads directly from device memory (bypasses 23 GB/s tgmem, uses 68 GB/s DRAM)
- 8 simdgroups × 32 threads = 256 threads per threadgroup

## Phase Plan (Remaining)

### Phase 7: flash_attention_mma_f16 — simdgroup_matrix Prefill
- **Kernel**: `flash_attention_mma_f16` (~300 lines added to kernels.metal)
- **Tiling**: BQ=32, BK=16, BD=64, WM=4, WN=2 → 8 simdgroups
- **tgmem**: Q_tile half[32×64]=4KB, S_tile float[32×16]=2KB, O_tile float[32×64]=8KB, m_i[32]+l_i[32]=256B → ~14KB total
- **Q·Kᵀ**: simdgroup_half8x8 × simdgroup_half8x8 → simdgroup_float8x8, iterate D in 8-element chunks
- **Softmax**: tgmem row-wise, 8 threads/row reduce via simd_shuffle_xor(1,2,4)
- **P·V**: simdgroup_half8x8 × simdgroup_half8x8 → simdgroup_float8x8, iterate K×D tiles, 4 D-passes
- **Online softmax persist**: m_i[32], l_i[32] in tgmem, renormalize O_tile per K-block
- **Causal**: block-level early exit + per-element masking in S_tile
- **GQA**: Q-rows split across gqa_ratio heads, rows_per_head=32/gqa_ratio
- **Engine**: Add `flashPrefillPipelineMMA`, `shouldUseMMAPrefill()` for .float16+headDim==64, route in prefillGPU

### Phase 8: flash_decode_mma_f16 — simdgroup_matrix Decode
- Single query (BQ=1), all 8 simdgroups cooperate
- Each simdgroup loads same Q, different K/V tokens
- GQA: all Q-heads for this KV-head processed in parallel

### Phase 9: Function Constants
- Replace `constant uint &causal`, `constant uint &head_dim` with `[[function_constant(n)]]`
- Compile pipeline variants at init time for all needed combos
- Enables compiler loop unrolling and dead code elimination

### Phase 10: Async Command Encoding
- Triple-buffered async: submit GPU work while CPU encodes next layer
- Already partially implemented in CommandBufferManager

### Phase 11: headDim Extension
- BD > 64: split D-dimension across simdgroup passes
- Update dispatch and shouldUseMMAPrefill for headDim≤256

### Phase 12: Validation & Cleanup
- Verify all 20 tests pass with MMA kernels
- Benchmark vs MLX comparison
- Delete old single-pass, tiled, FP8 attention kernels (keep backward, append, fused)

## Agent Instructions
1. **Always research syntax online** when uncertain about Metal API (simdgroup_matrix operations, function constants, etc.)
2. **Build and test after each phase**: `swift build && swift test` must pass
3. **No comments in production code** (Metal or Swift) — keep code clean
4. **Throw on failure** — never silently swallow errors in public APIs
5. **Check all changes** before moving to the next phase — review diff, build, test
6. **Work autonomously** through the phases without manual prompting
7. **Always run lint/typecheck** if available; otherwise at minimum build+test
8. When implementing new kernels, match the existing buffer layout and dispatch patterns exactly
9. Target absolute M-chip utilization: minimize tgmem usage, maximize device-direct reads, use simdgroup_matrix for all matmuls
