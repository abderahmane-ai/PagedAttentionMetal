# ``PagedAttentionMetal``

High-performance PagedAttention implementation for Apple Silicon GPUs using Metal.

## Overview

PagedAttentionMetal provides a GPU-accelerated implementation of the PagedAttention algorithm,
designed for efficient LLM inference on Apple Silicon (M1–M5) GPUs. It manages non-contiguous KV
cache memory in fixed-size blocks, enabling:

- Zero-copy shared memory between CPU and GPU
- Efficient memory utilization via paged block tables
- Batch processing with grouped-query attention (GQA/MQA)
- Multiple precision modes (FP32, FP16, FP8)

## Topics

### Essentials

- ``PagedAttentionEngine``
- ``PagedAttentionError``
- ``PagedAttentionDataType``

### KV Cache Management

- ``KVCacheManager``
- ``BatchKVCacheManager``
- ``BlockEvictionPolicy``
- ``KVCacheError``

### Configuration

- ``PagedLayerSpec``
- ``PagedAttentionPrefillRequest``
- ``PagedAttentionDecodeRequest``
- ``PagedAttentionFusedPrefillRequest``
- ``PagedKVAppendRequest``
- ``PagedAttentionStats``

### Scheduling & Inference

- ``ContinuousBatchingScheduler``
- ``PagedAttentionInference``
- ``SequenceStatus``

### MLX Integration

- ``PagedMetalKVCache``
- ``PagedAttentionMLXError``

### C API

- ``PAMCache``
- ``PAMContext``
