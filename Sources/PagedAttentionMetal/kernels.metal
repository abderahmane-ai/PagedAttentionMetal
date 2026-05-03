#include <metal_stdlib>
using namespace metal;

// MARK: - Atomic Add Helper (for float)

#if __METAL_VERSION__ >= 300
void atomic_add_float(device float* ptr, float value) {
    atomic_fetch_add_explicit((device atomic_float*)ptr, value, memory_order_relaxed);
}
#else
void atomic_add_float(device float* ptr, float value) {
    volatile device atomic_int* aptr = (volatile device atomic_int*)ptr;
    int old_val;
    int new_val;
    do {
        old_val = atomic_load_explicit(aptr, memory_order_relaxed);
        union { int i; float f; } u;
        u.i = old_val;
        u.f += value;
        new_val = u.i;
    } while(!atomic_compare_exchange_weak_explicit(aptr, &old_val, new_val,
                                                   memory_order_relaxed, memory_order_relaxed));
}
#endif

// MARK: - Constants

constant uint TILE_SIZE = 16;
constant uint BLOCK_SIZE = 16;

// MARK: - SIMD Vectorized Dot Product Helper

inline float simd_dot_product(const device float* a, const device float* b, uint head_dim) {
    float sum = 0.0f;
    uint d = 0;
    for (; d + 4 <= head_dim; d += 4) {
        float4 av = float4(a[d], a[d+1], a[d+2], a[d+3]);
        float4 bv = float4(b[d], b[d+1], b[d+2], b[d+3]);
        sum += dot(av, bv);
    }
    for (; d < head_dim; d++) {
        sum += a[d] * b[d];
    }
    return sum;
}

inline float simd_dot_tile(const threadgroup float* a, uint a_offset, const threadgroup float* b, uint b_offset, uint head_dim) {
    float sum = 0.0f;
    uint d = 0;
    for (; d + 4 <= head_dim; d += 4) {
        float4 av = float4(a[a_offset + d], a[a_offset + d + 1], a[a_offset + d + 2], a[a_offset + d + 3]);
        float4 bv = float4(b[b_offset + d], b[b_offset + d + 1], b[b_offset + d + 2], b[b_offset + d + 3]);
        sum += dot(av, bv);
    }
    for (; d < head_dim; d++) {
        sum += a[a_offset + d] * b[b_offset + d];
    }
    return sum;
}

// MARK: - Vector Kernels (SIMD float4)

kernel void vector_add(
    const device float* A [[buffer(0)]],
    const device float* B [[buffer(1)]],
    device float*       C [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    uint idx = id * 4;
    float4 a = float4(A[idx], A[idx+1], A[idx+2], A[idx+3]);
    float4 b = float4(B[idx], B[idx+1], B[idx+2], B[idx+3]);
    float4 c = a + b;
    C[idx] = c.x; C[idx+1] = c.y; C[idx+2] = c.z; C[idx+3] = c.w;
}

kernel void vector_multiply(
    const device float* A [[buffer(0)]],
    const device float* B [[buffer(1)]],
    device float*       C [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    uint idx = id * 4;
    float4 a = float4(A[idx], A[idx+1], A[idx+2], A[idx+3]);
    float4 b = float4(B[idx], B[idx+1], B[idx+2], B[idx+3]);
    float4 c = a * b;
    C[idx] = c.x; C[idx+1] = c.y; C[idx+2] = c.z; C[idx+3] = c.w;
}

// MARK: - Matrix Kernels (SIMD float4)

kernel void matrix_add(
    const device float* A [[buffer(0)]],
    const device float* B [[buffer(1)]],
    device float*       C [[buffer(2)]],
    constant uint      &cols [[buffer(3)]],
    uint2 gid       [[thread_position_in_grid]],
    uint2 grid_size [[threads_per_grid]]
) {
    // Each thread processes 4 consecutive elements in the row via simd_float4
    // gid.x is column group (0 to cols/4-1), gid.y is row
    uint col_start = gid.x * 4;
    uint row = gid.y;
    uint idx = row * cols + col_start;

    float4 a = float4(A[idx], A[idx + 1], A[idx + 2], A[idx + 3]);
    float4 b = float4(B[idx], B[idx + 1], B[idx + 2], B[idx + 3]);
    float4 c = a + b;
    C[idx] = c.x; C[idx + 1] = c.y; C[idx + 2] = c.z; C[idx + 3] = c.w;
}

kernel void matrix_hadamard(
    const device float* A [[buffer(0)]],
    const device float* B [[buffer(1)]],
    device float*       C [[buffer(2)]],
    constant uint      &cols [[buffer(3)]],
    uint2 gid       [[thread_position_in_grid]],
    uint2 grid_size [[threads_per_grid]]
) {
    // Each thread processes 4 consecutive elements in the row via simd_float4
    uint col_start = gid.x * 4;
    uint row = gid.y;
    uint idx = row * cols + col_start;

    float4 a = float4(A[idx], A[idx + 1], A[idx + 2], A[idx + 3]);
    float4 b = float4(B[idx], B[idx + 1], B[idx + 2], B[idx + 3]);
    float4 c = a * b;
    C[idx] = c.x; C[idx + 1] = c.y; C[idx + 2] = c.z; C[idx + 3] = c.w;
}

// MARK: - Threadgroup Memory Kernels

kernel void row_sum(
    const device float* A        [[buffer(0)]],
    device float*       sums     [[buffer(1)]],
    constant uint&      cols     [[buffer(2)]],
    uint2               gid      [[thread_position_in_grid]],
    uint2               tid      [[thread_position_in_threadgroup]],
    uint2               tgid     [[threadgroup_position_in_grid]],
    threadgroup float*  shared_row [[threadgroup(0)]]
) {
    uint local_col = tid.x;
    uint row = tgid.y;

    if (local_col < cols) {
        shared_row[local_col] = A[row * cols + local_col];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float sum = 0.0f;
    for (uint i = local_col; i < cols; i += 64) {
        sum += shared_row[i];
    }

    shared_row[local_col] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Note: MSL requires uint for indexing, not size_t.
    // The reduction guard must compare against the threadgroup width (64),
    // not `cols`, because shared_row was already populated for all 64 slots.
    uint stride = 64 / 2;
    while (stride > 0) {
        if (local_col < stride && (local_col + stride) < 64) {
            shared_row[local_col] += shared_row[local_col + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        stride /= 2;
    }

    if (tid.x == 0) {
        sums[row] = shared_row[0];
    }
}

// MARK: - Online Softmax Rows

kernel void online_softmax_rows(
    const device float* input   [[buffer(0)]],
    device float*       output  [[buffer(1)]],
    constant uint&      cols    [[buffer(2)]],
    uint2               tid     [[thread_position_in_threadgroup]],
    uint2               tgid    [[threadgroup_position_in_grid]],
    threadgroup float*  local_block [[threadgroup(0)]]
) {
    const uint row = tgid.y;
    const uint block_size = 32;

    float m_old = -INFINITY;
    float l_old = 0.0f;

    for (uint start = 0; start < cols; start += block_size) {
        uint col = start + tid.x;
        if (col < cols) {
            local_block[tid.x] = input[row * cols + col];
        } else {
            local_block[tid.x] = -INFINITY;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = block_size / 2; stride > 0; stride /= 2) {
            if (tid.x < stride) {
                local_block[tid.x] = max(local_block[tid.x], local_block[tid.x + stride]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        float block_max = local_block[0];

        uint load_idx = start + tid.x;
        if (load_idx < cols) {
            local_block[tid.x] = input[row * cols + load_idx];
        } else {
            local_block[tid.x] = -INFINITY;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        local_block[tid.x] = exp(local_block[tid.x] - block_max);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = block_size / 2; stride > 0; stride /= 2) {
            if (tid.x < stride) {
                local_block[tid.x] += local_block[tid.x + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        float block_sum_exp = local_block[0];

        float m_new = max(m_old, block_max);

        // Handle edge case where both m_old and block_max are -infinity (all-padding block)
        // exp(-inf - -inf) = exp(NaN) which gives NaN. Use 0.0 in this case.
        float l_old_scaled = (m_old == -INFINITY && block_max == -INFINITY) ? 0.0f : l_old * exp(m_old - m_new);
        float l_new = l_old_scaled + block_sum_exp * exp(block_max - m_new);

        m_old = m_new;
        l_old = l_new;
    }

    if (tid.x == 0) {
        output[row * cols + 0] = m_old;
        output[row * cols + 1] = l_old;
    }
}

// MARK: - Naive Attention (Phase 5)

kernel void naive_attention(
    device const float *Q   [[buffer(0)]],
    device const float *K   [[buffer(1)]],
    device const float *V   [[buffer(2)]],
    device float       *O   [[buffer(3)]],
    constant uint      &seq_len   [[buffer(4)]],
    constant uint      &head_dim  [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint row = gid.y;
    const uint col = gid.x;

    if (row >= seq_len || col >= head_dim) return;

    float m_i = -INFINITY;
    float l_i = 0.0f;
    float acc_o = 0.0f;

    float scale = 1.0f / sqrt(float(head_dim));

    for (uint j = 0; j < seq_len; j++) {
        // SIMD vectorized dot product
        float dot = simd_dot_product(Q + row * head_dim, K + j * head_dim, head_dim);
        dot *= scale;

        float m_old = m_i;
        m_i = max(m_i, dot);
        float correction = exp(m_old - m_i);
        l_i = l_i * correction + exp(dot - m_i);
        acc_o *= correction;
        acc_o += V[j * head_dim + col] * exp(dot - m_i);
    }

    O[row * head_dim + col] = acc_o / l_i;
}

// MARK: - Tiled Flash Attention (Phase 6)

kernel void flash_attention_forward(
    device const float *Q   [[buffer(0)]],
    device const float *K   [[buffer(1)]],
    device const float *V   [[buffer(2)]],
    device       float *O   [[buffer(3)]],
    constant uint  &seq_len  [[buffer(4)]],
    constant uint  &head_dim [[buffer(5)]],

    uint2  gid          [[threadgroup_position_in_grid]],
    uint2  lid          [[thread_position_in_threadgroup]],

    threadgroup float *Q_tile [[threadgroup(0)]],
    threadgroup float *K_tile [[threadgroup(1)]],
    threadgroup float *V_tile [[threadgroup(2)]]
) {
    const uint row_in_tile = lid.y;
    const uint col        = lid.x;
    const uint row        = gid.y * TILE_SIZE + row_in_tile;

    const float scale = 1.0f / sqrt(float(head_dim));

    float m_i   = -INFINITY;
    float l_i   = 0.0f;
    float acc_o = 0.0f;

    // Load Q tile with bounds check
    if (row < seq_len && col < head_dim) {
        Q_tile[row_in_tile * head_dim + col] = Q[row * head_dim + col];
    } else {
        Q_tile[row_in_tile * head_dim + col] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint k_tile_start = 0; k_tile_start < seq_len; k_tile_start += TILE_SIZE) {
        uint local_k_row = lid.y;
        uint global_k_row = k_tile_start + local_k_row;

        if (global_k_row < seq_len && col < head_dim) {
            K_tile[local_k_row * head_dim + col] = K[global_k_row * head_dim + col];
            V_tile[local_k_row * head_dim + col] = V[global_k_row * head_dim + col];
        } else {
            K_tile[local_k_row * head_dim + col] = 0.0f;
            V_tile[local_k_row * head_dim + col] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (row < seq_len && col < head_dim) {
            for (uint local_j = 0; local_j < TILE_SIZE; local_j++) {
                uint global_key_idx = k_tile_start + local_j;
                if (global_key_idx >= seq_len) continue;

                // SIMD vectorized dot product
                float dot = simd_dot_tile(
                    Q_tile, row_in_tile * head_dim,
                    K_tile, local_j * head_dim,
                    head_dim
                );
                dot *= scale;

                float m_old = m_i;
                m_i = max(m_i, dot);
                float correction = exp(m_old - m_i);
                l_i = l_i * correction + exp(dot - m_i);
                acc_o = acc_o * correction;
                acc_o += exp(dot - m_i) * V_tile[local_j * head_dim + col];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < seq_len && col < head_dim) {
        O[row * head_dim + col] = acc_o / l_i;
    }
}

// MARK: - Paged Attention (Phase 7)

// Q layout:      [seq_len, num_heads, head_dim]
// K/V pool layout: [num_physical_blocks, block_size, num_kv_heads, head_dim]
// Output layout: [seq_len, num_heads, head_dim]
// Grid:          (width=num_q_tiles, height=seq_len/block_size, depth=num_heads)
kernel void paged_attention_single(
    device const float   *Q            [[buffer(0)]],
    device const float   *K_pool       [[buffer(1)]],
    device const float   *V_pool       [[buffer(2)]],
    device const int     *block_table  [[buffer(3)]],
    device       float   *O            [[buffer(4)]],
    constant uint        &seq_len      [[buffer(5)]],
    constant uint        &head_dim     [[buffer(6)]],
    constant uint        &num_heads    [[buffer(7)]],
    constant uint        &num_kv_heads [[buffer(8)]],

    uint3  gid              [[threadgroup_position_in_grid]],
    uint3  lid              [[thread_position_in_threadgroup]],

    threadgroup float *Q_tile   [[threadgroup(0)]],
    threadgroup float *K_tile   [[threadgroup(1)]],
    threadgroup float *V_tile   [[threadgroup(2)]]
) {
    const uint row_in_tile = lid.y;
    const uint col         = lid.x;
    const uint row         = gid.y * BLOCK_SIZE + row_in_tile;
    const uint head_idx    = gid.z;
    const uint kv_head_idx = head_idx / (num_heads / num_kv_heads);

    const float scale = 1.0f / sqrt(float(head_dim));

    float m_i   = -INFINITY;
    float l_i   = 0.0f;
    float acc_o = 0.0f;

    // Guard against out-of-bounds access for misaligned sequences
    if (row < seq_len && col < head_dim) {
        Q_tile[row_in_tile * head_dim + col] = Q[(row * num_heads + head_idx) * head_dim + col];
    } else if (row_in_tile < BLOCK_SIZE && col < head_dim) {
        Q_tile[row_in_tile * head_dim + col] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint k_block = 0; k_block < seq_len; k_block += BLOCK_SIZE) {
        int logical_block = k_block / BLOCK_SIZE;
        int physical_block = block_table[logical_block];

        // Base of this physical block across all KV heads
        device const float* K_block_base = K_pool + physical_block * BLOCK_SIZE * num_kv_heads * head_dim;
        device const float* V_block_base = V_pool + physical_block * BLOCK_SIZE * num_kv_heads * head_dim;

        uint local_k_row = lid.y;
        uint global_k_row = k_block + local_k_row;

        if (global_k_row < seq_len && col < head_dim) {
            K_tile[local_k_row * head_dim + col] = K_block_base[local_k_row * num_kv_heads * head_dim + kv_head_idx * head_dim + col];
            V_tile[local_k_row * head_dim + col] = V_block_base[local_k_row * num_kv_heads * head_dim + kv_head_idx * head_dim + col];
        } else {
            K_tile[local_k_row * head_dim + col] = 0.0f;
            V_tile[local_k_row * head_dim + col] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (row < seq_len && col < head_dim) {
            for (uint local_j = 0; local_j < BLOCK_SIZE; local_j++) {
                uint global_key_idx = k_block + local_j;
                if (global_key_idx >= seq_len) continue;

                // SIMD vectorized dot product over the tile
                float dot = simd_dot_tile(
                    Q_tile, row_in_tile * head_dim,
                    K_tile, local_j * head_dim,
                    head_dim
                );
                dot *= scale;

                float m_old = m_i;
                m_i = max(m_i, dot);
                float correction = exp(m_old - m_i);
                l_i = l_i * correction + exp(dot - m_i);
                acc_o = acc_o * correction;
                acc_o += exp(dot - m_i) * V_tile[local_j * head_dim + col];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < seq_len && col < head_dim) {
        O[(row * num_heads + head_idx) * head_dim + col] = acc_o / l_i;
    }
}

// MARK: - Paged Attention Backward (Phase 8)

kernel void paged_attention_backward(
    device const float   *Q            [[buffer(0)]],
    device const float   *K_pool       [[buffer(1)]],
    device const float   *V_pool       [[buffer(2)]],
    device const int     *block_table  [[buffer(3)]],
    device const float   *dO           [[buffer(4)]],
    device const float   *m            [[buffer(5)]],
    device const float   *l            [[buffer(6)]],
    device       float   *dQ           [[buffer(7)]],
    device       float   *dK_pool      [[buffer(8)]],
    device       float   *dV_pool      [[buffer(9)]],
    constant uint        &seq_len      [[buffer(10)]],
    constant uint        &head_dim     [[buffer(11)]],
    uint2 gid [[thread_position_in_grid]]
)
{
    const uint i = gid.y;
    const uint d = gid.x;

    if (i >= seq_len || d >= head_dim) return;

    const float scale = 1.0f / sqrt(float(head_dim));
    const float q_id  = Q[i * head_dim + d];
    const float dO_id = dO[i * head_dim + d];
    const float m_i   = m[i];
    const float l_i   = l[i];

    const uint num_logical_blocks = (seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE;

    float rowsum = 0.0f;

    for (uint logical_block = 0; logical_block < num_logical_blocks; logical_block++) {
        int physical_block = block_table[logical_block];
        device const float* K_block = K_pool + physical_block * BLOCK_SIZE * head_dim;
        device const float* V_block = V_pool + physical_block * BLOCK_SIZE * head_dim;

        for (uint local_j = 0; local_j < BLOCK_SIZE; local_j++) {
            uint j = logical_block * BLOCK_SIZE + local_j;
            if (j >= seq_len) break;

            // SIMD vectorized dot products
            float dot = simd_dot_product(Q + i * head_dim, K_block + local_j * head_dim, head_dim);
            dot *= scale;

            float P_ij = exp(dot - m_i) / l_i;

            float dP_ij = simd_dot_product(dO + i * head_dim, V_block + local_j * head_dim, head_dim);

            rowsum += P_ij * dP_ij;
        }
    }

    float dQ_acc = 0.0f;

    for (uint logical_block = 0; logical_block < num_logical_blocks; logical_block++) {
        int physical_block = block_table[logical_block];
        device const float* K_block = K_pool + physical_block * BLOCK_SIZE * head_dim;
        device const float* V_block = V_pool + physical_block * BLOCK_SIZE * head_dim;

        for (uint local_j = 0; local_j < BLOCK_SIZE; local_j++) {
            uint j = logical_block * BLOCK_SIZE + local_j;
            if (j >= seq_len) break;

            // SIMD vectorized dot products
            float dot = simd_dot_product(Q + i * head_dim, K_block + local_j * head_dim, head_dim);
            dot *= scale;

            float P_ij = exp(dot - m_i) / l_i;

            float dP_ij = simd_dot_product(dO + i * head_dim, V_block + local_j * head_dim, head_dim);

            float dS_ij = P_ij * (dP_ij - rowsum);

            dQ_acc += dS_ij * K_block[local_j * head_dim + d] * scale;

            float dK_contrib = dS_ij * q_id * scale;
            atomic_add_float(&dK_pool[physical_block * BLOCK_SIZE * head_dim + local_j * head_dim + d], dK_contrib);

            float dV_contrib = P_ij * dO_id;
            atomic_add_float(&dV_pool[physical_block * BLOCK_SIZE * head_dim + local_j * head_dim + d], dV_contrib);
        }
    }

    // Note: No threadgroup barrier needed here as dQ computation does not depend on atomic dK/dV writes
    dQ[i * head_dim + d] = dQ_acc;
}

// MARK: - Paged Attention V2 (Two-Pass)

kernel void paged_attention_split_phase1(
    device const float *Q             [[buffer(0)]],
    device const float *K_pool        [[buffer(1)]],
    device const float *V_pool        [[buffer(2)]],
    device const int   *block_table   [[buffer(3)]],
    device float       *partial_out   [[buffer(4)]],
    device float       *partial_m     [[buffer(5)]],
    device float       *partial_l     [[buffer(6)]],
    constant uint      &seq_len       [[buffer(7)]],
    constant uint      &head_dim      [[buffer(8)]],
    constant uint      &num_blocks    [[buffer(9)]],
    constant uint      &num_heads     [[buffer(10)]],
    constant uint      &num_kv_heads  [[buffer(11)]],
    uint3 gid [[thread_position_in_grid]]
) {
    const uint row       = gid.y;
    const uint block_idx = gid.x;
    const uint head_idx  = gid.z;

    if (row >= seq_len || block_idx >= num_blocks || head_idx >= num_heads) return;

    const uint kv_head_idx = head_idx / (num_heads / num_kv_heads);

    int physical_block = block_table[block_idx];
    // K/V pool layout: [num_physical_blocks, block_size, num_kv_heads, head_dim]
    device const float* K_block = K_pool + physical_block * BLOCK_SIZE * num_kv_heads * head_dim;
    device const float* V_block = V_pool + physical_block * BLOCK_SIZE * num_kv_heads * head_dim;

    const float scale = 1.0f / sqrt(float(head_dim));

    float m_i = -INFINITY;
    float l_i = 0.0f;
    float acc_o[128]; // max head_dim = 128
    for (uint d = 0; d < head_dim; d++) acc_o[d] = 0.0f;

    for (uint local_j = 0; local_j < BLOCK_SIZE; local_j++) {
        uint global_j = block_idx * BLOCK_SIZE + local_j;
        if (global_j >= seq_len) break;

        device const float* K_token = K_block + local_j * num_kv_heads * head_dim + kv_head_idx * head_dim;
        device const float* V_token = V_block + local_j * num_kv_heads * head_dim + kv_head_idx * head_dim;

        // Q layout: [seq_len, num_heads, head_dim]
        float dot = simd_dot_product(Q + (row * num_heads + head_idx) * head_dim, K_token, head_dim);
        dot *= scale;

        float m_old = m_i;
        m_i = max(m_i, dot);
        float correction = exp(m_old - m_i);
        l_i = l_i * correction + exp(dot - m_i);

        for (uint d = 0; d < head_dim; d++) {
            acc_o[d] = acc_o[d] * correction + exp(dot - m_i) * V_token[d];
        }
    }

    // Partial tensors layout: [num_heads, seq_len, num_blocks]
    uint partial_idx = (head_idx * seq_len + row) * num_blocks + block_idx;
    partial_m[partial_idx] = m_i;
    partial_l[partial_idx] = l_i;
    for (uint d = 0; d < head_dim; d++) {
        partial_out[partial_idx * head_dim + d] = acc_o[d];
    }
}

kernel void paged_attention_split_phase2(
    device const float *partial_out   [[buffer(0)]],
    device const float *partial_m     [[buffer(1)]],
    device const float *partial_l     [[buffer(2)]],
    device float       *O             [[buffer(3)]],
    constant uint      &seq_len       [[buffer(4)]],
    constant uint      &head_dim      [[buffer(5)]],
    constant uint      &num_blocks    [[buffer(6)]],
    constant uint      &num_heads     [[buffer(7)]],
    uint3 gid [[thread_position_in_grid]]
) {
    const uint row      = gid.y;
    const uint col      = gid.x;
    const uint head_idx = gid.z;

    if (row >= seq_len || col >= head_dim || head_idx >= num_heads) return;

    float global_m = -INFINITY;
    float global_l = 0.0f;
    float global_acc = 0.0f;

    for (uint block_idx = 0; block_idx < num_blocks; block_idx++) {
        // Partial tensors layout: [num_heads, seq_len, num_blocks]
        uint partial_idx = (head_idx * seq_len + row) * num_blocks + block_idx;
        float p_m = partial_m[partial_idx];
        float p_l = partial_l[partial_idx];
        float p_out = partial_out[partial_idx * head_dim + col];

        if (p_m == -INFINITY) continue;

        float m_old = global_m;
        global_m = max(global_m, p_m);
        float correction = exp(m_old - global_m);

        global_l = global_l * correction + p_l * exp(p_m - global_m);
        global_acc = global_acc * correction + p_out * exp(p_m - global_m);
    }

    // Output layout: [seq_len, num_heads, head_dim]
    O[(row * num_heads + head_idx) * head_dim + col] = global_acc / global_l;
}