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

// MARK: - SIMD Vectorized Dot Product Helpers

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

inline float simd_dot_product_f16(const device half* a, const device half* b, uint head_dim) {
    float sum = 0.0f;
    uint d = 0;
    for (; d + 4 <= head_dim; d += 4) {
        float4 av = float4(*(device const half4*)(a + d));
        float4 bv = float4(*(device const half4*)(b + d));
        sum += dot(av, bv);
    }
    for (; d < head_dim; d++) {
        sum += float(a[d]) * float(b[d]);
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


// MARK: - Paged Attention Single-Pass (Prefill)

// Q layout:      [seq_len, num_heads, head_dim]
// K/V pool layout: [num_physical_blocks, block_size, num_kv_heads, head_dim]
// Output layout: [seq_len, num_heads, head_dim]
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
    constant uint        &block_size   [[buffer(9)]],
    constant uint        &causal       [[buffer(10)]],

    uint3  gid              [[threadgroup_position_in_grid]],
    uint3  lid              [[thread_position_in_threadgroup]],

    threadgroup float *Q_tile   [[threadgroup(0)]],
    threadgroup float *K_tile   [[threadgroup(1)]],
    threadgroup float *V_tile   [[threadgroup(2)]]
) {
    const uint row_in_tile = lid.y;
    const uint col         = lid.x;
    const uint row         = gid.y * block_size + row_in_tile;
    const uint head_idx    = gid.z;
    const uint kv_head_idx = head_idx / (num_heads / num_kv_heads);

    const float scale = 1.0f / sqrt(float(head_dim));

    float m_i   = -INFINITY;
    float l_i   = 0.0f;
    float acc_o = 0.0f;

    if (row < seq_len && col < head_dim) {
        Q_tile[row_in_tile * head_dim + col] = Q[(row * num_heads + head_idx) * head_dim + col];
    } else if (row_in_tile < block_size && col < head_dim) {
        Q_tile[row_in_tile * head_dim + col] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint k_block = 0; k_block < seq_len; k_block += block_size) {
        int logical_block = k_block / block_size;
        int physical_block = block_table[logical_block];

        device const float* K_block_base = K_pool + physical_block * block_size * num_kv_heads * head_dim;
        device const float* V_block_base = V_pool + physical_block * block_size * num_kv_heads * head_dim;

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
            for (uint local_j = 0; local_j < block_size; local_j++) {
                uint global_key_idx = k_block + local_j;
                if (global_key_idx >= seq_len) continue;
                if (causal && global_key_idx > row) continue;

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


kernel void paged_attention_single_f16(
    device const half    *Q            [[buffer(0)]],
    device const half    *K_pool       [[buffer(1)]],
    device const half    *V_pool       [[buffer(2)]],
    device const int     *block_table  [[buffer(3)]],
    device       half    *O            [[buffer(4)]],
    constant uint        &seq_len      [[buffer(5)]],
    constant uint        &head_dim     [[buffer(6)]],
    constant uint        &num_heads    [[buffer(7)]],
    constant uint        &num_kv_heads [[buffer(8)]],
    constant uint        &block_size   [[buffer(9)]],
    constant uint        &causal       [[buffer(10)]],

    uint3  gid              [[threadgroup_position_in_grid]],
    uint3  lid              [[thread_position_in_threadgroup]],

    threadgroup float *Q_tile   [[threadgroup(0)]],
    threadgroup float *K_tile   [[threadgroup(1)]],
    threadgroup float *V_tile   [[threadgroup(2)]]
) {
    const uint row_in_tile = lid.y;
    const uint col         = lid.x;
    const uint row         = gid.y * block_size + row_in_tile;
    const uint head_idx    = gid.z;
    const uint kv_head_idx = head_idx / (num_heads / num_kv_heads);

    const float scale = 1.0f / sqrt(float(head_dim));

    float m_i   = -INFINITY;
    float l_i   = 0.0f;
    float acc_o = 0.0f;

    if (row < seq_len && col < head_dim) {
        Q_tile[row_in_tile * head_dim + col] = float(Q[(row * num_heads + head_idx) * head_dim + col]);
    } else if (row_in_tile < block_size && col < head_dim) {
        Q_tile[row_in_tile * head_dim + col] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint k_block = 0; k_block < seq_len; k_block += block_size) {
        int logical_block = k_block / block_size;
        int physical_block = block_table[logical_block];

        device const half* K_block_base = K_pool + physical_block * block_size * num_kv_heads * head_dim;
        device const half* V_block_base = V_pool + physical_block * block_size * num_kv_heads * head_dim;

        uint local_k_row = lid.y;
        uint global_k_row = k_block + local_k_row;

        if (global_k_row < seq_len && col < head_dim) {
            K_tile[local_k_row * head_dim + col] = float(K_block_base[local_k_row * num_kv_heads * head_dim + kv_head_idx * head_dim + col]);
            V_tile[local_k_row * head_dim + col] = float(V_block_base[local_k_row * num_kv_heads * head_dim + kv_head_idx * head_dim + col]);
        } else {
            K_tile[local_k_row * head_dim + col] = 0.0f;
            V_tile[local_k_row * head_dim + col] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (row < seq_len && col < head_dim) {
            for (uint local_j = 0; local_j < block_size; local_j++) {
                uint global_key_idx = k_block + local_j;
                if (global_key_idx >= seq_len) continue;
                if (causal && global_key_idx > row) continue;

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
        O[(row * num_heads + head_idx) * head_dim + col] = half(acc_o / l_i);
    }
}


// MARK: - Paged Attention V2 Split-K (Two-Pass)

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
    constant uint      &block_size    [[buffer(12)]],
    constant uint      &causal        [[buffer(13)]],
    uint3 gid [[thread_position_in_grid]],
    threadgroup float* acc_o [[threadgroup(0)]]
) {
    const uint row       = gid.y;
    const uint block_idx = gid.x;
    const uint head_idx  = gid.z;

    if (row >= seq_len || block_idx >= num_blocks || head_idx >= num_heads) return;

    const uint kv_head_idx = head_idx / (num_heads / num_kv_heads);

    int physical_block = block_table[block_idx];
    device const float* K_block = K_pool + physical_block * block_size * num_kv_heads * head_dim;
    device const float* V_block = V_pool + physical_block * block_size * num_kv_heads * head_dim;

    const float scale = 1.0f / sqrt(float(head_dim));

    float m_i = -INFINITY;
    float l_i = 0.0f;
    
    for (uint d = 0; d < head_dim; d++) acc_o[d] = 0.0f;

    for (uint local_j = 0; local_j < block_size; local_j++) {
        uint global_j = block_idx * block_size + local_j;
        if (global_j >= seq_len) break;
        if (causal && global_j > row) continue;

        device const float* K_token = K_block + local_j * num_kv_heads * head_dim + kv_head_idx * head_dim;
        device const float* V_token = V_block + local_j * num_kv_heads * head_dim + kv_head_idx * head_dim;

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

    uint partial_idx = (head_idx * seq_len + row) * num_blocks + block_idx;
    partial_m[partial_idx] = m_i;
    partial_l[partial_idx] = l_i;
    for (uint d = 0; d < head_dim; d++) {
        partial_out[partial_idx * head_dim + d] = acc_o[d];
    }
}


kernel void paged_attention_split_phase1_f16(
    device const half  *Q             [[buffer(0)]],
    device const half  *K_pool        [[buffer(1)]],
    device const half  *V_pool        [[buffer(2)]],
    device const int   *block_table   [[buffer(3)]],
    device float       *partial_out   [[buffer(4)]],
    device float       *partial_m     [[buffer(5)]],
    device float       *partial_l     [[buffer(6)]],
    constant uint      &seq_len       [[buffer(7)]],
    constant uint      &head_dim      [[buffer(8)]],
    constant uint      &num_blocks    [[buffer(9)]],
    constant uint      &num_heads     [[buffer(10)]],
    constant uint      &num_kv_heads  [[buffer(11)]],
    constant uint      &block_size    [[buffer(12)]],
    constant uint      &causal        [[buffer(13)]],
    uint3 gid [[thread_position_in_grid]],
    threadgroup float* acc_o [[threadgroup(0)]]
) {
    const uint row       = gid.y;
    const uint block_idx = gid.x;
    const uint head_idx  = gid.z;

    if (row >= seq_len || block_idx >= num_blocks || head_idx >= num_heads) return;

    const uint kv_head_idx = head_idx / (num_heads / num_kv_heads);

    int physical_block = block_table[block_idx];
    device const half* K_block = K_pool + physical_block * block_size * num_kv_heads * head_dim;
    device const half* V_block = V_pool + physical_block * block_size * num_kv_heads * head_dim;

    const float scale = 1.0f / sqrt(float(head_dim));

    float m_i = -INFINITY;
    float l_i = 0.0f;
    
    for (uint d = 0; d < head_dim; d++) acc_o[d] = 0.0f;

    for (uint local_j = 0; local_j < block_size; local_j++) {
        uint global_j = block_idx * block_size + local_j;
        if (global_j >= seq_len) break;
        if (causal && global_j > row) continue;

        device const half* K_token = K_block + local_j * num_kv_heads * head_dim + kv_head_idx * head_dim;
        device const half* V_token = V_block + local_j * num_kv_heads * head_dim + kv_head_idx * head_dim;

        float dot = simd_dot_product_f16(Q + (row * num_heads + head_idx) * head_dim, K_token, head_dim);
        dot *= scale;

        float m_old = m_i;
        m_i = max(m_i, dot);
        float correction = exp(m_old - m_i);
        l_i = l_i * correction + exp(dot - m_i);

        float exp_weight = exp(dot - m_i);
        uint d = 0;
        for (; d + 4 <= head_dim; d += 4) {
            float4 v_vec = float4(*(device const half4*)(V_token + d));
            float4 acc_vec = float4(acc_o[d], acc_o[d+1], acc_o[d+2], acc_o[d+3]);
            acc_vec = acc_vec * correction + exp_weight * v_vec;
            acc_o[d]   = acc_vec.x;
            acc_o[d+1] = acc_vec.y;
            acc_o[d+2] = acc_vec.z;
            acc_o[d+3] = acc_vec.w;
        }
        for (; d < head_dim; d++) {
            acc_o[d] = acc_o[d] * correction + exp_weight * float(V_token[d]);
        }
    }

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

    O[(row * num_heads + head_idx) * head_dim + col] = global_acc / global_l;
}

kernel void paged_attention_split_phase2_f16(
    device const float *partial_out   [[buffer(0)]],
    device const float *partial_m     [[buffer(1)]],
    device const float *partial_l     [[buffer(2)]],
    device half        *O             [[buffer(3)]],
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

    O[(row * num_heads + head_idx) * head_dim + col] = half(global_acc / global_l);
}


// MARK: - Paged Attention Backward Pass

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
    constant uint        &num_heads    [[buffer(12)]],
    constant uint        &num_kv_heads [[buffer(13)]],
    constant uint        &block_size   [[buffer(14)]],
    uint3 gid [[thread_position_in_grid]]
)
{
    const uint d        = gid.x;
    const uint i        = gid.y;
    const uint head_idx = gid.z;

    if (i >= seq_len || d >= head_dim || head_idx >= num_heads) return;

    const uint kv_head_idx = head_idx / (num_heads / num_kv_heads);
    const float scale = 1.0f / sqrt(float(head_dim));
    
    const uint q_offset = (i * num_heads + head_idx) * head_dim;
    const float q_id  = Q[q_offset + d];
    const float dO_id = dO[q_offset + d];
    const float m_i   = m[i * num_heads + head_idx];
    const float l_i   = l[i * num_heads + head_idx];

    const uint num_logical_blocks = (seq_len + block_size - 1) / block_size;

    float rowsum = 0.0f;

    for (uint logical_block = 0; logical_block < num_logical_blocks; logical_block++) {
        int physical_block = block_table[logical_block];
        device const float* K_block = K_pool + physical_block * block_size * num_kv_heads * head_dim;
        device const float* V_block = V_pool + physical_block * block_size * num_kv_heads * head_dim;

        for (uint local_j = 0; local_j < block_size; local_j++) {
            uint j = logical_block * block_size + local_j;
            if (j >= seq_len) break;

            device const float* K_token = K_block + local_j * num_kv_heads * head_dim + kv_head_idx * head_dim;
            device const float* V_token = V_block + local_j * num_kv_heads * head_dim + kv_head_idx * head_dim;

            float dot = simd_dot_product(Q + q_offset, K_token, head_dim);
            dot *= scale;

            float P_ij = exp(dot - m_i) / l_i;
            float dP_ij = simd_dot_product(dO + q_offset, V_token, head_dim);

            rowsum += P_ij * dP_ij;
        }
    }

    float dQ_acc = 0.0f;

    for (uint logical_block = 0; logical_block < num_logical_blocks; logical_block++) {
        int physical_block = block_table[logical_block];
        device const float* K_block = K_pool + physical_block * block_size * num_kv_heads * head_dim;
        device const float* V_block = V_pool + physical_block * block_size * num_kv_heads * head_dim;

        for (uint local_j = 0; local_j < block_size; local_j++) {
            uint j = logical_block * block_size + local_j;
            if (j >= seq_len) break;

            device const float* K_token = K_block + local_j * num_kv_heads * head_dim + kv_head_idx * head_dim;
            device const float* V_token = V_block + local_j * num_kv_heads * head_dim + kv_head_idx * head_dim;

            float dot = simd_dot_product(Q + q_offset, K_token, head_dim);
            dot *= scale;

            float P_ij = exp(dot - m_i) / l_i;
            float dP_ij = simd_dot_product(dO + q_offset, V_token, head_dim);
            float dS_ij = P_ij * (dP_ij - rowsum);

            dQ_acc += dS_ij * K_token[d] * scale;

            float dK_contrib = dS_ij * q_id * scale;
            uint dK_offset = physical_block * block_size * num_kv_heads * head_dim + local_j * num_kv_heads * head_dim + kv_head_idx * head_dim + d;
            atomic_add_float(&dK_pool[dK_offset], dK_contrib);

            float dV_contrib = P_ij * dO_id;
            uint dV_offset = physical_block * block_size * num_kv_heads * head_dim + local_j * num_kv_heads * head_dim + kv_head_idx * head_dim + d;
            atomic_add_float(&dV_pool[dV_offset], dV_contrib);
        }
    }

    dQ[q_offset + d] = dQ_acc;
}


// MARK: - Paged Decode (Single Token Generation)

// Optimized kernel for generating one new token per sequence
// Q layout: [batch, num_heads, head_dim]
// Output layout: [batch, num_heads, head_dim]
kernel void paged_decode_single(
    device const float *Q                [[buffer(0)]],
    device const float *K_pool           [[buffer(1)]],
    device const float *V_pool           [[buffer(2)]],
    device const int   *block_tables     [[buffer(3)]],
    device const uint  *seq_lengths      [[buffer(4)]],
    device float       *O                [[buffer(5)]],
    constant uint      &batch_size       [[buffer(6)]],
    constant uint      &head_dim         [[buffer(7)]],
    constant uint      &num_heads        [[buffer(8)]],
    constant uint      &num_kv_heads     [[buffer(9)]],
    constant uint      &block_size       [[buffer(10)]],
    constant uint      &max_num_blocks   [[buffer(11)]],
    uint3 gid [[thread_position_in_grid]]
) {
    const uint d         = gid.x;
    const uint head_idx  = gid.y;
    const uint batch_idx = gid.z;

    if (d >= head_dim || head_idx >= num_heads || batch_idx >= batch_size) return;

    const uint kv_head_idx = head_idx / (num_heads / num_kv_heads);
    const uint seq_len = seq_lengths[batch_idx];
    const float scale = 1.0f / sqrt(float(head_dim));

    device const int* block_table = block_tables + batch_idx * max_num_blocks;
    device const float* q_vec = Q + (batch_idx * num_heads + head_idx) * head_dim;

    float m_i = -INFINITY;
    float l_i = 0.0f;
    float acc_o = 0.0f;

    uint num_logical_blocks = (seq_len + block_size - 1) / block_size;

    for (uint logical_block = 0; logical_block < num_logical_blocks; logical_block++) {
        int physical_block = block_table[logical_block];
        device const float* K_block = K_pool + physical_block * block_size * num_kv_heads * head_dim;
        device const float* V_block = V_pool + physical_block * block_size * num_kv_heads * head_dim;

        uint block_start = logical_block * block_size;
        uint block_end = min(block_start + block_size, seq_len);

        for (uint j = block_start; j < block_end; j++) {
            uint local_j = j - block_start;
            device const float* K_token = K_block + local_j * num_kv_heads * head_dim + kv_head_idx * head_dim;
            device const float* V_token = V_block + local_j * num_kv_heads * head_dim + kv_head_idx * head_dim;

            float dot = simd_dot_product(q_vec, K_token, head_dim);
            dot *= scale;

            float m_old = m_i;
            m_i = max(m_i, dot);
            float correction = exp(m_old - m_i);
            l_i = l_i * correction + exp(dot - m_i);
            acc_o = acc_o * correction + exp(dot - m_i) * V_token[d];
        }
    }

    O[(batch_idx * num_heads + head_idx) * head_dim + d] = acc_o / l_i;
}

kernel void paged_decode_single_f16(
    device const half  *Q                [[buffer(0)]],
    device const half  *K_pool           [[buffer(1)]],
    device const half  *V_pool           [[buffer(2)]],
    device const int   *block_tables     [[buffer(3)]],
    device const uint  *seq_lengths      [[buffer(4)]],
    device half        *O                [[buffer(5)]],
    constant uint      &batch_size       [[buffer(6)]],
    constant uint      &head_dim         [[buffer(7)]],
    constant uint      &num_heads        [[buffer(8)]],
    constant uint      &num_kv_heads     [[buffer(9)]],
    constant uint      &block_size       [[buffer(10)]],
    constant uint      &max_num_blocks   [[buffer(11)]],
    uint3 gid [[thread_position_in_grid]]
) {
    const uint d         = gid.x;
    const uint head_idx  = gid.y;
    const uint batch_idx = gid.z;

    if (d >= head_dim || head_idx >= num_heads || batch_idx >= batch_size) return;

    const uint kv_head_idx = head_idx / (num_heads / num_kv_heads);
    const uint seq_len = seq_lengths[batch_idx];
    const float scale = 1.0f / sqrt(float(head_dim));

    device const int* block_table = block_tables + batch_idx * max_num_blocks;
    device const half* q_vec = Q + (batch_idx * num_heads + head_idx) * head_dim;

    float m_i = -INFINITY;
    float l_i = 0.0f;
    float acc_o = 0.0f;

    uint num_logical_blocks = (seq_len + block_size - 1) / block_size;

    for (uint logical_block = 0; logical_block < num_logical_blocks; logical_block++) {
        int physical_block = block_table[logical_block];
        device const half* K_block = K_pool + physical_block * block_size * num_kv_heads * head_dim;
        device const half* V_block = V_pool + physical_block * block_size * num_kv_heads * head_dim;

        uint block_start = logical_block * block_size;
        uint block_end = min(block_start + block_size, seq_len);

        for (uint j = block_start; j < block_end; j++) {
            uint local_j = j - block_start;
            device const half* K_token = K_block + local_j * num_kv_heads * head_dim + kv_head_idx * head_dim;
            device const half* V_token = V_block + local_j * num_kv_heads * head_dim + kv_head_idx * head_dim;

            float dot = simd_dot_product_f16(q_vec, K_token, head_dim);
            dot *= scale;

            float m_old = m_i;
            m_i = max(m_i, dot);
            float correction = exp(m_old - m_i);
            l_i = l_i * correction + exp(dot - m_i);
            acc_o = acc_o * correction + exp(dot - m_i) * float(V_token[d]);
        }
    }

    O[(batch_idx * num_heads + head_idx) * head_dim + d] = half(acc_o / l_i);
}


// MARK: - KV Cache Append (GPU-side cache write)

// Writes new K/V tokens into the paged pool
// new_keys/values layout: [num_new_tokens, num_kv_heads, head_dim]
kernel void kv_cache_append(
    device const float *new_keys       [[buffer(0)]],
    device const float *new_values     [[buffer(1)]],
    device float       *K_pool         [[buffer(2)]],
    device float       *V_pool         [[buffer(3)]],
    device const int   *block_table    [[buffer(4)]],
    constant uint      &token_offset   [[buffer(5)]],
    constant uint      &num_new_tokens [[buffer(6)]],
    constant uint      &num_kv_heads   [[buffer(7)]],
    constant uint      &head_dim       [[buffer(8)]],
    constant uint      &block_size     [[buffer(9)]],
    uint3 gid [[thread_position_in_grid]]
) {
    const uint d           = gid.x;
    const uint kv_head_idx = gid.y;
    const uint token_idx   = gid.z;

    if (d >= head_dim || kv_head_idx >= num_kv_heads || token_idx >= num_new_tokens) return;

    uint global_token_pos = token_offset + token_idx;
    uint logical_block = global_token_pos / block_size;
    uint local_token_pos = global_token_pos % block_size;
    
    int physical_block = block_table[logical_block];
    
    uint pool_offset = physical_block * block_size * num_kv_heads * head_dim 
                     + local_token_pos * num_kv_heads * head_dim 
                     + kv_head_idx * head_dim 
                     + d;
    
    uint input_offset = token_idx * num_kv_heads * head_dim + kv_head_idx * head_dim + d;
    
    K_pool[pool_offset] = new_keys[input_offset];
    V_pool[pool_offset] = new_values[input_offset];
}

kernel void kv_cache_append_f16(
    device const half  *new_keys       [[buffer(0)]],
    device const half  *new_values     [[buffer(1)]],
    device half        *K_pool         [[buffer(2)]],
    device half        *V_pool         [[buffer(3)]],
    device const int   *block_table    [[buffer(4)]],
    constant uint      &token_offset   [[buffer(5)]],
    constant uint      &num_new_tokens [[buffer(6)]],
    constant uint      &num_kv_heads   [[buffer(7)]],
    constant uint      &head_dim       [[buffer(8)]],
    constant uint      &block_size     [[buffer(9)]],
    uint3 gid [[thread_position_in_grid]]
) {
    const uint d           = gid.x;
    const uint kv_head_idx = gid.y;
    const uint token_idx   = gid.z;

    if (d >= head_dim || kv_head_idx >= num_kv_heads || token_idx >= num_new_tokens) return;

    uint global_token_pos = token_offset + token_idx;
    uint logical_block = global_token_pos / block_size;
    uint local_token_pos = global_token_pos % block_size;
    
    int physical_block = block_table[logical_block];
    
    uint pool_offset = physical_block * block_size * num_kv_heads * head_dim 
                     + local_token_pos * num_kv_heads * head_dim 
                     + kv_head_idx * head_dim 
                     + d;
    
    uint input_offset = token_idx * num_kv_heads * head_dim + kv_head_idx * head_dim + d;
    
    K_pool[pool_offset] = new_keys[input_offset];
    V_pool[pool_offset] = new_values[input_offset];
}
