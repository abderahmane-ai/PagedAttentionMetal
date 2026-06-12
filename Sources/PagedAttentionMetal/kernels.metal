#include <metal_stdlib>
using namespace metal;

// MARK: - FlashAttention Prefill (GQA-aware, Direct Device Reads)

// Grid: (num_kv_heads, (seq_len + 31) / 32, 1)
// TG: 256 threads = 8 simdgroups × 32 threads
//   Each simdgroup: 4 Q-rows per Q-head in the GQA group
//   Each thread: 8 D-elements
// Total Q-rows per TG: 32, distributed across gqa_ratio heads
// GQA: one TG handles all Q-heads sharing one KV-head → KV reads × GQA-ratio
// No threadgroup memory — direct device reads bypass 23 GB/s tgmem bottleneck

static void fa_load8_half(device const half* base, uint off, thread float* v) {
    half4 v0 = *(device const half4*)(base + off);
    half4 v1 = *(device const half4*)(base + off + 4);
    v[0] = float(v0[0]); v[1] = float(v0[1]);
    v[2] = float(v0[2]); v[3] = float(v0[3]);
    v[4] = float(v1[0]); v[5] = float(v1[1]);
    v[6] = float(v1[2]); v[7] = float(v1[3]);
}

kernel void flash_attention_prefill_f16(
    device const half  *Q            [[buffer(0)]],
    device const half  *K_pool       [[buffer(1)]],
    device const half  *V_pool       [[buffer(2)]],
    device const int   *block_table  [[buffer(3)]],
    device       half  *O            [[buffer(4)]],
    constant uint      &seq_len      [[buffer(5)]],
    constant uint      &head_dim     [[buffer(6)]],
    constant uint      &num_heads    [[buffer(7)]],
    constant uint      &num_kv_heads [[buffer(8)]],
    constant uint      &block_size   [[buffer(9)]],
    constant uint      &causal       [[buffer(10)]],
    constant uint      &window_start [[buffer(11)]],
    uint3  gid            [[threadgroup_position_in_grid]],
    uint3  lid            [[thread_position_in_threadgroup]]
) {
    const uint kv_head_idx = gid.x;
    const uint q_tile_idx = gid.y;
    const uint gqa_ratio = num_heads / num_kv_heads;
    const uint sg_per_head = 8u / gqa_ratio;

    const uint tid = lid.x;
    const uint sg_id = tid / 32u;
    const uint lane = tid % 32u;
    const uint row_in_sg = lane / 8u;
    const uint d_group = lane % 8u;

    const uint head_within_group = sg_id / sg_per_head;
    const uint sg_within_head = sg_id % sg_per_head;
    const uint row_within_head = sg_within_head * 4u + row_in_sg;
    const uint head_idx = kv_head_idx * gqa_ratio + head_within_group;

    const uint rows_per_head = 32u / gqa_ratio;
    const uint q_start = q_tile_idx * rows_per_head;
    const uint q_row = q_start + row_within_head;
    const uint d_start = d_group * 8u;

    if (q_row >= seq_len || d_start + 7u >= head_dim) return;

    const float scale = 1.0f / sqrt(float(head_dim));
    device const half* q_base = Q + (q_row * num_heads + head_idx) * head_dim;
    float qv[8];
    fa_load8_half(q_base, d_start, qv);

    float m_i = -INFINITY;
    float l_i = 0.0f;
    float ov[8] = {0.0f};

    const uint k_end = causal ? min(q_row + 1u, seq_len) : seq_len;

    for (uint k_block = window_start; k_block < k_end; k_block += block_size) {
        int physical_block = block_table[k_block / block_size];
        device const half* K_block = K_pool + physical_block * block_size * num_kv_heads * head_dim;
        device const half* V_block = V_pool + physical_block * block_size * num_kv_heads * head_dim;

        uint blk_end = min(k_block + block_size, k_end);
        if (blk_end <= window_start) continue;

        for (uint local_k = 0u; local_k < blk_end - k_block; local_k++) {
            device const half* k_base = K_block + local_k * num_kv_heads * head_dim
                                        + kv_head_idx * head_dim;
            float kv[8];
            fa_load8_half(k_base, d_start, kv);

            float partial = 0.0f;
            for (uint dd = 0u; dd < 8u; dd++) partial += qv[dd] * kv[dd];

            float dot = partial;
            dot += simd_shuffle_xor(dot, 1);
            dot += simd_shuffle_xor(dot, 2);
            dot += simd_shuffle_xor(dot, 4);
            dot *= scale;

            float m_old = m_i;
            m_i = max(m_i, dot);
            float corr = exp(m_old - m_i);
            float exp_dot = exp(dot - m_i);
            l_i = l_i * corr + exp_dot;

            device const half* v_base = V_block + local_k * num_kv_heads * head_dim
                                        + kv_head_idx * head_dim;
            for (uint dd = 0u; dd < 8u; dd++) {
                ov[dd] = ov[dd] * corr + exp_dot * float(v_base[d_start + dd]);
            }
        }
    }

    device half* o_base = O + (q_row * num_heads + head_idx) * head_dim;
    for (uint dd = 0u; dd < 8u; dd++) o_base[d_start + dd] = half(ov[dd] / l_i);
}


kernel void flash_attention_prefill_f32(
    device const float *Q            [[buffer(0)]],
    device const float *K_pool       [[buffer(1)]],
    device const float *V_pool       [[buffer(2)]],
    device const int   *block_table  [[buffer(3)]],
    device       float *O            [[buffer(4)]],
    constant uint      &seq_len      [[buffer(5)]],
    constant uint      &head_dim     [[buffer(6)]],
    constant uint      &num_heads    [[buffer(7)]],
    constant uint      &num_kv_heads [[buffer(8)]],
    constant uint      &block_size   [[buffer(9)]],
    constant uint      &causal       [[buffer(10)]],
    constant uint      &window_start [[buffer(11)]],
    uint3  gid            [[threadgroup_position_in_grid]],
    uint3  lid            [[thread_position_in_threadgroup]]
) {
    const uint kv_head_idx = gid.x;
    const uint q_tile_idx = gid.y;
    const uint gqa_ratio = num_heads / num_kv_heads;
    const uint sg_per_head = 8u / gqa_ratio;

    const uint tid = lid.x;
    const uint sg_id = tid / 32u;
    const uint lane = tid % 32u;
    const uint row_in_sg = lane / 8u;
    const uint d_group = lane % 8u;

    const uint head_within_group = sg_id / sg_per_head;
    const uint sg_within_head = sg_id % sg_per_head;
    const uint row_within_head = sg_within_head * 4u + row_in_sg;
    const uint head_idx = kv_head_idx * gqa_ratio + head_within_group;

    const uint rows_per_head = 32u / gqa_ratio;
    const uint q_start = q_tile_idx * rows_per_head;
    const uint q_row = q_start + row_within_head;
    const uint d_start = d_group * 8u;

    if (q_row >= seq_len || d_start + 7u >= head_dim) return;

    const float scale = 1.0f / sqrt(float(head_dim));
    device const float* q_base = Q + (q_row * num_heads + head_idx) * head_dim;
    float qv[8];
    for (uint dd = 0u; dd < 8u; dd++) qv[dd] = q_base[d_start + dd];

    float m_i = -INFINITY;
    float l_i = 0.0f;
    float ov[8] = {0.0f};

    const uint k_end = causal ? min(q_row + 1u, seq_len) : seq_len;

    for (uint k_block = window_start; k_block < k_end; k_block += block_size) {
        int physical_block = block_table[k_block / block_size];
        device const float* K_block = K_pool + physical_block * block_size * num_kv_heads * head_dim;
        device const float* V_block = V_pool + physical_block * block_size * num_kv_heads * head_dim;

        uint blk_end = min(k_block + block_size, k_end);
        if (blk_end <= window_start) continue;

        for (uint local_k = 0u; local_k < blk_end - k_block; local_k++) {
            device const float* k_base = K_block + local_k * num_kv_heads * head_dim
                                         + kv_head_idx * head_dim;
            float partial = 0.0f;
            for (uint dd = 0u; dd < 8u; dd += 4u) {
                float4 qw(qv[dd], qv[dd+1], qv[dd+2], qv[dd+3]);
                float4 kw(k_base[d_start + dd], k_base[d_start + dd + 1],
                          k_base[d_start + dd + 2], k_base[d_start + dd + 3]);
                partial += dot(qw, kw);
            }

            float dot = partial;
            dot += simd_shuffle_xor(dot, 1);
            dot += simd_shuffle_xor(dot, 2);
            dot += simd_shuffle_xor(dot, 4);
            dot *= scale;

            float m_old = m_i;
            m_i = max(m_i, dot);
            float corr = exp(m_old - m_i);
            float exp_dot = exp(dot - m_i);
            l_i = l_i * corr + exp_dot;

            device const float* v_base = V_block + local_k * num_kv_heads * head_dim
                                         + kv_head_idx * head_dim;
            for (uint dd = 0u; dd < 8u; dd++) {
                ov[dd] = ov[dd] * corr + exp_dot * v_base[d_start + dd];
            }
        }
    }

    device float* o_base = O + (q_row * num_heads + head_idx) * head_dim;
    for (uint dd = 0u; dd < 8u; dd++) o_base[d_start + dd] = ov[dd] / l_i;
}

constant uint causal [[function_constant(0)]];
constant uint head_dim [[function_constant(1)]];

kernel void flash_attention_mma_f16(
    device const half  *Q            [[buffer(0)]],
    device const half  *K_pool       [[buffer(1)]],
    device const half  *V_pool       [[buffer(2)]],
    device const int   *block_table  [[buffer(3)]],
    device       half  *O            [[buffer(4)]],
    constant uint      &seq_len      [[buffer(5)]],
    constant uint      &num_heads    [[buffer(6)]],
    constant uint      &num_kv_heads [[buffer(7)]],
    constant uint      &block_size   [[buffer(8)]],
    constant uint      &window_start [[buffer(9)]],
    uint3  gid            [[threadgroup_position_in_grid]],
    uint3  lid            [[thread_position_in_threadgroup]],

    threadgroup half  *Q_tile_ptr  [[threadgroup(0)]],
    threadgroup half  *P_tile_ptr  [[threadgroup(1)]],
    threadgroup float *O_tile_ptr  [[threadgroup(2)]],
    threadgroup float *ml_i_ptr    [[threadgroup(3)]]
) {
    constexpr uint BQ = 32;
    constexpr uint BK = 16;
    constexpr uint WN = 2;

    const uint kv_head_idx = gid.x;
    const uint q_tile_idx = gid.y;
    const uint gqa_ratio = num_heads / num_kv_heads;
    const uint rows_per_head = BQ / gqa_ratio;
    const uint q_start = q_tile_idx * BQ;

    const uint tid = lid.x;
    const uint sg_id = tid / 32;
    const uint sg_row = sg_id / WN;
    const uint sg_col = sg_id % WN;

    threadgroup float* m_i = ml_i_ptr;
    threadgroup float* l_i = ml_i_ptr + BQ;

    {
        const uint q_row = tid / 8;
        const uint chunk = tid % 8;
        if (q_row < BQ) {
            const uint head_within = q_row / rows_per_head;
            const uint head_idx = kv_head_idx * gqa_ratio + head_within;
            const bool valid = (head_within < gqa_ratio) && (q_start + q_row < seq_len);
            device const half* qb = Q + ((q_start + q_row) * num_heads + head_idx) * head_dim;
            for (uint d = chunk * 8; d < head_dim; d += 64) {
                if (valid) {
                    for (uint i = 0; i < 8; i++)
                        Q_tile_ptr[q_row * head_dim + d + i] = qb[d + i];
                } else {
                    for (uint i = 0; i < 8; i++)
                        Q_tile_ptr[q_row * head_dim + d + i] = 0.0h;
                }
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < BQ) { m_i[tid] = -INFINITY; l_i[tid] = 0.0f; }
    for (uint o = tid; o < BQ * head_dim; o += 256) O_tile_ptr[o] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float scale = 1.0f / sqrt(float(head_dim));
    const uint k_limit = causal ? min(q_start + BQ, seq_len) : seq_len;

    for (uint k_block = window_start; k_block < k_limit; k_block += BK) {
        int pb = block_table[k_block / BK];
        device const half* Kb = K_pool + pb * BK * num_kv_heads * head_dim;
        device const half* Vb = V_pool + pb * BK * num_kv_heads * head_dim;

        float S_local[BK];
        {
            const uint qrow_in_sg = (tid % 32) / 8;
            const uint d_chunk = (tid % 32) % 8;
            const uint q_row = sg_id * 4 + qrow_in_sg;
            for (uint kt = 0; kt < BK; kt++) {
                float full = 0.0f;
                for (uint d_block = 0; d_block < head_dim; d_block += 64) {
                    uint d_start = d_block + d_chunk * 8;
                    float partial = 0.0f;
                    for (uint i = 0; i < 8; i++)
                        partial += float(Q_tile_ptr[q_row * head_dim + d_start + i])
                                 * float(Kb[kt * num_kv_heads * head_dim + kv_head_idx * head_dim + d_start + i]);
                    full += partial;
                }
                float red = full;
                red += simd_shuffle_xor(red, 1);
                red += simd_shuffle_xor(red, 2);
                red += simd_shuffle_xor(red, 4);
                S_local[kt] = red * scale;
            }
        }

        {
            uint row = tid / 8;
            uint lir = tid % 8;
            uint es = lir * 2;

            if (row < BQ) {
                float m_old = m_i[row];
                float l_old = l_i[row];
                float S[2];
                for (uint i = 0; i < 2; i++) {
                    uint j = es + i;
                    float sv = S_local[j];
                    if (causal && k_block + j > q_start + row) sv = -INFINITY;
                    S[i] = sv;
                }

                float pm = max(S[0], S[1]);
                pm = max(pm, simd_shuffle_xor(pm, 1));
                pm = max(pm, simd_shuffle_xor(pm, 2));
                pm = max(pm, simd_shuffle_xor(pm, 4));
                float m_new = max(m_old, pm);
                float alpha = exp(m_old - m_new);

                float ps = 0.0f;
                for (uint i = 0; i < 2; i++) {
                    float pv = exp(S[i] - m_new);
                    S[i] = pv;
                    ps += pv;
                }
                ps = ps + simd_shuffle_xor(ps, 1);
                ps = ps + simd_shuffle_xor(ps, 2);
                ps = ps + simd_shuffle_xor(ps, 4);
                float l_new = l_old * alpha + ps;

                if (lir == 0) { m_i[row] = m_new; l_i[row] = l_new; }

                for (uint d = lir * 8; d < head_dim; d += 64)
                    for (uint i = 0; i < 8; i++)
                        O_tile_ptr[row * head_dim + d + i] *= alpha;

                for (uint i = 0; i < 2; i++)
                    P_tile_ptr[row * BK + es + i] = half(S[i]);
            }
        }

        {
            uint qr = sg_row * 8;
            for (uint dp = 0; dp < head_dim / (WN * 8); dp++) {
                uint d_base = dp * WN * 8;
                uint d_off = d_base + sg_col * 8;

                simdgroup_float8x8 O_local(0.0f);
                simdgroup_load(O_local, O_tile_ptr, head_dim, ulong2(d_off, qr));

                for (uint k = 0; k < BK; k += 8) {
                    simdgroup_half8x8 Pm, Vm;
                    simdgroup_load(Pm, P_tile_ptr, BK, ulong2(k, qr));
                    simdgroup_load(Vm, Vb, num_kv_heads * head_dim,
                        ulong2(kv_head_idx * head_dim + d_off, k));
                    simdgroup_multiply_accumulate(O_local, Pm, Vm, O_local);
                }

                simdgroup_store(O_local, O_tile_ptr, head_dim, ulong2(d_off, qr));
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    {
        const uint num_elems = BQ * head_dim;
        for (uint e = tid * 8; e < num_elems; e += 2048) {
            for (uint i = 0; i < 8; i++) {
                uint idx = e + i;
                if (idx < num_elems) {
                    uint r = idx / head_dim;
                    uint c = idx % head_dim;
                    uint gqr = q_start + r;
                    if (gqr < seq_len) {
                        uint hwg = r / rows_per_head;
                        uint hi = kv_head_idx * gqa_ratio + hwg;
                        if (hwg < gqa_ratio) {
                            device half* ob = O + (gqr * num_heads + hi) * head_dim;
                            ob[c] = half(O_tile_ptr[idx] / l_i[r]);
                        }
                    }
                }
            }
        }
    }
}

// Performs decode (single query, multi-token attention) with simdgroup_matrix for
// Q·K dot products and P·V accumulation.
kernel void flash_decode_mma_f16(
    device const half  *Q                [[buffer(0)]],
    device const half  *K_pool           [[buffer(1)]],
    device const half  *V_pool           [[buffer(2)]],
    device const int   *block_tables     [[buffer(3)]],
    device const uint  *seq_lengths      [[buffer(4)]],
    device half        *O                [[buffer(5)]],
    constant uint      &batch_size       [[buffer(6)]],
    constant uint      &num_heads        [[buffer(7)]],
    constant uint      &num_kv_heads     [[buffer(8)]],
    constant uint      &block_size       [[buffer(9)]],
    constant uint      &max_num_blocks   [[buffer(10)]],
    constant uint      &window_size      [[buffer(11)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],

    threadgroup float *partial_scores [[threadgroup(0)]],
    threadgroup float *m_l_ptr        [[threadgroup(1)]]
) {
    constexpr uint BD = 64;
    if (head_dim != BD) return;

    const uint kv_head_idx = gid.y;
    const uint batch_idx = gid.z;
    const uint gqa_ratio = num_heads / num_kv_heads;
    const uint tid = lid.x;
    const uint sg_id = tid / 32;
    const uint lane = tid % 32;
    const uint d_chunk = sg_id * 8;

    if (d_chunk >= BD) return;

    const uint seq_len = seq_lengths[batch_idx];
    const float scale = 1.0f / sqrt(float(BD));
    const uint ws = (window_size > 0 && seq_len > window_size) ? seq_len - window_size : 0;

    device const int* bt = block_tables + batch_idx * max_num_blocks;
    uint nb = (seq_len + block_size - 1u) / block_size;
    if (nb > max_num_blocks) nb = max_num_blocks;
    uint sb = ws / block_size;

    threadgroup float* m_i = m_l_ptr;
    threadgroup float* l_i = m_l_ptr + gqa_ratio;
    threadgroup float* ps = partial_scores;

    if (tid < gqa_ratio) { m_i[tid] = -INFINITY; l_i[tid] = 0.0f; }
    device half* O_head = O + (batch_idx * num_heads + kv_head_idx * gqa_ratio) * BD;
    for (uint o = tid; o < gqa_ratio * BD; o += 256) O_head[o] = 0.0h;
    for (uint p = tid; p < gqa_ratio * 8 + gqa_ratio * 2; p += 256) ps[p] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint lb = sb; lb < nb; lb++) {
        int pb = bt[lb];
        device const half* Kb = K_pool + pb * block_size * num_kv_heads * BD + kv_head_idx * BD;
        device const half* Vb = V_pool + pb * block_size * num_kv_heads * BD + kv_head_idx * BD;

        uint bs = lb * block_size;
        uint be = min(bs + block_size, seq_len);

        for (uint j = bs; j < be; j++) {
            if (j < ws) continue;
            uint lj = j - bs;
            device const half* K_tok = Kb + lj * num_kv_heads * BD;
            device const half* V_tok = Vb + lj * num_kv_heads * BD;

            // Phase 1: Cooperative Q·K dot product
            // 32 threads per simdgroup, 4 per head × 2 D-values = 8 D per head
            uint head = lane / 4;
            uint d0 = 2 * (lane % 4);
            uint d1 = d0 + 1;
            float partial = 0.0f;
            if (head < gqa_ratio && d1 < 8) {
                uint q_base = (batch_idx * num_heads + kv_head_idx * gqa_ratio + head) * BD;
                float qv0 = float(Q[q_base + d_chunk + d0]);
                float qv1 = float(Q[q_base + d_chunk + d1]);
                float kv0 = float(K_tok[d_chunk + d0]);
                float kv1 = float(K_tok[d_chunk + d1]);
                partial = qv0 * kv0 + qv1 * kv1;
            }
            partial += simd_shuffle_xor(partial, 1);
            partial += simd_shuffle_xor(partial, 2);
            float part_sum = partial;
            if (head < gqa_ratio) {
                ps[head * 8 + sg_id] = part_sum;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Phase 2: Cross-simdgroup reduction + online softmax
            if (tid < gqa_ratio) {
                float score = 0.0f;
                for (uint s = 0; s < 8; s++) score += ps[tid * 8 + s];
                score *= scale;
                float m_old = m_i[tid];
                float m_new = max(m_old, score);
                float alpha = exp(m_old - m_new);
                float p = exp(score - m_new);
                float l_new = l_i[tid] * alpha + p;
                m_i[tid] = m_new;
                l_i[tid] = l_new;
                ps[gqa_ratio * 8 + tid] = alpha;
                ps[gqa_ratio * 8 + gqa_ratio + tid] = p;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Phase 3: P·V accumulation (all threads)
            uint ne = gqa_ratio * BD;
            for (uint e = tid; e < ne; e += 256) {
                uint h = e / BD;
                uint d = e % BD;
                float v_val = float(V_tok[d]);
                float alphav = ps[gqa_ratio * 8 + h];
                float prob = ps[gqa_ratio * 8 + gqa_ratio + h];
                O_head[h * BD + d] = half(float(O_head[h * BD + d]) * alphav + prob * v_val);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // Final normalization: divide O by l_i
    uint total_elements = gqa_ratio * BD;
    for (uint o = tid; o < total_elements; o += 256) {
        uint h = o / BD;
        uint d = o % BD;
        O_head[o] = half(float(O_head[o]) / l_i[h]);
    }
}

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

// MARK: - Atomic Add Helper (for half, CAS-based)

void atomic_add_half(device half* ptr, half value) {
    device atomic_int* aptr = (device atomic_int*)ptr;
    int expected = atomic_load_explicit(aptr, memory_order_relaxed);
    int desired;
    do {
        half current = as_type<half>(static_cast<ushort>(expected & 0xFFFF));
        current += value;
        ushort new_bits = as_type<ushort>(current);
        desired = (expected & ~0xFFFF) | static_cast<int>(new_bits);
    } while (!atomic_compare_exchange_weak_explicit(aptr, &expected, desired,
                                                     memory_order_relaxed, memory_order_relaxed));
}

// MARK: - Atomic Max Helper (for float, CAS-based)

void atomic_max_float(device float* ptr, float value) {
    device atomic_int* aptr = (device atomic_int*)ptr;
    int old_val = atomic_load_explicit(aptr, memory_order_relaxed);
    int new_val;
    do {
        union { int i; float f; } u;
        u.i = old_val;
        if (u.f >= value) break;
        u.f = value;
        new_val = u.i;
    } while (!atomic_compare_exchange_weak_explicit(aptr, &old_val, new_val,
                                                     memory_order_relaxed, memory_order_relaxed));
}

// MARK: - FP8 E4M3 Helpers

inline half dequant_fp8(uchar v, half scale) {
    int sign = (v >> 7) & 1;
    int exp = (v >> 3) & 0xF;
    int mant = v & 0x7;
    if (exp == 0 && mant == 0) return 0.0h;
    if (exp == 0) {
        return (sign ? -1.0h : 1.0h) * scale * (mant / 8.0h) * 0.015625h;
    }
    return (sign ? -1.0h : 1.0h) * scale * (1.0h + mant / 8.0h) * pow(2.0h, half(exp - 7));
}

inline uchar quantize_fp8(half value, half scale) {
    float v = float(value) / float(scale);
    v = clamp(v, -448.0f, 448.0f);

    uint sign = v < 0.0f ? 1 : 0;
    v = abs(v);

    if (v == 0.0f) return 0;

    if (v < 0x1p-6f) {
        int mant_int = int(round(v * 512.0f));
        if (mant_int > 7) mant_int = 7;
        if (mant_int < 0) mant_int = 0;
        return uchar((sign << 7) | mant_int);
    }

    int exp;
    float mant = frexp(v, exp);
    mant = 2.0f * mant;
    exp -= 1;

    if (exp > 8) {
        return uchar((sign << 7) | (0xF << 3) | 0x6);
    }

    int exp_bits = exp + 7;
    float mant_frac = mant - 1.0f;
    int mant_int = int(round(mant_frac * 8.0f));
    if (mant_int > 7) { mant_int = 0; exp_bits++; }
    if (exp_bits > 0xF) {
        return uchar((sign << 7) | (0xF << 3) | 0x6);
    }
    if (exp_bits == 0xF && mant_int > 6) {
        mant_int = 6;
    }

    return uchar((sign << 7) | (exp_bits << 3) | mant_int);
}

inline float simd_dot_product_fp8(const device half* a, const device uchar* b, half scale, uint head_dim) {
    float sum = 0.0f;
    uint d = 0;
    for (; d + 4 <= head_dim; d += 4) {
        half4 av = *(device const half4*)(a + d);
        half b0 = dequant_fp8(b[d], scale);
        half b1 = dequant_fp8(b[d+1], scale);
        half b2 = dequant_fp8(b[d+2], scale);
        half b3 = dequant_fp8(b[d+3], scale);
        sum += dot(float4(av), float4(b0, b1, b2, b3));
    }
    for (; d < head_dim; d++) {
        sum += float(a[d]) * float(dequant_fp8(b[d], scale));
    }
    return sum;
}

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
        half4 av_h = *(device const half4*)(a + d);
        half4 bv_h = *(device const half4*)(b + d);
        float4 av = float4(av_h);
        float4 bv = float4(bv_h);
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

inline float simd_dot_tg_dev(const threadgroup float* a, uint a_off, const device float* b, uint head_dim) {
    float sum = 0.0f;
    uint d = 0;
    for (; d + 4 <= head_dim; d += 4) {
        float4 av(a[a_off + d], a[a_off + d + 1], a[a_off + d + 2], a[a_off + d + 3]);
        float4 bv(b[d], b[d+1], b[d+2], b[d+3]);
        sum += dot(av, bv);
    }
    for (; d < head_dim; d++) {
        sum += a[a_off + d] * b[d];
    }
    return sum;
}

inline float simd_dot_tg_dev_f16(const threadgroup float* a, uint a_off, const device half* b, uint head_dim) {
    float sum = 0.0f;
    uint d = 0;
    for (; d + 4 <= head_dim; d += 4) {
        float4 av(a[a_off + d], a[a_off + d + 1], a[a_off + d + 2], a[a_off + d + 3]);
        half4 bv_h = *(device const half4*)(b + d);
        float4 bv = float4(bv_h);
        sum += dot(av, bv);
    }
    for (; d < head_dim; d++) {
        sum += a[a_off + d] * float(b[d]);
    }
    return sum;
}


// MARK: - Paged Attention Tiled (Replaces Split-K)

// Q layout:      [seq_len, num_heads, head_dim]
// K/V pool layout: [num_physical_blocks, block_size, num_kv_heads, head_dim]
// Output layout: [seq_len, num_heads, head_dim]
// Grid: (1, num_qtiles, num_heads), Threadgroup: (head_dim, q_tile_size, 1)
kernel void paged_attention_tiled(
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
    constant uint        &q_tile_size  [[buffer(11)]],
    constant uint        &window_start [[buffer(12)]],

    uint3  gid              [[threadgroup_position_in_grid]],
    uint3  lid              [[thread_position_in_threadgroup]],

    threadgroup float *Q_tile   [[threadgroup(0)]]
) {
    const uint local_q    = lid.y;
    const uint col        = lid.x;
    const uint q_start    = gid.y * q_tile_size;
    const uint q_row      = q_start + local_q;
    const uint head_idx   = gid.z;
    const uint kv_head_idx = head_idx / (num_heads / num_kv_heads);
    const float scale = 1.0f / sqrt(float(head_dim));

    if (q_row < seq_len && col < head_dim) {
        Q_tile[local_q * head_dim + col] = Q[(q_row * num_heads + head_idx) * head_dim + col];
    } else if (local_q < q_tile_size && col < head_dim) {
        Q_tile[local_q * head_dim + col] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float m_i   = -INFINITY;
    float l_i   = 0.0f;
    float acc_o = 0.0f;

    const uint num_logical_blocks = (seq_len + block_size - 1) / block_size;

    for (uint k_block = window_start; k_block < seq_len; k_block += block_size) {
        uint logical_block = k_block / block_size;
        if (logical_block >= num_logical_blocks) break;
        int physical_block = block_table[logical_block];

        device const float* K_block = K_pool + physical_block * block_size * num_kv_heads * head_dim;
        device const float* V_block = V_pool + physical_block * block_size * num_kv_heads * head_dim;

        for (uint local_j = 0; local_j < block_size; local_j++) {
            uint global_j = k_block + local_j;
            if (global_j >= seq_len) break;
            if (causal && global_j > q_row) continue;

            device const float* K_token = K_block + local_j * num_kv_heads * head_dim + kv_head_idx * head_dim;
            device const float* V_token = V_block + local_j * num_kv_heads * head_dim + kv_head_idx * head_dim;

            float dot = simd_dot_tg_dev(Q_tile, local_q * head_dim, K_token, head_dim);
            dot *= scale;

            float m_old = m_i;
            m_i = max(m_i, dot);
            float correction = exp(m_old - m_i);
            float exp_dot = exp(dot - m_i);
            l_i = l_i * correction + exp_dot;
            acc_o = acc_o * correction + exp_dot * V_token[col];
        }
    }

    if (q_row < seq_len && col < head_dim) {
        O[(q_row * num_heads + head_idx) * head_dim + col] = acc_o / l_i;
    }
}


kernel void paged_attention_tiled_f16(
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
    constant uint        &q_tile_size  [[buffer(11)]],
    constant uint        &window_start [[buffer(12)]],

    uint3  gid              [[threadgroup_position_in_grid]],
    uint3  lid              [[thread_position_in_threadgroup]],

    threadgroup float *Q_tile   [[threadgroup(0)]]
) {
    const uint local_q    = lid.y;
    const uint col        = lid.x;
    const uint q_start    = gid.y * q_tile_size;
    const uint q_row      = q_start + local_q;
    const uint head_idx   = gid.z;
    const uint kv_head_idx = head_idx / (num_heads / num_kv_heads);
    const float scale = 1.0f / sqrt(float(head_dim));

    if (q_row < seq_len && col < head_dim) {
        Q_tile[local_q * head_dim + col] = float(Q[(q_row * num_heads + head_idx) * head_dim + col]);
    } else if (local_q < q_tile_size && col < head_dim) {
        Q_tile[local_q * head_dim + col] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float m_i   = -INFINITY;
    float l_i   = 0.0f;
    float acc_o = 0.0f;

    const uint num_logical_blocks = (seq_len + block_size - 1) / block_size;

    for (uint k_block = window_start; k_block < seq_len; k_block += block_size) {
        uint logical_block = k_block / block_size;
        if (logical_block >= num_logical_blocks) break;
        int physical_block = block_table[logical_block];

        device const half* K_block = K_pool + physical_block * block_size * num_kv_heads * head_dim;
        device const half* V_block = V_pool + physical_block * block_size * num_kv_heads * head_dim;

        for (uint local_j = 0; local_j < block_size; local_j++) {
            uint global_j = k_block + local_j;
            if (global_j >= seq_len) break;
            if (causal && global_j > q_row) continue;

            device const half* K_token = K_block + local_j * num_kv_heads * head_dim + kv_head_idx * head_dim;
            device const half* V_token = V_block + local_j * num_kv_heads * head_dim + kv_head_idx * head_dim;

            float dot = simd_dot_tg_dev_f16(Q_tile, local_q * head_dim, K_token, head_dim);
            dot *= scale;

            float m_old = m_i;
            m_i = max(m_i, dot);
            float correction = exp(m_old - m_i);
            float exp_dot = exp(dot - m_i);
            l_i = l_i * correction + exp_dot;
            acc_o = acc_o * correction + exp_dot * float(V_token[col]);
        }
    }

    if (q_row < seq_len && col < head_dim) {
        O[(q_row * num_heads + head_idx) * head_dim + col] = half(acc_o / l_i);
    }
}


kernel void paged_attention_tiled_fp8(
    device const half    *Q            [[buffer(0)]],
    device const uchar   *K_pool       [[buffer(1)]],
    device const uchar   *V_pool       [[buffer(2)]],
    device const int     *block_table  [[buffer(3)]],
    device       half    *O            [[buffer(4)]],
    constant uint        &seq_len      [[buffer(5)]],
    constant uint        &head_dim     [[buffer(6)]],
    constant uint        &num_heads    [[buffer(7)]],
    constant uint        &num_kv_heads [[buffer(8)]],
    constant uint        &block_size   [[buffer(9)]],
    constant uint        &causal       [[buffer(10)]],
    constant uint        &q_tile_size  [[buffer(11)]],
    constant uint        &window_start [[buffer(12)]],
    device const half    *k_scale_pool [[buffer(13)]],
    device const half    *v_scale_pool [[buffer(14)]],

    uint3  gid              [[threadgroup_position_in_grid]],
    uint3  lid              [[thread_position_in_threadgroup]],

    threadgroup float *Q_tile   [[threadgroup(0)]]
) {
    const uint local_q    = lid.y;
    const uint col        = lid.x;
    const uint q_start    = gid.y * q_tile_size;
    const uint q_row      = q_start + local_q;
    const uint head_idx   = gid.z;
    const uint kv_head_idx = head_idx / (num_heads / num_kv_heads);
    const float scale = 1.0f / sqrt(float(head_dim));

    if (q_row < seq_len && col < head_dim) {
        Q_tile[local_q * head_dim + col] = float(Q[(q_row * num_heads + head_idx) * head_dim + col]);
    } else if (local_q < q_tile_size && col < head_dim) {
        Q_tile[local_q * head_dim + col] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float m_i   = -INFINITY;
    float l_i   = 0.0f;
    float acc_o = 0.0f;

    const uint num_logical_blocks = (seq_len + block_size - 1) / block_size;

    for (uint k_block = window_start; k_block < seq_len; k_block += block_size) {
        uint logical_block = k_block / block_size;
        if (logical_block >= num_logical_blocks) break;
        int physical_block = block_table[logical_block];

        device const uchar* K_block = K_pool + physical_block * block_size * num_kv_heads * head_dim;
        device const uchar* V_block = V_pool + physical_block * block_size * num_kv_heads * head_dim;
        half k_scale = k_scale_pool[physical_block];
        half v_scale = v_scale_pool[physical_block];

        for (uint local_j = 0; local_j < block_size; local_j++) {
            uint global_j = k_block + local_j;
            if (global_j >= seq_len) break;
            if (causal && global_j > q_row) continue;

            device const uchar* K_token = K_block + local_j * num_kv_heads * head_dim + kv_head_idx * head_dim;
            device const uchar* V_token = V_block + local_j * num_kv_heads * head_dim + kv_head_idx * head_dim;

            float dot = simd_dot_product_fp8(Q + (q_row * num_heads + head_idx) * head_dim, K_token, k_scale, head_dim);
            dot *= scale;

            float m_old = m_i;
            m_i = max(m_i, dot);
            float correction = exp(m_old - m_i);
            float exp_dot = exp(dot - m_i);
            l_i = l_i * correction + exp_dot;
            acc_o = acc_o * correction + exp_dot * float(dequant_fp8(V_token[col], v_scale));
        }
    }

    if (q_row < seq_len && col < head_dim) {
        O[(q_row * num_heads + head_idx) * head_dim + col] = half(acc_o / l_i);
    }
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
    device       float   *m_out        [[buffer(11)]],
    device       float   *l_out        [[buffer(12)]],
    constant uint        &window_start [[buffer(13)]],

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

    const uint num_logical_blocks = (seq_len + block_size - 1) / block_size;

    for (uint k_block = window_start; k_block < seq_len; k_block += block_size) {
        uint logical_block = k_block / block_size;
        if (logical_block >= num_logical_blocks) break;
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
                float exp_dot = exp(dot - m_i);
                l_i = l_i * correction + exp_dot;
                acc_o = acc_o * correction;
                acc_o += exp_dot * V_tile[local_j * head_dim + col];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < seq_len && col < head_dim) {
        O[(row * num_heads + head_idx) * head_dim + col] = acc_o / l_i;
    }
    if (row < seq_len && col == 0) {
        m_out[row * num_heads + head_idx] = m_i;
        l_out[row * num_heads + head_idx] = l_i;
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
    device       float   *m_out        [[buffer(11)]],
    device       float   *l_out        [[buffer(12)]],
    constant uint        &window_start [[buffer(13)]],

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

    const uint num_logical_blocks = (seq_len + block_size - 1) / block_size;

    for (uint k_block = window_start; k_block < seq_len; k_block += block_size) {
        uint logical_block = k_block / block_size;
        if (logical_block >= num_logical_blocks) break;
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
                float exp_dot = exp(dot - m_i);
                l_i = l_i * correction + exp_dot;
                acc_o = acc_o * correction;
                acc_o += exp_dot * V_tile[local_j * head_dim + col];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < seq_len && col < head_dim) {
        O[(row * num_heads + head_idx) * head_dim + col] = half(acc_o / l_i);
    }
    if (row < seq_len && col == 0) {
        m_out[row * num_heads + head_idx] = m_i;
        l_out[row * num_heads + head_idx] = l_i;
    }
}


kernel void paged_attention_single_fp8(
    device const half    *Q            [[buffer(0)]],
    device const uchar   *K_pool       [[buffer(1)]],
    device const uchar   *V_pool       [[buffer(2)]],
    device const int     *block_table  [[buffer(3)]],
    device       half    *O            [[buffer(4)]],
    constant uint        &seq_len      [[buffer(5)]],
    constant uint        &head_dim     [[buffer(6)]],
    constant uint        &num_heads    [[buffer(7)]],
    constant uint        &num_kv_heads [[buffer(8)]],
    constant uint        &block_size   [[buffer(9)]],
    constant uint        &causal       [[buffer(10)]],
    device       float   *m_out        [[buffer(11)]],
    device       float   *l_out        [[buffer(12)]],
    constant uint        &window_start [[buffer(13)]],
    device const half    *k_scale_pool [[buffer(14)]],
    device const half    *v_scale_pool [[buffer(15)]],

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

    const uint num_logical_blocks = (seq_len + block_size - 1) / block_size;

    for (uint k_block = window_start; k_block < seq_len; k_block += block_size) {
        uint logical_block = k_block / block_size;
        if (logical_block >= num_logical_blocks) break;
        int physical_block = block_table[logical_block];

        device const uchar* K_block_base = K_pool + physical_block * block_size * num_kv_heads * head_dim;
        device const uchar* V_block_base = V_pool + physical_block * block_size * num_kv_heads * head_dim;
        half k_scale = k_scale_pool[physical_block];
        half v_scale = v_scale_pool[physical_block];

        uint local_k_row = lid.y;
        uint global_k_row = k_block + local_k_row;

        if (global_k_row < seq_len && col < head_dim) {
            K_tile[local_k_row * head_dim + col] = float(dequant_fp8(K_block_base[local_k_row * num_kv_heads * head_dim + kv_head_idx * head_dim + col], k_scale));
            V_tile[local_k_row * head_dim + col] = float(dequant_fp8(V_block_base[local_k_row * num_kv_heads * head_dim + kv_head_idx * head_dim + col], v_scale));
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
                float exp_dot = exp(dot - m_i);
                l_i = l_i * correction + exp_dot;
                acc_o = acc_o * correction;
                acc_o += exp_dot * V_tile[local_j * head_dim + col];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < seq_len && col < head_dim) {
        O[(row * num_heads + head_idx) * head_dim + col] = half(acc_o / l_i);
    }
    if (row < seq_len && col == 0) {
        m_out[row * num_heads + head_idx] = m_i;
        l_out[row * num_heads + head_idx] = l_i;
    }
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
    constant uint        &window_start [[buffer(15)]],
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

    uint start_block = window_start / block_size;
    for (uint logical_block = start_block; logical_block < num_logical_blocks; logical_block++) {
        int physical_block = block_table[logical_block];
        device const float* K_block = K_pool + physical_block * block_size * num_kv_heads * head_dim;
        device const float* V_block = V_pool + physical_block * block_size * num_kv_heads * head_dim;

        for (uint local_j = 0; local_j < block_size; local_j++) {
            uint j = logical_block * block_size + local_j;
            if (j >= seq_len) break;
            if (j < window_start) continue;

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

    for (uint logical_block = start_block; logical_block < num_logical_blocks; logical_block++) {
        int physical_block = block_table[logical_block];
        device const float* K_block = K_pool + physical_block * block_size * num_kv_heads * head_dim;
        device const float* V_block = V_pool + physical_block * block_size * num_kv_heads * head_dim;

        for (uint local_j = 0; local_j < block_size; local_j++) {
            uint j = logical_block * block_size + local_j;
            if (j >= seq_len) break;
            if (j < window_start) continue;

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


kernel void paged_attention_backward_f16(
    device const half    *Q            [[buffer(0)]],
    device const half    *K_pool       [[buffer(1)]],
    device const half    *V_pool       [[buffer(2)]],
    device const int     *block_table  [[buffer(3)]],
    device const half    *dO           [[buffer(4)]],
    device const float   *m            [[buffer(5)]],
    device const float   *l            [[buffer(6)]],
    device       half    *dQ           [[buffer(7)]],
    device       half    *dK_pool      [[buffer(8)]],
    device       half    *dV_pool      [[buffer(9)]],
    constant uint        &seq_len      [[buffer(10)]],
    constant uint        &head_dim     [[buffer(11)]],
    constant uint        &num_heads    [[buffer(12)]],
    constant uint        &num_kv_heads [[buffer(13)]],
    constant uint        &block_size   [[buffer(14)]],
    constant uint        &window_start [[buffer(15)]],
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
    const float q_id  = float(Q[q_offset + d]);
    const float dO_id = float(dO[q_offset + d]);
    const float m_i   = m[i * num_heads + head_idx];
    const float l_i   = l[i * num_heads + head_idx];

    const uint num_logical_blocks = (seq_len + block_size - 1) / block_size;

    float rowsum = 0.0f;

    uint start_block = window_start / block_size;
    for (uint logical_block = start_block; logical_block < num_logical_blocks; logical_block++) {
        int physical_block = block_table[logical_block];
        device const half* K_block = K_pool + physical_block * block_size * num_kv_heads * head_dim;
        device const half* V_block = V_pool + physical_block * block_size * num_kv_heads * head_dim;

        for (uint local_j = 0; local_j < block_size; local_j++) {
            uint j = logical_block * block_size + local_j;
            if (j >= seq_len) break;
            if (j < window_start) continue;

            device const half* K_token = K_block + local_j * num_kv_heads * head_dim + kv_head_idx * head_dim;
            device const half* V_token = V_block + local_j * num_kv_heads * head_dim + kv_head_idx * head_dim;

            float dot = simd_dot_product_f16(Q + q_offset, K_token, head_dim);
            dot *= scale;

            float P_ij = exp(dot - m_i) / l_i;
            float dP_ij = simd_dot_product_f16(dO + q_offset, V_token, head_dim);

            rowsum += P_ij * dP_ij;
        }
    }

    float dQ_acc = 0.0f;

    for (uint logical_block = start_block; logical_block < num_logical_blocks; logical_block++) {
        int physical_block = block_table[logical_block];
        device const half* K_block = K_pool + physical_block * block_size * num_kv_heads * head_dim;
        device const half* V_block = V_pool + physical_block * block_size * num_kv_heads * head_dim;

        for (uint local_j = 0; local_j < block_size; local_j++) {
            uint j = logical_block * block_size + local_j;
            if (j >= seq_len) break;
            if (j < window_start) continue;

            device const half* K_token = K_block + local_j * num_kv_heads * head_dim + kv_head_idx * head_dim;
            device const half* V_token = V_block + local_j * num_kv_heads * head_dim + kv_head_idx * head_dim;

            float dot = simd_dot_product_f16(Q + q_offset, K_token, head_dim);
            dot *= scale;

            float P_ij = exp(dot - m_i) / l_i;
            float dP_ij = simd_dot_product_f16(dO + q_offset, V_token, head_dim);
            float dS_ij = P_ij * (dP_ij - rowsum);

            dQ_acc += dS_ij * float(K_token[d]) * scale;

            float dK_contrib = dS_ij * q_id * scale;
            uint dK_offset = physical_block * block_size * num_kv_heads * head_dim + local_j * num_kv_heads * head_dim + kv_head_idx * head_dim + d;
            atomic_add_half(&dK_pool[dK_offset], half(dK_contrib));

            float dV_contrib = P_ij * dO_id;
            uint dV_offset = physical_block * block_size * num_kv_heads * head_dim + local_j * num_kv_heads * head_dim + kv_head_idx * head_dim + d;
            atomic_add_half(&dV_pool[dV_offset], half(dV_contrib));
        }
    }

    dQ[q_offset + d] = half(dQ_acc);
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
    constant uint      &window_size      [[buffer(12)]],
    uint3 gid [[thread_position_in_grid]]
) {
    const uint d         = gid.x;
    const uint head_idx  = gid.y;
    const uint batch_idx = gid.z;

    if (d >= head_dim || head_idx >= num_heads || batch_idx >= batch_size) return;

    const uint kv_head_idx = head_idx / (num_heads / num_kv_heads);
    const uint seq_len = seq_lengths[batch_idx];
    const float scale = 1.0f / sqrt(float(head_dim));
    const uint window_start = (window_size > 0 && seq_len > window_size) ? seq_len - window_size : 0;

    device const int* block_table = block_tables + batch_idx * max_num_blocks;
    device const float* q_vec = Q + (batch_idx * num_heads + head_idx) * head_dim;

    float m_i = -INFINITY;
    float l_i = 0.0f;
    float acc_o = 0.0f;

    uint num_logical_blocks = (seq_len + block_size - 1) / block_size;
    if (num_logical_blocks > max_num_blocks) num_logical_blocks = max_num_blocks;

    uint start_logical_block = window_start / block_size;
    for (uint logical_block = start_logical_block; logical_block < num_logical_blocks; logical_block++) {
        int physical_block = block_table[logical_block];
        device const float* K_block = K_pool + physical_block * block_size * num_kv_heads * head_dim;
        device const float* V_block = V_pool + physical_block * block_size * num_kv_heads * head_dim;

        uint block_start = logical_block * block_size;
        uint block_end = min(block_start + block_size, seq_len);

        for (uint j = block_start; j < block_end; j++) {
            if (j < window_start) continue;
            uint local_j = j - block_start;
            device const float* K_token = K_block + local_j * num_kv_heads * head_dim + kv_head_idx * head_dim;
            device const float* V_token = V_block + local_j * num_kv_heads * head_dim + kv_head_idx * head_dim;

            float dot = simd_dot_product(q_vec, K_token, head_dim);
            dot *= scale;

            float m_old = m_i;
            m_i = max(m_i, dot);
            float correction = exp(m_old - m_i);
            float exp_dot = exp(dot - m_i);
            l_i = l_i * correction + exp_dot;
            acc_o = acc_o * correction + exp_dot * V_token[d];
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
    constant uint      &window_size      [[buffer(12)]],
    uint3 gid [[thread_position_in_grid]]
) {
    const uint d         = gid.x;
    const uint head_idx  = gid.y;
    const uint batch_idx = gid.z;

    if (d >= head_dim || head_idx >= num_heads || batch_idx >= batch_size) return;

    const uint kv_head_idx = head_idx / (num_heads / num_kv_heads);
    const uint seq_len = seq_lengths[batch_idx];
    const float scale = 1.0f / sqrt(float(head_dim));
    const uint window_start = (window_size > 0 && seq_len > window_size) ? seq_len - window_size : 0;

    device const int* block_table = block_tables + batch_idx * max_num_blocks;
    device const half* q_vec = Q + (batch_idx * num_heads + head_idx) * head_dim;

    float m_i = -INFINITY;
    float l_i = 0.0f;
    float acc_o = 0.0f;

    uint num_logical_blocks = (seq_len + block_size - 1) / block_size;
    if (num_logical_blocks > max_num_blocks) num_logical_blocks = max_num_blocks;

    uint start_logical_block = window_start / block_size;
    for (uint logical_block = start_logical_block; logical_block < num_logical_blocks; logical_block++) {
        int physical_block = block_table[logical_block];
        device const half* K_block = K_pool + physical_block * block_size * num_kv_heads * head_dim;
        device const half* V_block = V_pool + physical_block * block_size * num_kv_heads * head_dim;

        uint block_start = logical_block * block_size;
        uint block_end = min(block_start + block_size, seq_len);

        for (uint j = block_start; j < block_end; j++) {
            if (j < window_start) continue;
            uint local_j = j - block_start;
            device const half* K_token = K_block + local_j * num_kv_heads * head_dim + kv_head_idx * head_dim;
            device const half* V_token = V_block + local_j * num_kv_heads * head_dim + kv_head_idx * head_dim;

            float dot = simd_dot_product_f16(q_vec, K_token, head_dim);
            dot *= scale;

            float m_old = m_i;
            m_i = max(m_i, dot);
            float correction = exp(m_old - m_i);
            float exp_dot = exp(dot - m_i);
            l_i = l_i * correction + exp_dot;
            acc_o = acc_o * correction + exp_dot * float(V_token[d]);
        }
    }

    O[(batch_idx * num_heads + head_idx) * head_dim + d] = half(acc_o / l_i);
}


kernel void paged_decode_single_fp8(
    device const half  *Q                [[buffer(0)]],
    device const uchar *K_pool           [[buffer(1)]],
    device const uchar *V_pool           [[buffer(2)]],
    device const int   *block_tables     [[buffer(3)]],
    device const uint  *seq_lengths      [[buffer(4)]],
    device half        *O                [[buffer(5)]],
    constant uint      &batch_size       [[buffer(6)]],
    constant uint      &head_dim         [[buffer(7)]],
    constant uint      &num_heads        [[buffer(8)]],
    constant uint      &num_kv_heads     [[buffer(9)]],
    constant uint      &block_size       [[buffer(10)]],
    constant uint      &max_num_blocks   [[buffer(11)]],
    constant uint      &window_size      [[buffer(12)]],
    device const half  *k_scale_pool     [[buffer(13)]],
    device const half  *v_scale_pool     [[buffer(14)]],
    uint3 gid [[thread_position_in_grid]]
) {
    const uint d         = gid.x;
    const uint head_idx  = gid.y;
    const uint batch_idx = gid.z;

    if (d >= head_dim || head_idx >= num_heads || batch_idx >= batch_size) return;

    const uint kv_head_idx = head_idx / (num_heads / num_kv_heads);
    const uint seq_len = seq_lengths[batch_idx];
    const float scale = 1.0f / sqrt(float(head_dim));
    const uint window_start = (window_size > 0 && seq_len > window_size) ? seq_len - window_size : 0;

    device const int* block_table = block_tables + batch_idx * max_num_blocks;
    device const half* q_vec = Q + (batch_idx * num_heads + head_idx) * head_dim;

    float m_i = -INFINITY;
    float l_i = 0.0f;
    float acc_o = 0.0f;

    uint num_logical_blocks = (seq_len + block_size - 1) / block_size;
    if (num_logical_blocks > max_num_blocks) num_logical_blocks = max_num_blocks;

    uint start_logical_block = window_start / block_size;
    for (uint logical_block = start_logical_block; logical_block < num_logical_blocks; logical_block++) {
        int physical_block = block_table[logical_block];
        device const uchar* K_block = K_pool + physical_block * block_size * num_kv_heads * head_dim;
        device const uchar* V_block = V_pool + physical_block * block_size * num_kv_heads * head_dim;
        half k_scale = k_scale_pool[physical_block];
        half v_scale = v_scale_pool[physical_block];

        uint block_start = logical_block * block_size;
        uint block_end = min(block_start + block_size, seq_len);

        for (uint j = block_start; j < block_end; j++) {
            if (j < window_start) continue;
            uint local_j = j - block_start;
            device const uchar* K_token = K_block + local_j * num_kv_heads * head_dim + kv_head_idx * head_dim;
            device const uchar* V_token = V_block + local_j * num_kv_heads * head_dim + kv_head_idx * head_dim;

            float dot = simd_dot_product_fp8(q_vec, K_token, k_scale, head_dim);
            dot *= scale;

            float m_old = m_i;
            m_i = max(m_i, dot);
            float correction = exp(m_old - m_i);
            float exp_dot = exp(dot - m_i);
            l_i = l_i * correction + exp_dot;
            acc_o = acc_o * correction + exp_dot * float(dequant_fp8(V_token[d], v_scale));
        }
    }

    O[(batch_idx * num_heads + head_idx) * head_dim + d] = half(acc_o / l_i);
}


// MARK: - FlashAttention Decode (GQA-aware, Direct Device Reads, Cooperative Dot)

kernel void flash_decode_f16(
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
    constant uint      &window_size      [[buffer(12)]],
    uint3 gid [[thread_position_in_grid]]
) {
    const uint d           = gid.x;
    const uint kv_head_idx = gid.y;
    const uint batch_idx   = gid.z;
    const uint gqa_ratio   = num_heads / num_kv_heads;
    const uint lane        = d;

    if (d >= head_dim || kv_head_idx >= num_kv_heads || batch_idx >= batch_size) return;

    const uint seq_len = seq_lengths[batch_idx];
    const float scale = 1.0f / sqrt(float(head_dim));
    const uint window_start = (window_size > 0 && seq_len > window_size) ? seq_len - window_size : 0;

    device const int* block_table = block_tables + batch_idx * max_num_blocks;

    uint num_logical_blocks = (seq_len + block_size - 1u) / block_size;
    if (num_logical_blocks > max_num_blocks) num_logical_blocks = max_num_blocks;
    uint start_logical_block = window_start / block_size;

    float m_i_arr[8];
    float l_i_arr[8];
    float acc_o_arr[8];
    for (uint h = 0u; h < gqa_ratio; h++) {
        m_i_arr[h] = -INFINITY;
        l_i_arr[h] = 0.0f;
        acc_o_arr[h] = 0.0f;
    }

    for (uint logical_block = start_logical_block; logical_block < num_logical_blocks; logical_block++) {
        int physical_block = block_table[logical_block];
        device const half* K_block = K_pool + physical_block * block_size * num_kv_heads * head_dim;
        device const half* V_block = V_pool + physical_block * block_size * num_kv_heads * head_dim;

        uint block_start = logical_block * block_size;
        uint block_end = min(block_start + block_size, seq_len);

        for (uint j = block_start; j < block_end; j++) {
            if (j < window_start) continue;
            uint local_j = j - block_start;
            device const half* K_token = K_block + local_j * num_kv_heads * head_dim + kv_head_idx * head_dim;
            device const half* V_token = V_block + local_j * num_kv_heads * head_dim + kv_head_idx * head_dim;

            float k_val = float(K_token[d]);
            float v_val = float(V_token[d]);

            for (uint h = 0u; h < gqa_ratio; h++) {
                const uint head_idx = kv_head_idx * gqa_ratio + h;
                device const half* q_vec = Q + (batch_idx * num_heads + head_idx) * head_dim;

                float partial = float(q_vec[d]) * k_val;
                for (uint stride = 32u; stride < head_dim; stride += 32u) {
                    partial += float(q_vec[d + stride]) * float(K_token[d + stride]);
                }
                partial += simd_shuffle_xor(partial, 1);
                partial += simd_shuffle_xor(partial, 2);
                partial += simd_shuffle_xor(partial, 4);
                partial += simd_shuffle_xor(partial, 8);
                partial += simd_shuffle_xor(partial, 16);
                float dot = partial * scale;

                float m_old = m_i_arr[h];
                m_i_arr[h] = max(m_old, dot);
                float corr = exp(m_old - m_i_arr[h]);
                float exp_dot = exp(dot - m_i_arr[h]);
                l_i_arr[h] = l_i_arr[h] * corr + exp_dot;
                acc_o_arr[h] = acc_o_arr[h] * corr + exp_dot * v_val;
            }
        }
    }

    for (uint h = 0u; h < gqa_ratio; h++) {
        const uint head_idx = kv_head_idx * gqa_ratio + h;
        O[(batch_idx * num_heads + head_idx) * head_dim + d] = half(acc_o_arr[h] / l_i_arr[h]);
    }
}


kernel void flash_decode_f32(
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
    constant uint      &window_size      [[buffer(12)]],
    uint3 gid [[thread_position_in_grid]]
) {
    const uint d           = gid.x;
    const uint kv_head_idx = gid.y;
    const uint batch_idx   = gid.z;
    const uint gqa_ratio   = num_heads / num_kv_heads;
    const uint lane        = d;

    if (d >= head_dim || kv_head_idx >= num_kv_heads || batch_idx >= batch_size) return;

    const uint seq_len = seq_lengths[batch_idx];
    const float scale = 1.0f / sqrt(float(head_dim));
    const uint window_start = (window_size > 0 && seq_len > window_size) ? seq_len - window_size : 0;

    device const int* block_table = block_tables + batch_idx * max_num_blocks;

    uint num_logical_blocks = (seq_len + block_size - 1u) / block_size;
    if (num_logical_blocks > max_num_blocks) num_logical_blocks = max_num_blocks;
    uint start_logical_block = window_start / block_size;

    float m_i_arr[8];
    float l_i_arr[8];
    float acc_o_arr[8];
    for (uint h = 0u; h < gqa_ratio; h++) {
        m_i_arr[h] = -INFINITY;
        l_i_arr[h] = 0.0f;
        acc_o_arr[h] = 0.0f;
    }

    for (uint logical_block = start_logical_block; logical_block < num_logical_blocks; logical_block++) {
        int physical_block = block_table[logical_block];
        device const float* K_block = K_pool + physical_block * block_size * num_kv_heads * head_dim;
        device const float* V_block = V_pool + physical_block * block_size * num_kv_heads * head_dim;

        uint block_start = logical_block * block_size;
        uint block_end = min(block_start + block_size, seq_len);

        for (uint j = block_start; j < block_end; j++) {
            if (j < window_start) continue;
            uint local_j = j - block_start;
            device const float* K_token = K_block + local_j * num_kv_heads * head_dim + kv_head_idx * head_dim;
            device const float* V_token = V_block + local_j * num_kv_heads * head_dim + kv_head_idx * head_dim;

            float k_val = K_token[d];
            float v_val = V_token[d];

            for (uint h = 0u; h < gqa_ratio; h++) {
                const uint head_idx = kv_head_idx * gqa_ratio + h;
                device const float* q_vec = Q + (batch_idx * num_heads + head_idx) * head_dim;

                float partial = q_vec[d] * k_val;
                for (uint stride = 32u; stride < head_dim; stride += 32u) {
                    partial += q_vec[d + stride] * K_token[d + stride];
                }
                partial += simd_shuffle_xor(partial, 1);
                partial += simd_shuffle_xor(partial, 2);
                partial += simd_shuffle_xor(partial, 4);
                partial += simd_shuffle_xor(partial, 8);
                partial += simd_shuffle_xor(partial, 16);
                float dot = partial * scale;

                float m_old = m_i_arr[h];
                m_i_arr[h] = max(m_old, dot);
                float corr = exp(m_old - m_i_arr[h]);
                float exp_dot = exp(dot - m_i_arr[h]);
                l_i_arr[h] = l_i_arr[h] * corr + exp_dot;
                acc_o_arr[h] = acc_o_arr[h] * corr + exp_dot * v_val;
            }
        }
    }

    for (uint h = 0u; h < gqa_ratio; h++) {
        const uint head_idx = kv_head_idx * gqa_ratio + h;
        O[(batch_idx * num_heads + head_idx) * head_dim + d] = acc_o_arr[h] / l_i_arr[h];
    }
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


kernel void kv_cache_scale_fp8(
    device const half  *keys            [[buffer(0)]],
    device const half  *values          [[buffer(1)]],
    device const int   *block_table     [[buffer(2)]],
    constant uint      &token_offset    [[buffer(3)]],
    constant uint      &num_new_tokens  [[buffer(4)]],
    constant uint      &num_kv_heads    [[buffer(5)]],
    constant uint      &head_dim        [[buffer(6)]],
    constant uint      &block_size      [[buffer(7)]],
    device float       *scratch_max     [[buffer(8)]],
    uint3 gid [[thread_position_in_grid]]
) {
    const uint d           = gid.x;
    const uint kv_head_idx = gid.y;
    const uint token_idx   = gid.z;

    if (d >= head_dim || kv_head_idx >= num_kv_heads || token_idx >= num_new_tokens) return;

    uint global_token_pos = token_offset + token_idx;
    uint logical_block = global_token_pos / block_size;
    int physical_block = block_table[logical_block];

    uint input_offset = token_idx * num_kv_heads * head_dim + kv_head_idx * head_dim + d;

    float k_abs = abs(float(keys[input_offset]));
    float v_abs = abs(float(values[input_offset]));

    atomic_max_float(&scratch_max[physical_block * 2 + 0], k_abs);
    atomic_max_float(&scratch_max[physical_block * 2 + 1], v_abs);
}


kernel void kv_cache_append_fp8(
    device const half  *keys            [[buffer(0)]],
    device const half  *values          [[buffer(1)]],
    device uchar       *K_pool          [[buffer(2)]],
    device uchar       *V_pool          [[buffer(3)]],
    device const int   *block_table     [[buffer(4)]],
    constant uint      &token_offset    [[buffer(5)]],
    constant uint      &num_new_tokens  [[buffer(6)]],
    constant uint      &num_kv_heads    [[buffer(7)]],
    constant uint      &head_dim        [[buffer(8)]],
    constant uint      &block_size      [[buffer(9)]],
    device const float *scratch_max     [[buffer(10)]],
    device half        *k_scale_pool    [[buffer(11)]],
    device half        *v_scale_pool    [[buffer(12)]],
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

    float k_max_abs = scratch_max[physical_block * 2 + 0];
    float v_max_abs = scratch_max[physical_block * 2 + 1];

    float existing_k_scale = float(k_scale_pool[physical_block]);
    float existing_k_max = existing_k_scale * 448.0f;
    k_max_abs = max(k_max_abs, existing_k_max);

    float existing_v_scale = float(v_scale_pool[physical_block]);
    float existing_v_max = existing_v_scale * 448.0f;
    v_max_abs = max(v_max_abs, existing_v_max);

    half k_scale = k_max_abs > 0.0f ? half(k_max_abs / 448.0f) : 1.0h;
    half v_scale = v_max_abs > 0.0f ? half(v_max_abs / 448.0f) : 1.0h;

    K_pool[pool_offset] = quantize_fp8(keys[input_offset], k_scale);
    V_pool[pool_offset] = quantize_fp8(values[input_offset], v_scale);

    k_scale_pool[physical_block] = k_scale;
    v_scale_pool[physical_block] = v_scale;
}

// MARK: - Fused KV Cache Append + Prefill

kernel void paged_attention_fused_prefill_f32(
    device const float   *Q            [[buffer(0)]],
    device const float   *raw_K        [[buffer(1)]],
    device const float   *raw_V        [[buffer(2)]],
    device const int     *block_table  [[buffer(3)]],
    device       float   *K_pool       [[buffer(4)]],
    device       float   *V_pool       [[buffer(5)]],
    device       float   *O            [[buffer(6)]],
    constant uint        &seq_len      [[buffer(7)]],
    constant uint        &head_dim     [[buffer(8)]],
    constant uint        &num_heads    [[buffer(9)]],
    constant uint        &num_kv_heads [[buffer(10)]],
    constant uint        &block_size   [[buffer(11)]],
    constant uint        &causal       [[buffer(12)]],
    constant uint        &window_start [[buffer(13)]],
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

    if (row < seq_len && col < head_dim) {
        Q_tile[row_in_tile * head_dim + col] = Q[(row * num_heads + head_idx) * head_dim + col];
    } else if (row_in_tile < block_size && col < head_dim) {
        Q_tile[row_in_tile * head_dim + col] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint num_logical_blocks = (seq_len + block_size - 1) / block_size;

    for (uint k_block = 0; k_block < seq_len; k_block += block_size) {
        uint logical_block = k_block / block_size;
        if (logical_block >= num_logical_blocks) break;
        int physical_block = block_table[logical_block];

        uint local_k_row = lid.y;
        uint global_k_row = k_block + local_k_row;

        if (global_k_row < seq_len && col < head_dim) {
            uint local_token_pos = global_k_row % block_size;
            uint pool_offset = physical_block * block_size * num_kv_heads * head_dim
                             + local_token_pos * num_kv_heads * head_dim
                             + kv_head_idx * head_dim + col;
            uint raw_offset = global_k_row * num_kv_heads * head_dim + kv_head_idx * head_dim + col;
            K_pool[pool_offset] = raw_K[raw_offset];
            V_pool[pool_offset] = raw_V[raw_offset];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float m_i   = -INFINITY;
    float l_i   = 0.0f;
    float acc_o = 0.0f;

    for (uint k_block = window_start; k_block < seq_len; k_block += block_size) {
        uint logical_block = k_block / block_size;
        if (logical_block >= num_logical_blocks) break;
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
                float exp_dot = exp(dot - m_i);
                l_i = l_i * correction + exp_dot;
                acc_o = acc_o * correction;
                acc_o += exp_dot * V_tile[local_j * head_dim + col];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < seq_len && col < head_dim) {
        O[(row * num_heads + head_idx) * head_dim + col] = acc_o / l_i;
    }
}


kernel void paged_attention_fused_prefill_f16(
    device const half    *Q            [[buffer(0)]],
    device const half    *raw_K        [[buffer(1)]],
    device const half    *raw_V        [[buffer(2)]],
    device const int     *block_table  [[buffer(3)]],
    device       half    *K_pool       [[buffer(4)]],
    device       half    *V_pool       [[buffer(5)]],
    device       half    *O            [[buffer(6)]],
    constant uint        &seq_len      [[buffer(7)]],
    constant uint        &head_dim     [[buffer(8)]],
    constant uint        &num_heads    [[buffer(9)]],
    constant uint        &num_kv_heads [[buffer(10)]],
    constant uint        &block_size   [[buffer(11)]],
    constant uint        &causal       [[buffer(12)]],
    constant uint        &window_start [[buffer(13)]],
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

    if (row < seq_len && col < head_dim) {
        Q_tile[row_in_tile * head_dim + col] = float(Q[(row * num_heads + head_idx) * head_dim + col]);
    } else if (row_in_tile < block_size && col < head_dim) {
        Q_tile[row_in_tile * head_dim + col] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint num_logical_blocks = (seq_len + block_size - 1) / block_size;

    for (uint k_block = 0; k_block < seq_len; k_block += block_size) {
        uint logical_block = k_block / block_size;
        if (logical_block >= num_logical_blocks) break;
        int physical_block = block_table[logical_block];

        uint local_k_row = lid.y;
        uint global_k_row = k_block + local_k_row;

        if (global_k_row < seq_len && col < head_dim) {
            uint local_token_pos = global_k_row % block_size;
            uint pool_offset = physical_block * block_size * num_kv_heads * head_dim
                             + local_token_pos * num_kv_heads * head_dim
                             + kv_head_idx * head_dim + col;
            uint raw_offset = global_k_row * num_kv_heads * head_dim + kv_head_idx * head_dim + col;
            K_pool[pool_offset] = raw_K[raw_offset];
            V_pool[pool_offset] = raw_V[raw_offset];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float m_i   = -INFINITY;
    float l_i   = 0.0f;
    float acc_o = 0.0f;

    for (uint k_block = window_start; k_block < seq_len; k_block += block_size) {
        uint logical_block = k_block / block_size;
        if (logical_block >= num_logical_blocks) break;
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
                float exp_dot = exp(dot - m_i);
                l_i = l_i * correction + exp_dot;
                acc_o = acc_o * correction;
                acc_o += exp_dot * V_tile[local_j * head_dim + col];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < seq_len && col < head_dim) {
        O[(row * num_heads + head_idx) * head_dim + col] = half(acc_o / l_i);
    }
}


kernel void paged_attention_fused_prefill_fp8(
    device const half    *Q            [[buffer(0)]],
    device const half    *raw_K        [[buffer(1)]],
    device const half    *raw_V        [[buffer(2)]],
    device const int     *block_table  [[buffer(3)]],
    device       uchar   *K_pool       [[buffer(4)]],
    device       uchar   *V_pool       [[buffer(5)]],
    device       half    *O            [[buffer(6)]],
    constant uint        &seq_len      [[buffer(7)]],
    constant uint        &head_dim     [[buffer(8)]],
    constant uint        &num_heads    [[buffer(9)]],
    constant uint        &num_kv_heads [[buffer(10)]],
    constant uint        &block_size   [[buffer(11)]],
    constant uint        &causal       [[buffer(12)]],
    constant uint        &window_start [[buffer(13)]],
    device       half    *k_scale_pool [[buffer(14)]],
    device       half    *v_scale_pool [[buffer(15)]],
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

    if (row < seq_len && col < head_dim) {
        Q_tile[row_in_tile * head_dim + col] = float(Q[(row * num_heads + head_idx) * head_dim + col]);
    } else if (row_in_tile < block_size && col < head_dim) {
        Q_tile[row_in_tile * head_dim + col] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint num_logical_blocks = (seq_len + block_size - 1) / block_size;

    for (uint k_block = 0; k_block < seq_len; k_block += block_size) {
        uint logical_block = k_block / block_size;
        if (logical_block >= num_logical_blocks) break;
        int physical_block = block_table[logical_block];

        uint local_k_row = lid.y;
        uint global_k_row = k_block + local_k_row;

        uint raw_offset = 0;
        if (global_k_row < seq_len && col < head_dim) {
            raw_offset = global_k_row * num_kv_heads * head_dim + kv_head_idx * head_dim + col;
            K_tile[lid.y * head_dim + col] = fabs(float(raw_K[raw_offset]));
            V_tile[lid.y * head_dim + col] = fabs(float(raw_V[raw_offset]));
        } else if (col < head_dim) {
            K_tile[lid.y * head_dim + col] = 0.0f;
            V_tile[lid.y * head_dim + col] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (lid.x == 0 && lid.y == 0) {
            float bk_max = 0.0f;
            float bv_max = 0.0f;
            uint tile_size = block_size * head_dim;
            for (uint i = 0; i < tile_size; i++) {
                bk_max = max(bk_max, K_tile[i]);
                bv_max = max(bv_max, V_tile[i]);
            }

            float existing_k_max = float(k_scale_pool[physical_block]) * 448.0f;
            float existing_v_max = float(v_scale_pool[physical_block]) * 448.0f;
            bk_max = max(bk_max, existing_k_max);
            bv_max = max(bv_max, existing_v_max);

            half new_k_scale = bk_max > 0.0h ? half(bk_max / 448.0f) : 1.0h;
            half new_v_scale = bv_max > 0.0h ? half(bv_max / 448.0f) : 1.0h;

            K_tile[0] = float(new_k_scale);
            V_tile[0] = float(new_v_scale);

            k_scale_pool[physical_block] = new_k_scale;
            v_scale_pool[physical_block] = new_v_scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (global_k_row < seq_len && col < head_dim) {
            uint local_token_pos = global_k_row % block_size;
            uint pool_offset = physical_block * block_size * num_kv_heads * head_dim
                             + local_token_pos * num_kv_heads * head_dim
                             + kv_head_idx * head_dim + col;

            half k_scale = half(K_tile[0]);
            half v_scale = half(V_tile[0]);

            K_pool[pool_offset] = quantize_fp8(raw_K[raw_offset], k_scale);
            V_pool[pool_offset] = quantize_fp8(raw_V[raw_offset], v_scale);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float m_i   = -INFINITY;
    float l_i   = 0.0f;
    float acc_o = 0.0f;

    for (uint k_block = window_start; k_block < seq_len; k_block += block_size) {
        uint logical_block = k_block / block_size;
        if (logical_block >= num_logical_blocks) break;
        int physical_block = block_table[logical_block];

        device const uchar* K_block_base = K_pool + physical_block * block_size * num_kv_heads * head_dim;
        device const uchar* V_block_base = V_pool + physical_block * block_size * num_kv_heads * head_dim;
        half k_scale_val = k_scale_pool[physical_block];
        half v_scale_val = v_scale_pool[physical_block];

        uint local_k_row = lid.y;
        uint global_k_row = k_block + local_k_row;

        if (global_k_row < seq_len && col < head_dim) {
            K_tile[local_k_row * head_dim + col] = float(dequant_fp8(K_block_base[local_k_row * num_kv_heads * head_dim + kv_head_idx * head_dim + col], k_scale_val));
            V_tile[local_k_row * head_dim + col] = float(dequant_fp8(V_block_base[local_k_row * num_kv_heads * head_dim + kv_head_idx * head_dim + col], v_scale_val));
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
                float exp_dot = exp(dot - m_i);
                l_i = l_i * correction + exp_dot;
                acc_o = acc_o * correction;
                acc_o += exp_dot * V_tile[local_j * head_dim + col];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < seq_len && col < head_dim) {
        O[(row * num_heads + head_idx) * head_dim + col] = half(acc_o / l_i);
    }
}
