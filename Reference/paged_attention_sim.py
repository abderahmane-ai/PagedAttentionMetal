"""
PagedAttention Simulation for Batch Size 1

This script simulates the forward attention pass in PagedAttention,
where logical tokens are mapped to physical KV blocks via a block table.
Implements online safe softmax as described in the FlashAttention paper.
"""

import numpy as np


def paged_attention_forward(
    q: np.ndarray,
    k_pool: np.ndarray,
    v_pool: np.ndarray,
    block_table: np.ndarray,
    block_size: int = 4,
    head_dim: int = 8
) -> tuple:
    """
    Simulate PagedAttention forward pass for batch size 1.

    In PagedAttention, the KV cache is managed in fixed-size blocks.
    The block table maps logical block indices to physical block indices
    in the KV cache pool.

    The online softmax algorithm processes key blocks sequentially:
    - For ALL queries, iterate through ALL key blocks
    - Maintain running max (m), sum (l), and output (O)
    - Update using the FlashAttention online softmax rules

    Args:
        q: Query tensor of shape (seq_len, head_dim)
        k_pool: Physical key blocks, shape (num_physical_blocks, block_size, head_dim)
        v_pool: Physical value blocks, shape (num_physical_blocks, block_size, head_dim)
        block_table: Maps logical block -> physical block, shape (num_logical_blocks,)
        block_size: Number of tokens per block
        head_dim: Dimension per attention head

    Returns:
        output: Attention output (seq_len, head_dim)
        m_history: Running max values after each block (for debugging)
        l_history: Running sum values after each block (for debugging)
    """
    seq_len = q.shape[0]
    num_logical_blocks = len(block_table)

    print(f"\n{'='*60}")
    print("PagedAttention Forward Pass (Batch Size = 1)")
    print(f"{'='*60}")
    print(f"Sequence length: {seq_len}")
    print(f"Block size: {block_size}")
    print(f"Head dimension: {head_dim}")
    print(f"Number of logical blocks: {num_logical_blocks}")
    print(f"Block table: {block_table}")
    print(f"{'='*60}\n")

    # Initialize running statistics for online softmax
    # m: per-query running max of scores
    # l: per-query running sum of exp(scores)
    # O: running weighted sum of values
    m = np.full(seq_len, -np.inf)  # (seq_len,)
    l = np.zeros(seq_len)          # (seq_len,)
    O = np.zeros((seq_len, head_dim))  # (seq_len, head_dim)

    m_history = []
    l_history = []

    # Process each logical block of keys
    for logical_block_idx in range(num_logical_blocks):
        physical_block_idx = block_table[logical_block_idx]

        # Token range for this logical block
        start_key = logical_block_idx * block_size
        end_key = min(start_key + block_size, seq_len)
        actual_block_len = end_key - start_key

        print(f"Processing Logical Block {logical_block_idx} (Physical Block {physical_block_idx}):")
        print(f"  Key token range: [{start_key}, {end_key})")
        print()

        # Look up physical key/value block from pool
        k_block = k_pool[physical_block_idx, :actual_block_len, :]  # (block_len, head_dim)
        v_block = v_pool[physical_block_idx, :actual_block_len, :]  # (block_len, head_dim)

        # Compute attention scores: ALL queries @ THIS key block
        # Shape: (seq_len, block_len)
        scores = q @ k_block.T

        print(f"  Scores shape: {scores.shape}")
        print(f"  Scores[0:2, 0:2]:\n{scores[:2, :2]}")
        print()

        # Online safe softmax update
        # Step 1: Find max score per query in this block
        m_block = np.max(scores, axis=1)  # (seq_len,)

        # Step 2: Compute new running max
        m_new = np.maximum(m, m_block)  # (seq_len,)

        # Step 3: Compute rescaling factor
        # exp(m_old - m_new) for previously processed tokens, 0 for first block
        rescale = np.where(m == -np.inf, 0.0, np.exp(m - m_new))  # (seq_len,)

        # Step 4: Compute exp(scores - m_new) for this block
        # Broadcasting: scores (seq_len, block_len) - m_new (seq_len, 1)
        exp_scores = np.exp(scores - m_new[:, np.newaxis])  # (seq_len, block_len)

        # Step 5: Sum of exponentials for this block
        l_block_sum = np.sum(exp_scores, axis=1)  # (seq_len,)

        # Step 6: Rescale previous output
        O = O * rescale[:, np.newaxis]  # (seq_len, head_dim)

        # Step 7: Add contribution from this block
        O_block = exp_scores @ v_block  # (seq_len, head_dim)
        O = O + O_block

        # Step 8: Update running sum
        l = l * rescale + l_block_sum

        # Step 9: Update running max
        m = m_new

        m_history.append(m.copy())
        l_history.append(l.copy())

        print(f"  m_block (max per query): {m_block[:4]}...")
        print(f"  l_block_sum (sum exp per query): {l_block_sum[:4]}...")
        print(f"  Running m (first 4): {m[:4]}...")
        print(f"  Running l (first 4): {l[:4]}...")
        print(f"  Rescale (first 4): {rescale[:4]}...")
        print(f"{'='*60}\n")

    # Final normalization: output = O / l
    output = O / l[:, np.newaxis]

    return output, m_history, l_history


def main():
    # Set random seed for reproducibility
    np.random.seed(42)

    # Configuration
    seq_len = 8
    block_size = 4
    head_dim = 8
    num_physical_blocks = 16
    num_logical_blocks = (seq_len + block_size - 1) // block_size

    print("\n" + "="*60)
    print("PagedAttention Simulation Setup")
    print("="*60)
    print(f"Sequence length: {seq_len}")
    print(f"Block size: {block_size}")
    print(f"Head dimension: {head_dim}")
    print(f"Physical blocks in pool: {num_physical_blocks}")
    print(f"Logical blocks needed: {num_logical_blocks}")
    print("="*60 + "\n")

    # Generate random Q, K, V data
    q = np.random.randn(seq_len, head_dim).astype(np.float32)

    # Physical KV pool - pre-allocated blocks
    k_pool = np.random.randn(num_physical_blocks, block_size, head_dim).astype(np.float32)
    v_pool = np.random.randn(num_physical_blocks, block_size, head_dim).astype(np.float32)

    print("Query tensor Q shape:", q.shape)
    print("Key pool shape:", k_pool.shape)
    print("Value pool shape:", v_pool.shape)
    print()

    # Create block table: maps logical block -> physical block
    # In a real system, this would be dynamic based on allocation
    block_table = np.array([3, 7], dtype=np.int32)

    print("Block Table:")
    for i, pb in enumerate(block_table):
        print(f"  Logical block {i} -> Physical block {pb}")
    print()

    # Run PagedAttention forward pass
    output, m_history, l_history = paged_attention_forward(
        q, k_pool, v_pool, block_table,
        block_size=block_size,
        head_dim=head_dim
    )

    print("\n" + "="*60)
    print("Final Results")
    print("="*60)
    print(f"Output shape: {output.shape}")
    print(f"\nOutput (first 4 rows, first 4 columns):\n{output[:4, :4]}")

    print("\n" + "="*60)
    print("Intermediate m and l Values")
    print("="*60)
    for i, (m_val, l_val) in enumerate(zip(m_history, l_history)):
        print(f"\nBlock {i}:")
        print(f"  m (first 4): {m_val[:4]}...")
        print(f"  l (first 4): {l_val[:4]}...")

    # Verify with standard attention for comparison
    print("\n" + "="*60)
    print("Verification: Standard Attention")
    print("="*60)

    # Reconstruct full K, V from blocks using block table
    k_full = np.zeros((seq_len, head_dim))
    v_full = np.zeros((seq_len, head_dim))

    for logical_idx, physical_idx in enumerate(block_table):
        start = logical_idx * block_size
        end = min(start + block_size, seq_len)
        length = end - start
        k_full[start:end] = k_pool[physical_idx, :length, :]
        v_full[start:end] = v_pool[physical_idx, :length, :]

    # Standard attention: softmax(q @ k.T) @ v
    scores_standard = q @ k_full.T
    m_standard = np.max(scores_standard, axis=1, keepdims=True)
    exp_scores = np.exp(scores_standard - m_standard)
    l_standard = np.sum(exp_scores, axis=1, keepdims=True)
    attention_weights = exp_scores / l_standard
    output_standard = attention_weights @ v_full

    print(f"Standard attention output (first 4 rows, first 4 columns):\n{output_standard[:4, :4]}")

    # Check if outputs match
    diff = np.abs(output - output_standard).max()
    print(f"\nMax difference from standard attention: {diff:.6e}")

    if diff < 1e-5:
        print("✓ PagedAttention output matches standard attention!")
    else:
        print("✗ Warning: Outputs differ")

    print("="*60 + "\n")


if __name__ == "__main__":
    main()
