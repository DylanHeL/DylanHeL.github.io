+++

title = "attention with kvcache"

date = 2026-02-04T05:30:13-08:00

draft = false

categories = ["AI-Infra","unfinished"]

math= true

tags = ["AI-Infra","unfinished"]

description: "Simple Implementation of Attention with KV Cache in Triton and Performance Comparison with Flash-Decoding"

+++



### simple_pytorch_version

#### Workflow

This version is designed for conceptual clarity and performance comparison.

the `k_cache` and `v_cache` are represented as two large tensors with the shape `[num_blocks, block_size, num_kv_heads, head_dim]`

- For each batch, we calculate the number of blocks required for each sequence based on `cache_seqlens` and `block_size`

-  We use the `block_table` to map the logical block index (`block_idx`) to the physical block ID (`block_id`), which serves as the index for the first dimension of the `k_cache` and `v_cache`.

- then use the block_id to get tokens from the k_cache and v_cache

- concatenate the blocks and use `.view()` or `.reshape()` to restore them to the size `[seq_len, num_kv_heads, head_dim]`.

- In cases where Grouped-Query Attention (GQA) is used (meaning one KV head is shared by multiple Q heads), we apply `repeat_interleave` to the `k_seq` and `v_seq` to match the dimensions of the `q`.

#### Deficiencies 

##### 1. Redundant Data Movement & Bandwidth Bottleneck (Main)

The frequent use of `torch.cat` and `repeat_interleave` forces **physical data copying** within the HBM. In the Decode phase, where computational density is extremely low, the GPU spends the vast majority of its time moving data rather than performing calculations. Furthermore, the constant allocation and deallocation of temporary tensors result in **memory fragmentation**, compromising the long-term stability of the system.

##### 2. Kernel Launch Overhead & Latency Accumulation

Due to the lack of **kernel fusion**, every discrete step ( `view`, `einsum`, `softmax`) triggers an independent CUDA kernel launch. Python-level `for` loops force these kernels to be launched serially.

##### 3. Insufficient Parallelism & Compute Under-utilization

PyTorch operators lack the  task partitioning capabilities necessary for `batch=1` workloads. They cannot shard a single long-sequence task across all Streaming Multiprocessors of the GPU, failing to leverage the massive parallel compute power of GPU. And The existence of `for` loops prevents the batch dimension from being parallelized at the hardware level.

```python
def attn_decode_simple_pytorch(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cache_seqlens: torch.Tensor,
    block_table: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    batch_size, _, num_heads, head_dim = q.shape
    block_size = k_cache.shape[1]
    num_kv_heads = k_cache.shape[2] 
    num_queries_per_kv = num_heads 
    
    outputs = []
    
    for b in range(batch_size):
        seq_len = cache_seqlens[b].item()
        blocks = block_table[b]
        num_blocks = (seq_len + block_size - 1) // block_size
        
        k_seq = []
        v_seq = []
        
        for block_idx in range(num_blocks):
            block_id = blocks[block_idx].item()
            if block_id == -1:
                break
            
            k_block = k_cache[block_id]
            v_block = v_cache[block_id]
            
            if block_idx == num_blocks - 1:
                valid_tokens = seq_len - block_idx * block_size
                k_block = k_block[:valid_tokens]
                v_block = v_block[:valid_tokens]
            
            k_seq.append(k_block)
            v_seq.append(v_block)
        
        k_seq = torch.cat(k_seq, dim=0)
        v_seq = torch.cat(v_seq, dim=0)
        
        k_seq = k_seq.view(seq_len, num_kv_heads, head_dim)
        v_seq = v_seq.view(seq_len, num_kv_heads, head_dim)
        
        if num_queries_per_kv > 1:
            k_seq = k_seq.repeat_interleave(num_queries_per_kv, dim=1)
            v_seq = v_seq.repeat_interleave(num_queries_per_kv, dim=1)
        
        q_b = q[b, 0]
        
        scores = torch.einsum('hd,shd->hs', q_b, k_seq) * softmax_scale
        attn_weights = torch.softmax(scores, dim=-1)
        output_b = torch.einsum('hs,shd->hd', attn_weights, v_seq)
        
        outputs.append(output_b)
    
    output = torch.stack(outputs, dim=0)
    return output.unsqueeze(1)

```



### pytorch_version

The baseline for performance comparison, compared to the `simple_pytorch_version`, this version utilizes `F.scaled_dot_product_attention` (SDPA) instead of the discrete `einsum` + `softmax` chain.

This method implements Kernel Fusion for the computation phase. By keeping intermediate results within registers or L1/L2 cache, it minimizes redundant memory access. Furthermore, it dynamically selects the most efficient algorithm (such as FlashAttention or Memory-Efficient Attention) based on the underlying hardware, specifically optimizing the computational component.

No improvement in Addressing & Movement and Workflow Deficiencies

```python

        # Reshape for PyTorch SDPA
        q_b = q[b:b+1].transpose(1, 2)  # [1, num_heads, 1, head_dim]
        k_b = k_seq.unsqueeze(0).transpose(1, 2)  # [1, num_heads, seq_len, head_dim]
        v_b = v_seq.unsqueeze(0).transpose(1, 2)
        
        output_b = F.scaled_dot_product_attention(
            q_b, k_b, v_b,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
            scale=softmax_scale
        )
```



### flash_triton_version

#### About The Online Softmax

The idea of online softmax comes from such a observation:  The fundamental bottleneck in traditional Attention mechanisms is not the raw computational FLOPs, but rather the **Memory** Wall

In a standard implementation, Softmax is a 3-pass operation in GPU

- Read all scores to find the global maximum 
- Read all scores again to calculate the sum of exponentials
- Read all scores a third time to normalize and multiply by V

$$
m = \max(x_i), \quad d = \sum e^{x_i - m}, \quad o = \sum \left( \frac{e^{x_i - m}}{d} \cdot v_i \right)
$$



According to this logic, the intermediate data generated by each shader must be stored, and operations on the overall sequence can only begin after all computations are complete. This “Read-Write-Read” cycle forces the GPU to constantly swap data in and out of HBM, resulting in extremely low utilization of the GPU's computational power, with significant time wasted on data movement.

To overcome this, we use an incremental update approach. Instead of storing and moving all intermediate results, we compute Softmax online, If we have two blocks of data, we can compute their partial Softmax results independently and then merge them.

assume that we have two shard of data

for the first shard **(indices 1 to k):** We compute the local maximum, the local denominator , and the unnormalized weighted sum :
$$
m^{(1)} = \max_{1 \le i \le k} x_i ,\quad d^{(1)} = \sum_{i=1}^k e^{x_i - m^{(1)}},\quad \mathbf{o}^{(1)} = \sum_{i=1}^k \left( e^{x_i - m^{(1)}} \cdot v_i \right)
$$
**For the second chunk (indices k+1 to n):** Similarly, we compute the local statistics for the next segment:
$$
m^{(2)} = \max_{k+1 \le i \le n} x_i,\quad d^{(2)} = \sum_{i=k+1}^n e^{x_i - m^{(2)}},\quad \mathbf{o}^{(2)} = \sum_{i=k+1}^n \left( e^{x_i - m^{(2)}} \cdot v_i \right)
$$
**Merging the States:**We can merge these two partial results into a global state without re-scanning the original logits. The core logic relies on a Rescaling Factor to align the previous results with the new global maximum:
$$
m_{new} = \max(m^{(1)}, m^{(2)})\\
d_{new} = d^{(1)} \cdot e^{m^{(1)} - m_{new}} + d^{(2)} \cdot e^{m^{(2)} - m_{new}}\\
\mathbf{o}_{new} = \mathbf{o}^{(1)} \cdot e^{m^{(1)} - m_{new}} + \mathbf{o}^{(2)} \cdot e^{m^{(2)} - m_{new}}
$$
In other words, we can implement the update based solely on the intermediate data of each shard, without waiting for the entire sequence to be available.

In a practical Triton kernel implementation, we only need to maintain three status variables in the registers:
$$
m_{curr} ,\quad d_{curr},\quad o_{curr}
$$
Using the incremental update algorithm described above, we can update these states in real-time. This approach offers significant advantages:

- **Register Residency:** Intermediate results (Attention Scores) can be discarded immediately after updating m, d and o, completely eliminating the need to write them back to HBM.
- **Parallel Reduction:** In **Split-K (Flash-Decoding)** scenarios, each split calculates a local (m, d, o). The final Combine Kernel then applies the merging formulas to yield a result mathematically identical to the global computation.

This tiled computation is **fully equivalent** to traditional Attention, while reducing memory access complexity from O(N^2) to O(N), effectively breaking through the Memory Wall.



#### computational workflow.

We launch a 2D CUDA grid of size `[batch_size, num_heads]`. Each program (thread block) is responsible for calculating the attention output of **one Query head for one specific sequence**. 

First, we need to clarify our computational workflow.

To respect the GPU's limited SRAM and register capacity, processing the KV cache using a **tiling strategy**:

- **Tiling (kv_block_size)**: We move k v cache data into registers in chunks of `kv_block_size`. While `block_size` defines the storage granularity in VRAM, `kv_block_size` defines the computational granularity for the kernel loop (SRAM/Register usage).
- **Online Softmax Update**: After calculating QK for each chunk, we immediately perform an **Online Softmax** update. By maintaining running statistics (max score and exponential sum), we update the attention accumulator in-place, eliminating the need to store a massive global attention matrix.



#### Data Orchestration and Fetching Logic

Once the computational flow is established, the focus shifts to data movement. To maximize throughput, we orchestrate the transfer of data from High Bandwidth Memory (HBM) to GPU registers using a tiled indexing strategy.

#### 1. Query Vector Indexing

Each program instance (thread block) is assigned to a specific Query Head. We compute the absolute memory offset for the Query using its batch and head strides:

- **Offset Calculation:** The `q_offset` is determined by `(pid_batch * q_stride_batch) + (pid_head * q_stride_head) + offs_d`.
- **Persistence:** Since the Query remains constant throughout the lifetime of a single kernel execution, it is loaded into registers once and reused across all subsequent KV iterations.

#### 2. Paged KV Cache Retrieval

Because we utilize a **PagedAttention** architecture, the KV cache is stored in non-contiguous physical blocks. We must resolve logical token positions into physical memory addresses:

- **Block Mapping:** For each token index in the sequence, we identify its logical block via `token_idx // block_size`.
- **Physical Lookup:** We fetch the corresponding physical `block_id` from the `block_table_ptr`.
- **Offset Resolution:** The final `kv_cache_offset` is derived by combining the `block_id` with the token's relative position within that block (`token_idx % block_size`) and the specific head's stride.

#### 3. Tiling and Vectorized Loading

To balance computational intensity with register pressure, we move data in tiles of `kv_block_size`. This is the granularity at which we move K and V tensors into the registers for calculation:

- **Granularity:** While `block_size` defines how data is *stored* in memory, `kv_block_size` defines the chunk size for *computation*.
- **Masking and Padding:** For blocks that are not fully occupied by valid tokens (e.g., at the end of a sequence), we apply a boolean mask.
  - Invalid positions in the K cache are padded with `-inf` to ensure they contribute zero weight during the `exp` stage of the Softmax.
  - Invalid positions in the V cache are masked to `0` to prevent them from corrupting the weighted sum accumulation.

#### 4. The Iterative Update Loop

With the data prepared, the kernel executes an iterative loop:

1. **Fetch:** Vectorized load of the next `kv_block_size` chunk of K and V data.
2. **Compute:** Perform the dot product between Q and the K-tile.
3. **Update:** Immediately apply the **Online Softmax** logic to update the running state (m, d, o) before moving to the next chunk.

```python
@triton.jit
def flash_attn_decode_kernel(
    q_ptr, q_stride_batch, q_stride_head,
    k_cache_ptr, v_cache_ptr,
    cache_stride_block, cache_stride_token,
    block_table_ptr, block_table_stride,
    seqlens_ptr,
    out_ptr, out_stride_batch, out_stride_head,
    num_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    kv_block_size: tl.constexpr,
    softmax_scale: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    
    num_queries_per_kv = num_heads // num_kv_heads
    kv_head_idx = pid_head // num_queries_per_kv
    
    seq_len = tl.load(seqlens_ptr + pid_batch)
    
    cols = tl.arange(0, head_dim)
    q_offset = pid_batch * q_stride_batch + pid_head * q_stride_head + cols
    q = tl.load(q_ptr + q_offset, mask=cols < head_dim)
    
    m_i = float('-inf')
    l_i = 0.0
    acc = tl.zeros([head_dim], dtype=tl.float32)
    
    for kv_start in range(0, seq_len, kv_block_size):
        offs_n = kv_start + tl.arange(0, kv_block_size)
        mask_n = offs_n < seq_len
        
        cache_block_indices = offs_n // block_size
        tokens_in_block = offs_n % block_size
        
        bt_offsets = pid_batch * block_table_stride + cache_block_indices
        block_ids = tl.load(block_table_ptr + bt_offsets, mask=mask_n, other=-1)
        
        k_v_offsets = (block_ids[:, None] * cache_stride_block +
                       tokens_in_block[:, None] * cache_stride_token +
                       kv_head_idx * head_dim +
                       cols[None, :])
        
        k = tl.load(k_cache_ptr + k_v_offsets, mask=mask_n[:, None], other=0.0)
        v = tl.load(v_cache_ptr + k_v_offsets, mask=mask_n[:, None], other=0.0)
        
        qk = tl.sum(q[None, :] * k, axis=1) * softmax_scale
        qk = tl.where(mask_n, qk, float('-inf'))
        
        m_ij = tl.max(qk)
        m_i_new = tl.maximum(m_i, m_ij)
        
        p = tl.exp(qk - m_i_new)
        alpha = tl.exp(m_i - m_i_new)
        
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        l_i = l_i * alpha + tl.sum(p)
        m_i = m_i_new
    
    acc = acc / l_i
    out_offset = pid_batch * out_stride_batch + pid_head * out_stride_head + cols
    tl.store(out_ptr + out_offset, acc, mask=cols < head_dim)


def flash_attn_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cache_seqlens: torch.Tensor,
    block_table: torch.Tensor,
    softmax_scale: float,
    kv_block_size: int = 32,
) -> torch.Tensor:
    batch_size, _, num_heads, head_dim = q.shape
    block_size = k_cache.shape[1]
    num_kv_heads = k_cache.shape[2] // head_dim
    
    output = torch.empty_like(q)
    q_squeezed = q.squeeze(1)
    out_squeezed = output.squeeze(1)
    
    grid = (batch_size, num_heads)
    
    flash_attn_decode_optimized_kernel[grid](
        q_squeezed, q_squeezed.stride(0), q_squeezed.stride(1),
        k_cache, v_cache,
        k_cache.stride(0), k_cache.stride(1),
        block_table, block_table.stride(0),
        cache_seqlens,
        out_squeezed, out_squeezed.stride(0), out_squeezed.stride(1),
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=block_size,
        kv_block_size=kv_block_size,
        softmax_scale=softmax_scale,
    )
    
    return output
```



### split_k_version

The idea behind split-k is that when the batch size is 1 but the sequence length is extremely long, assigning a single sequence to only one program process clearly results in insufficient parallelism, as shown in the figure below.

![img](/posts/parallelization.gif)

The long sequence is then split into k segments to further enhance parallelism, and a reduction step is incorporated to update the final output.

![img](/posts/parallelization_kv.gif)

```python
@triton.jit
def flash_attn_split_k_kernel(
    q_ptr, k_cache_ptr, v_cache_ptr,
    block_table_ptr, seqlens_ptr,
    mid_out_ptr, mid_lse_ptr,
    q_stride_batch, q_stride_head,
    cache_stride_block, cache_stride_token,
    block_table_stride,
    num_heads: tl.constexpr, num_kv_heads: tl.constexpr, head_dim: tl.constexpr,
    block_size: tl.constexpr, kv_block_size: tl.constexpr,
    num_splits: tl.constexpr, softmax_scale: tl.constexpr,
):
    # get the idx
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    pid_split = tl.program_id(2)

    # GQA
    num_queries_per_kv = num_heads // num_kv_heads
    kv_head_idx = pid_head // num_queries_per_kv

    # the token range for current split
    seq_len = tl.load(seqlens_ptr + pid_batch)
    tokens_per_split = (seq_len + num_splits - 1) // num_splits
    start_token = pid_split * tokens_per_split
    end_token = tl.minimum(start_token + tokens_per_split, seq_len)

  	# blank shard
    if start_token >= end_token:
        cols = tl.arange(0, head_dim)
        split_offset = (pid_batch * num_heads * num_splits + pid_head * num_splits + pid_split)
        tl.store(mid_out_ptr + split_offset * head_dim + cols, tl.zeros([head_dim], dtype=tl.float32))
        tl.store(mid_lse_ptr + split_offset, float('-inf'))
        return

    # load q
    cols = tl.arange(0, head_dim)
    q_offset = pid_batch * q_stride_batch + pid_head * q_stride_head + cols
    q = tl.load(q_ptr + q_offset, mask=cols < head_dim)

    m_i = float('-inf')
    l_i = 0.0
    acc = tl.zeros([head_dim], dtype=tl.float32)

    for kv_start in range(start_token, end_token, kv_block_size):
        offs_n = kv_start + tl.arange(0, kv_block_size)
        mask_n = offs_n < end_token

        # PagedAttention block_idx -> block_id
        bt_offsets = pid_batch * block_table_stride + (offs_n // block_size)
        block_ids = tl.load(block_table_ptr + bt_offsets, mask=mask_n, other=-1)

        k_v_offsets = (block_ids[:, None] * cache_stride_block + 
                       (offs_n % block_size)[:, None] * cache_stride_token + 
                       kv_head_idx * head_dim + cols[None, :])

        k = tl.load(k_cache_ptr + k_v_offsets, mask=mask_n[:, None], other=0.0)
        v = tl.load(v_cache_ptr + k_v_offsets, mask=mask_n[:, None], other=0.0)

        # Online Update and dot
        qk = tl.sum(q[None, :] * k, axis=1) * softmax_scale
        qk = tl.where(mask_n, qk, float('-inf'))

        m_ij = tl.max(qk)
        m_i_new = tl.maximum(m_i, m_ij)
        p = tl.exp(qk - m_i_new)
        alpha = tl.exp(m_i - m_i_new)

        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        l_i = l_i * alpha + tl.sum(p)
        m_i = m_i_new

    split_offset = (pid_batch * num_heads * num_splits + pid_head * num_splits + pid_split)
    tl.store(mid_out_ptr + split_offset * head_dim + cols, acc)
    tl.store(mid_lse_ptr + split_offset, m_i + tl.log(l_i))
```

 

```python
@triton.jit
def flash_attn_combine_kernel(
    mid_out_ptr, mid_lse_ptr, out_ptr,
    out_stride_batch, out_stride_head,
    num_heads: tl.constexpr, num_splits: tl.constexpr, head_dim: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    cols = tl.arange(0, head_dim)
    base_offset = (pid_batch * num_heads + pid_head) * num_splits

    # Pass1:find the maxLSE
    m_max = float('-inf')
    for s in range(num_splits):
        lse = tl.load(mid_lse_ptr + base_offset + s)
        m_max = tl.maximum(m_max, lse)

    # Pass2:Calculate the normalization factor and perform weighted summation.
    sum_exp = 0.0
    acc = tl.zeros([head_dim], dtype=tl.float32)
    for s in range(num_splits):
        lse = tl.load(mid_lse_ptr + base_offset + s)
        weight = tl.exp(lse - m_max)
        sum_exp += weight
        
        split_out_offset = (base_offset + s) * head_dim + cols
        split_out = tl.load(mid_out_ptr + split_out_offset)
        acc += weight * split_out

    # normalized
    out_offset = pid_batch * out_stride_batch + pid_head * out_stride_head + cols
    tl.store(out_ptr + out_offset, acc / sum_exp)
```

