+++

title = "attention with kvcache"

date = 2026-02-04T05:30:13-08:00

draft = false

categories = ["AI-Infra","unfinished"]

tags = ["AI-Infra","unfinished"]

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

We launch a 2D CUDA grid of size `[batch_size, num_heads]`. Each program (thread block) is responsible for calculating the attention output of **one Query head for one specific sequence**. 

First, we need to clarify our computational workflow.

#### computational workflow.

To respect the GPU's limited SRAM and register capacity, processing the KV cache using a **tiling strategy**:

- **Tiling (kv_block_size)**: We move k v cache data into registers in chunks of `kv_block_size`. While `block_size` defines the storage granularity in VRAM, `kv_block_size` defines the computational granularity for the kernel loop (SRAM/Register usage).
- **Online Softmax Update**: After calculating QK for each chunk, we immediately perform an **Online Softmax** update. By maintaining running statistics (max score and exponential sum), we update the attention accumulator in-place, eliminating the need to store a massive global attention matrix.

##### About The Online Softmax



we calculate q*k for k_cache size of kv_block_size for one time (kv_block_size is the block size for calculation, means the size of k_cache we want to move to the 寄存器, the block_size is the size for storing)

then we calculate online softmax and update after each calculation for one kv_block_size

So what we need to do is to prepare the data for the calculation

for the first part, q is from q_ptr, we use the  q_stride_batch and q_stride_head to get the q_offset for each program_process

k,v are got from k_cache_ptr, v_cache_ptr, using the cache_stride_block, cache_stride_token and the block_id got from block_table_ptr to calculate the kv_cache_offset

for the blocks that is not full of token , we use the mask to make the padding of '-inf' to make sure they have no influence on the calculation

```
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



