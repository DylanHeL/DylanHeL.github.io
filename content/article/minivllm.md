+++
title = "minivllm sequence and blockmanager"
date = 2026-02-02T03:50:13-08:00
draft = false
categories = ["AI-Infra"]
tags = ["AI-Infra"]

+++



## Sequence

 The `Sequence` object acts as the **logical glue** between the Scheduler and the GPU Executor. It abstracts a single inference request into a manageable state machine that tracks **generation progress**, manages **KV cache mapping** (via the Block Table), and serves as a **lightweight data carrier** for low-latency Inter-Process Communication.

### sequence

| Attributes              | Description                                                  |
| ----------------------- | ------------------------------------------------------------ |
| **`seq_id`**            | A unique integer identifier for the sequence, generated via a global counter. |
| **`status`**            | the state of the sequence: `WAITING`, `RUNNING`, `FINISHED`  |
| **`token_ids`**         | The complete list of token IDs (Prompt + Generated) belonging to this sequence. |
| **`last_token`**        | the last token_id, **Optimization**: we need this attribute because in the **`RUNNING`** state (generating process), the information of historical tokens are already stored in the KV cache block. so only this attribute is sent during IPC (Inter-Process Communication) via `__getstate__` |
| **`num_tokens`**        | the number of tokens in this sequence(Prompt + Generated)    |
| **`num_prompt_tokens`** | the number of tokens in the original prompt part             |
| **`num_cached_tokens`** | the number of tokens that had been cached                    |
| **`block_table`**       | A mapping table that tracks which physical **Block IDs** are assigned to this sequence |



| Methods               | Description                                                  |
| --------------------- | ------------------------------------------------------------ |
| **`tokens_in_block`** | return the list of tokens in a specific block                |
| **`append_token`**    | append a new token to the sequence                           |
| **`__getstate__`**    | Determining which data to persist and which to discard before the object is serialized and transferred from the scheduler process to the GPU worker process.<br />**Serialization Hook**: Prepares the object for cross-process transfer. It strips the redundant `token_ids` list (except for the `last_token`) to minimize metadata overhead during scheduling |
| **`__setstate__`**    | It serves as the **receiver and inspector** on the GPU side. Upon arrival, it reconstructs the object |



## BlockManager

### class Block

**Represents a physical memory block in the KV cache.**

To ensure **contextual uniqueness**, the hash is computed as a composite function of the current tokens and the **preceding block's hash**. This hierarchical hashing ensures that identical token sequences in different contexts remain distinct.

To maximize **memory utilization**, sequences sharing the same prefix will map to the same physical block. The system uses **reference counting** to track and manage these shared resources.

| Attributes       | Description                                                  |
| ---------------- | ------------------------------------------------------------ |
| **`ref_count`**  | A block is considered **free** only when its **reference count** reaches zero, the ref_count is used to describe the sequences that use the block as a KV cache |
| **`block_id`**   | A unique integer identifier for the physical memory block    |
| **`token_ids`**  | A list storing the specific token IDs assigned to this block |
| **`block_hash`** | The current hash depends on both the tokens inside and the **previous block's hash** |



### class BlockManager

**Manages the lifecycle of physical blocks in GPU memory and handles the logical-to-physical mapping for sequences.**

The `BlockManager` optimizes GPU memory by ensuring that no two blocks with the identical context are stored twice. It maintains a pool of available blocks and tracks their usage across multiple active sequences.

So, It acts like the Operating System's Page Manager, the main responsibility of this class is the management(allocate and free) of blocks and the maintaining the table of block_hash

We should pay attention to two aspects of this class:

- **Reference Counting and Copy-on-Write (CoW):** When multiple sequences share a common prefix, they point to the same physical block to optimize KV cache storage. However, as soon as these sequences **diverge**, the `BlockManager` must trigger a **Copy-on-Write** operation. This involves duplicating the shared block into a new physical location to ensure that each sequence has an independent and modifiable memory space, satisfying their unique generation requirements.

  **Maintenance of the Hash Table (Contextual Integrity):** When managing the `hash_table`, we must account for **contextual dependency**. Even if the `token_ids` within two blocks are identical, they may represent distinct states and thus require different `block_hash` values. This is because the KV cache values are **contingent upon** the entire prefix history. Therefore, the `block_hash` must be carefully computed as a chained function of both the current tokens and the preceding block's hash to prevent incorrect cache hits.

| Attributes          | Description                                    |
| ------------------- | ---------------------------------------------- |
| **`block_size`**    | how many tokens in one block                   |
| **`blocks`**        | the list of all blocks_id in the GPU memory    |
| **`hash_to_block`** | the dictionary that map block_hash to block_id |
| **`free_blocks`**   | the blocks that are free                       |
| **`used_blocks`**   | the blocks that are used                       |



| Methods            | Description                                                  |
| ------------------ | ------------------------------------------------------------ |
| **`compute_hash`** | Generates a unique deterministic fingerprint for a **full block**. It incorporates the current `token_ids` and the `block_hash` of the preceding block to ensure **contextual uniqueness** |
| **`allocate`**     | The entry point for new sequences. It performs a **prefix cache lookup**; if the prefix hash matches an existing block, it increments the `ref_count`. Otherwise, it claims a new block from the free pool. |
| **`deallocate`**   | Decrements the **`ref_count`** of the associated blocks. If a block's count reaches zero, it is returned to the free pool, though its content might be preserved in the hash table for future "warm" hits. |
| **`can_allocate`** | A predictive check to ensure the GPU has sufficient free blocks to accommodate a new sequence or a growth request, preventing runtime **OOM** errors. |
| **`append_token`** | Manages the dynamic growth of a sequence during the **Decode** phase. It determines when a block is fully "sealed" (triggering hash computation) and when a new physical block must be provisioned. |

