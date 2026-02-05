+++
title = "minivllm sequence and blockmanager"
date = 2026-02-02T03:50:13-08:00
draft = false
categories = ["AI-Infra"]
tags = ["AI-Infra"]

description = "Implementation of Sequence and BlockManager"

+++

`` 

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



```python
from enum import Enum, auto
from gc import freeze
from itertools import count


class SequenceStatus(Enum):
    RUNNING = auto()
    WAITING = auto()
    FINISHED = auto()

class Sequence:
    block_size = 256
    counter = count()

    def __init__(self, token_ids: list[int]):
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = token_ids
        self.last_token = token_ids[-1] if token_ids else None
        self.num_tokens = len(token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
        self.num_blocks = 0
        self.block_list = []

    def __len__(self):
        return self.num_tokens

    def __iter__(self):
        return iter(self.token_ids)

    def __getitem__(self, k):
        return self.token_ids[k]

    def __getstate__(self):
        return {
            'seq_id': self.seq_id, 'status': self.status, 'last_token': self.last_token,
            'num_tokens': self.num_tokens, 'num_prompt_tokens': self.num_prompt_tokens,
            'num_cached_tokens': self.num_cached_tokens,'block_list': self.block_list
        }
    def __setstate__(self, state):
        self.seq_id = state['seq_id']
        self.status = state['status']
        self.num_tokens = state['num_tokens']
        self.num_prompt_tokens = state['num_prompt_tokens']
        self.num_cached_tokens = state['num_cached_tokens']
        self.block_list = state['block_list']
        self.last_token = state['last_token']

    def tokens_in_block(self,block_idx):
        assert 0 <= block_idx < self.num_blocks
        if block_idx != self.num_blocks-1:
            return self.token_ids[block_idx*self.block_size:(block_idx+1)*self.block_size]
        else:
            return self.token_ids[block_idx*self.block_size:]

    def append_token(self, token_id):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_cached_tokens += 1

    def reset(self):
        self.block_list = []
        self.num_blocks = 0
        self.num_cached_tokens = 0
        self.status = SequenceStatus.WAITING



```



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



```python
class Block:
    def __init__(self, block_id):
        self.block_id = block_id
        self.token_ids = []
        self.ref_cache = 0
        self.block_hash = None

    def __len__(self):
        return len(self.token_ids)

    def __iter__(self):
        return iter(self.token_ids)

    def __getitem__(self, k):
        return self.token_ids[k]

    def free(self):
        self.ref_cache = 0
        self.token_ids = []
        self.block_hash = None

```



### class BlockManager

**Manages the lifecycle of physical blocks in GPU memory and handles the logical-to-physical mapping for sequences.**

The `BlockManager` optimizes GPU memory by ensuring that no two blocks with the identical context are stored twice. It maintains a pool of available blocks and tracks their usage across multiple active sequences.

So, It acts like the Operating System's Page Manager, the main responsibility of this class is the management(allocate and free) of blocks and the maintaining the table of block_hash

We should pay attention to two aspects of this class:

- **Reference Counting and Copy-on-Write (CoW):** When multiple sequences share a common prefix, they point to the same physical block to optimize KV cache storage. However, as soon as these sequences **diverge**, the `BlockManager` must trigger a **Copy-on-Write** operation. This involves duplicating the shared block into a new physical location to ensure that each sequence has an independent and modifiable memory space, satisfying their unique generation requirements.

  **Maintenance of the Hash Table (Contextual Integrity):** When managing the `hash_table`, we must account for **contextual dependency**. Even if the `token_ids` within two blocks are identical, they may represent distinct states and thus require different `block_hash` values. This is because the KV cache values are **contingent upon** the entire prefix history. Therefore, the `block_hash` must be carefully computed as a chained function of both the current tokens and the preceding block's hash to prevent incorrect cache hits.

| Attributes          | Description                                              |
| ------------------- | -------------------------------------------------------- |
| **`block_size`**    | how many tokens in one block                             |
| **`blocks`**        | the list of all blocks (not block_ids) in the GPU memory |
| **`hash_to_block`** | the dictionary that map block_hash to block_id           |
| **`free_blocks`**   | the blocks that are free                                 |



| Methods                          | Description                                                  |
| -------------------------------- | ------------------------------------------------------------ |
| **`box_block`**                  | Generates a unique deterministic fingerprint for a **full block**. It incorporates the current `token_ids` and the `block_hash` of the preceding block to ensure **contextual uniqueness** |
| **`allocate_for_sequence`**      | The entry point for new sequences. It performs a **prefix cache lookup**; if the prefix hash matches an existing block, it increments the `ref_count`. Otherwise, it claims a new block from the free pool. |
| **`deallocate_for_sequence`**    | Decrements the **`ref_count`** of the associated blocks. If a block's count reaches zero, it is returned to the free pool, though its content might be preserved in the hash table for future "warm" hits. |
| **`manage_blocks_after_append`** | Manages the dynamic growth of a sequence during the **Decode** phase. It determines when a block is fully "boxed" (triggering hash computation) and when a new physical block must be provisioned. |



```python
import hashlib


class BlockManager:
    def __init__(self, block_size, num_blocks):
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.blocks: list[Block] = [Block(i) for i in range(self.num_blocks)]
        self.hash_to_block = dict()
        self.free_blocks = list(range(num_blocks))

    def box_block(self, sequence:Sequence, block_idx):
        assert len(sequence) >= self.block_size * (block_idx+1)
        token_ids = sequence.tokens_in_block(block_idx)

        if block_idx == 0:
            data = str(tuple(token_ids)).encode('utf-8')
            hash = hashlib.sha256(data).hexdigest()
        else:
            pre_hash = self.blocks[sequence.block_list[block_idx-1]].block_hash
            data = (pre_hash+":"+str(tuple(token_ids))).encode('utf-8')
            hash = hashlib.sha256(data).hexdigest()
        if hash in self.hash_to_block:
            self.blocks[self.hash_to_block[hash]].ref_cache += 1
            self.blocks[sequence.block_list[block_idx]].free()
            self.free_blocks.append(sequence.block_list[block_idx])
            sequence.block_list[block_idx] = self.hash_to_block[hash]
        else:
            self.hash_to_block[hash] = sequence.block_list[block_idx]
            self.blocks[sequence.block_list[block_idx]].ref_cache += 1
            self.blocks[sequence.block_list[block_idx]].block_hash = hash


    def allocate_for_sequence(self, sequence:Sequence):
        assert not sequence.block_list
        full_load = 1 if len(sequence)%self.block_size==0 else 0
        blocks_needed = len(sequence)//self.block_size +1-full_load
        if len(self.free_blocks)<blocks_needed:
            return False
        else:
            sequence.num_blocks = blocks_needed
            for i in range(blocks_needed):
                new_block_id = self.free_blocks.pop()
                self.blocks[new_block_id].token_ids = list(sequence.tokens_in_block(i))
                sequence.block_list.append(new_block_id)
                if full_load or i != blocks_needed - 1:
                    self.box_block(sequence, i)
            return True

    def deallocate_for_sequence(self, sequence:Sequence):
        block_freed = 0
        for block_id in sequence.block_list:
            assert self.blocks[block_id].ref_cache > 0, f"Block {block_id} ref_cache is {self.blocks[block_id].ref_cache}, cannot decrement"
            self.blocks[block_id].ref_cache -= 1
            if self.blocks[block_id].ref_cache == 0:
                block_freed += 1
                if self.blocks[block_id].block_hash is not None and self.blocks[block_id].block_hash in self.hash_to_block:
                    del self.hash_to_block[self.blocks[block_id].block_hash]
                self.blocks[block_id].free()
                self.free_blocks.append(block_id)
        sequence.reset()
        return block_freed

    def manage_blocks_after_append(self, sequence:Sequence):
        # after sequence.append_token() manage block allocate
        if len(sequence) % self.block_size == 0:
            self.box_block(sequence, sequence.num_blocks-1)
        elif len(sequence) % self.block_size == 1:
            if len(self.free_blocks) == 0:
                return False
            else:
                new_block_id = self.free_blocks.pop()
                sequence.num_blocks+=1
                sequence.block_list.append(new_block_id)
        else:
            pass
        return True

```



- **manage_blocks_after_append** in **BlockManager** is solely responsible for block encapsulation and allocation; it does not handle appending token_id to the Sequence or adding tokens to the physical blocks.
- The order of **free_blocks** does not affect functionality, so **pop()** and **append()** are used to maintain **O(1)** complexity.
- **manage_blocks_after_append** processes a single token during the generation phase, while **allocate_for_sequence** handles the entire sequence during the prompt phase.
- **box_block** calculates a hash using the **prefix** and **token_ids**. If a match is found (hash hit), the current block is freed and replaced with the index of the block corresponding to the previous hash.
