## <b>Puzzle 2: Global Memory Access</b>

### <b>Learning Objective</b>
Learn how to safely read and write global memory with proper bounds checking.

<br>

### <b>Challenge</b>
Write a kernel that adds 10 to each element of a 128-element array in parallel.

```py
def p02():
    """
    Add 10 to each element in parallel.
    Input: [0, 1, 2, ..., 127]
    Output: [10, 11, 12, ..., 137]
    """
    @cute.kernel
    def add_10_kernel(tensor: cute.Tensor):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        bdim, _, _ = cute.arch.block_dim()
        # TODO: Calculate global index
        # TODO: Check bounds
        # TODO: Add 10 to tensor[global_idx]
        pass
    
    @cute.jit
    def run_kernel(data: cute.Tensor):
        # TODO: Launch kernel with 4 blocks, 32 threads each
        pass
    
    return run_kernel
```

**Expected Output**: 
```
Input:  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, ...]
Output: [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, ...]
```

<br>

### <b>Key Concepts</b>

**Global Memory**: <br>
- Largest but slowest GPU memory
- Accessible by all threads across all blocks
- Requires bounds checking to avoid out-of-bounds access

**Bounds Checking**: <br>
`if global_idx < cute.size(tensor)`: Prevents writing beyond array bounds <br>

**Memory Access Pattern**: <br>
```python
global_idx = block_idx * block_dim + thread_idx
tensor[global_idx] = new_value  # Read-modify-write
```

**cute.size()**: Returns the total number of elements in a tensor <br>

<br>

### <b>Why Bounds Checking?</b>

In this puzzle:
- Total threads: 4 blocks × 32 threads = **128 threads**
- Array size: **128 elements**

Perfect match! But in general:
- If `num_threads > array_size`: Extra threads must be disabled
- Without bounds checking: **Undefined behavior** (crashes or corrupt data)

<br>

### <b>Tips</b>
- Always use `if global_idx < cute.size(tensor)` before memory access
- Launch enough threads to cover the entire array: `num_threads ≥ array_size`
- Use `cute.size(tensor)` to get tensor dimensions at runtime
