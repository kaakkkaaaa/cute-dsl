## <b>Puzzle 3: Shared Memory</b>

### <b>Learning Objective</b>
Learn to use shared memory for fast inter-thread communication within a block.

<br>

### <b>Challenge</b>
Reverse the array within each block using shared memory.

```py
def p03():
    """
    Reverse array within each block using shared memory.
    Input: [0, 1, 2, ..., 127]
    Output: [31, 30, ..., 0, 63, 62, ..., 32, ...] (each block reversed)
    """
    @cute.kernel
    def reverse_block_kernel(input_t: cute.Tensor, output_t: cute.Tensor):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        
        block_size = 32
        
        # TODO: Allocate shared memory
        # TODO: Load from global memory to shared memory
        # TODO: Synchronize threads
        # TODO: Write reversed data back to global memory
        pass
    
    @cute.jit
    def run_kernel(input_data: cute.Tensor, output_data: cute.Tensor):
        # TODO: Launch with smem parameter
        pass
    
    return run_kernel
```

**Expected Output**: 
```
Block 0: [31, 30, 29, ..., 2, 1, 0]
Block 1: [63, 62, 61, ..., 34, 33, 32]
Block 2: [95, 94, 93, ..., 66, 65, 64]
Block 3: [127, 126, 125, ..., 98, 97, 96]
```

<br>

### <b>Key Concepts</b>

**Memory Hierarchy**: <br>
- **Global Memory**: Slow (~500 cycles), large, accessible by all threads
- **Shared Memory**: Fast (~5 cycles), small, shared within a block
- **Registers**: Fastest, private to each thread

**Shared Memory Allocation**: <br>
```python
smem = cutlass.utils.SmemAllocator()
shared = smem.allocate_tensor(element_type, size)
```

**Thread Synchronization**: <br>
`cute.arch.sync_threads()`: Ensures all threads in block reach this point before continuing <br>

**Launch with Shared Memory**: <br>
```python
kernel().launch(
    grid=...,
    block=...,
    smem=bytes_needed  # Size in bytes
)
```

<br>

### <b>Algorithm</b>

1. **Load**: Each thread loads one element from global → shared memory
2. **Sync**: Wait for all threads to finish loading
3. **Reverse**: Read from shared memory in reverse order
4. **Write**: Write to global memory

```
Thread 0: shared[0] = input[0]     →  output[0] = shared[31]
Thread 1: shared[1] = input[1]     →  output[1] = shared[30]
...
Thread 31: shared[31] = input[31]  →  output[31] = shared[0]
```

<br>

### <b>Why Synchronization?</b>

Without `cute.arch.sync_threads()`:
- Thread 0 might read `shared[31]` before Thread 31 writes to it
- **Race condition** → incorrect results

With synchronization:
- All threads finish loading before any thread starts reading
- **Correct execution** guaranteed

<br>

### <b>Tips</b>
- Shared memory size calculation: `num_elements * bytes_per_element`
- For float32: `32 elements * 4 bytes = 128 bytes`
- Always synchronize between read and write phases
- Shared memory is limited (typically 48KB per block)
