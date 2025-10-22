## <b>Puzzle 4: Register Memory</b>

### <b>Learning Objective</b>
Learn to use register memory (fastest GPU memory) for thread-local computations.

<br>

### <b>Challenge</b>
Compute a running sum of 8 consecutive elements per thread using register memory.

```py
def p04():
    """
    Compute running sum using register memory.
    Input: [0, 1, 2, 3, 4, 5, 6, 7, ...]
    Output: [0, 1, 3, 6, 10, 15, 21, 28, ...] (running sum per thread)
    """
    @cute.kernel
    def running_sum_kernel(input_t: cute.Tensor, output_t: cute.Tensor):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        bdim, _, _ = cute.arch.block_dim()
        
        # TODO: Create register buffer layout for 8 elements
        # TODO: Allocate fragment (register memory)
        # TODO: Load 8 elements from global memory
        # TODO: Compute running sum in registers
        # TODO: Write results back to global memory
        pass
    
    @cute.jit
    def run_kernel(input_data: cute.Tensor, output_data: cute.Tensor):
        # TODO: Launch kernel
        pass
    
    return run_kernel
```

**Expected Output (first thread)**: 
```
Input:  [0, 1, 2, 3, 4, 5, 6, 7]
Output: [0, 1, 3, 6, 10, 15, 21, 28]
         ↑  ↑  ↑  ↑   ↑   ↑   ↑   ↑
         0  0+1 1+2 3+3 6+4 10+5 15+6 21+7
```

<br>

### <b>Key Concepts</b>

**Memory Hierarchy Performance**: <br>
- **Registers**: ~1 cycle latency, private per thread, fastest
- **Shared Memory**: ~5 cycles, shared within block
- **Global Memory**: ~500 cycles, accessible by all threads, slowest

**Register Memory in CuTe**: <br>
```python
# 1. Define layout (shape of data in registers)
layout = cute.make_layout((num_elements,))

# 2. Allocate fragment (register storage)
fragment = cute.make_fragment(layout, data_type)

# 3. Access like array: fragment[i]
```

**Running Sum Algorithm**: <br>
```python
# Load
reg[0] = input[base + 0]
reg[1] = input[base + 1]
...

# Accumulate
reg[1] = reg[1] + reg[0]  # reg[1] now contains sum of first 2
reg[2] = reg[2] + reg[1]  # reg[2] now contains sum of first 3
...
```

<br>

### <b>Work Distribution</b>

Each thread processes **8 consecutive elements**:
- Thread 0: elements [0-7]
- Thread 1: elements [8-15]
- Thread 2: elements [16-23]
- ...

```
base_idx = (block_idx * block_dim + thread_idx) * elements_per_thread
```

<br>

### <b>Why Use Registers?</b>

**Speed**: 
- Registers are 100-500× faster than global memory
- Perfect for thread-local computations

**Use Cases**:
- Accumulating values
- Temporary computations
- Small buffers for tiling algorithms

**Limitations**:
- Limited quantity (~255 registers per thread on modern GPUs)
- Not shared between threads
- Spilling to local memory if overused

<br>

### <b>Tips</b>
- Use `cute.make_layout((size,))` for 1D register arrays
- Use `cute.make_fragment(layout, type)` to allocate registers
- Access fragments with standard indexing: `fragment[i]`
- Keep register usage moderate to avoid spilling
- Ideal for small, thread-private working sets
