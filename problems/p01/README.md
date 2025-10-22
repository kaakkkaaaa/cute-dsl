## <b>Puzzle 1: Thread and Block Indexing</b>

### <b>Learning Objective</b>
Understand the GPU's hierarchical execution model: grids, blocks, threads, and how to calculate global thread indices.

<br>

### <b>Part 1a: Understanding the Hierarchy</b>

Visualize how GPU organizes threads into blocks and blocks into grids.

```py
def p01a():
    @cute.kernel
    def thread_id_kernel():
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        bdim, _, _ = cute.arch.block_dim()
        # TODO: Calculate global thread index
        # TODO: Print block, thread, and global ID
        pass
    
    @cute.jit
    def run_thread_id():
        # TODO: Launch with multiple blocks and threads
        pass
    
    return run_thread_id
```

<br>

### <b>Part 1b: Global Indexing</b>

**Challenge**: Fill a 1D array where each thread writes its global index to the array.

```py
def p01b():
    """
    Create a kernel that fills a 1D array with thread indices.
    Each thread should write its global thread index to the array.
    """
    @cute.kernel
    def fill_indices(arr: cute.Tensor):
        # TODO: Calculate global thread index
        # TODO: Write to array
        pass
    
    @cute.jit
    def run_kernel(arr: cute.Tensor):
        # TODO: Launch with correct grid/block dims
        pass
    
    return run_kernel
```

**Expected Output**: `[0, 1, 2, 3, ..., 31]`

<br>

### <b>Part 1c: 2D Indexing</b>

**Challenge**: Copy a 4×8 2D tensor using proper 2D thread indexing.

```py
def p01c():
    """
    Copy 2D tensor (4x8) from input to output using proper 2D indexing.
    
    Input:  [[1, 2, 3, ...], [9, 10, 11, ...], ...]
    Output: Same as input
    """
    @cute.kernel
    def copy_2d_kernel(output: cute.Tensor, input: cute.Tensor):
        tidx, tidy, _ = cute.arch.thread_idx()
        # TODO: Copy input[row, col] to output[row, col]
        pass
    
    @cute.jit
    def run_kernel(output: cute.Tensor, input: cute.Tensor):
        # TODO: Launch with 2D thread layout
        pass
    
    return run_kernel
```

<br>

### <b>Key Concepts</b>

**GPU Hierarchy**:
- **Grid**: Collection of thread blocks
- **Block**: Group of threads that can cooperate (up to 1024 threads)
- **Thread**: Individual execution unit

**CuTe DSL Functions**: <br>
`cute.arch.thread_idx()`: Thread index within block (tidx, tidy, tidz) <br>
`cute.arch.block_idx()`: Block index within grid (bidx, bidy, bidz) <br>
`cute.arch.block_dim()`: Number of threads per block dimension <br>

**Global Thread Index Formula**:
```python
global_id = block_idx * block_dim + thread_idx
```

**2D Indexing**:
- Use `tidx, tidy` for column and row
- Access tensors with `tensor[row, col]`

<br>

### <b>Tips</b>
- Launch with `grid=(num_blocks, 1, 1)` and `block=(threads_per_block, 1, 1)` for 1D problems
- For 2D: arrange threads as `block=(COLS, ROWS, 1)` to match tensor layout
- Always ensure `num_blocks * threads_per_block >= array_size`
