## <b>Puzzle 10: Copy Operations</b>

### <b>Learning Objective</b>
Master different strategies for copying data between global and shared memory efficiently.

### <b>Part 10a: Loop-Based Copy</b>

Copy data when you have fewer threads than elements - each thread handles multiple elements.

```py
def p10a():
    """Copy using block_dim to calculate work per thread"""
    
    @cute.kernel
    def copy_kernel(gmem_in: cute.Tensor, gmem_out: cute.Tensor):
        tidx, _, _ = cute.arch.thread_idx()
        bdimx, _, _ = cute.arch.block_dim()
        
        # Allocate shared memory
        smem = cutlass.utils.SmemAllocator()
        shared = smem.allocate_tensor(element_type, tile_size)
        
        # TODO: Calculate elements per thread
        # TODO: Loop to load multiple elements per thread
        # TODO: Synchronize
        # TODO: Write back to global memory
        pass
    
    @cute.jit
    def run_kernel(data_in: cute.Tensor, data_out: cute.Tensor):
        # TODO: Launch with fewer threads than data size
        pass
    
    return run_kernel
```

**Configuration**:
```
Tile size: 256 elements
Threads: 128
Elements per thread: 256 ÷ 128 = 2

Thread 0 handles: elements [0, 128]
Thread 1 handles: elements [1, 129]
Thread 127 handles: elements [127, 255]
```

### <b>Part 10b: Grid-Based Copy</b>

**Challenge**: Use multiple blocks so each thread handles exactly one element.

```py
def p10b():
    """Use enough blocks so each thread handles 1 element"""
    
    @cute.kernel
    def copy_kernel(gmem_in: cute.Tensor, gmem_out: cute.Tensor, total_size: int):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        
        # TODO: Allocate shared memory per block
        # TODO: Calculate global index
        # TODO: Simple 1:1 copy (1 element per thread)
        pass
    
    @cute.jit
    def run_kernel(data_in: cute.Tensor, data_out: cute.Tensor):
        # TODO: Calculate number of blocks needed
        # TODO: Launch with multiple blocks
        pass
    
    return run_kernel
```

**Configuration**:
```
Total size: 256 elements
Threads per block: 128
Number of blocks: ⌈256 ÷ 128⌉ = 2

Block 0, Thread 0: element 0
Block 0, Thread 1: element 1
...
Block 1, Thread 0: element 128
Block 1, Thread 1: element 129
```

### <b>Part 10c: 2D Tile Copy</b>

**Challenge**: Copy a 2D tile with proper coalesced access patterns.

```py
def p10c():
    """Copy a 2D tile to shared memory with proper indexing"""
    
    @cute.kernel
    def copy_2d_kernel(gmem_in: cute.Tensor, gmem_out: cute.Tensor):
        tidx, tidy, _ = cute.arch.thread_idx()
        TILE_M, TILE_N = 16, 32  # 16×32 tile
        
        # TODO: Allocate 2D shared memory
        # TODO: Each thread copies multiple elements
        # TODO: Ensure coalesced access
        pass
    
    @cute.jit
    def run_kernel(data_in: cute.Tensor, data_out: cute.Tensor):
        # TODO: Launch with 2D thread block
        pass
    
    return run_kernel
```

**Configuration**:
```
Tile: 16×32 = 512 elements
Thread block: (16, 8) = 128 threads
Elements per thread: 512 ÷ 128 = 4

Thread (0,0): elements at (0,0), (0,16), (8,0), (8,16)
Thread (1,0): elements at (0,1), (0,17), (8,1), (8,17)
```

### <b>Key Concepts</b>

**Elements Per Thread**: <br>
```python
elements_per_thread = (total_elements + num_threads - 1) // num_threads
# Ceiling division ensures coverage
```

**Loop-Based Pattern**: <br>
```python
for i in range(elements_per_thread):
    idx = thread_id + i * block_dim
    if idx < total_size:
        process(idx)
```

**Global Index Calculation**: <br>
```python
global_idx = block_idx * threads_per_block + thread_idx
```

**Shared Memory Allocation**: <br>
```python
smem = cutlass.utils.SmemAllocator()
shared = smem.allocate_tensor(dtype, shape)
```

**2D Index Calculation**: <br>
```python
row = thread_idy + offset_y
col = thread_idx + offset_x
linear_idx = row * width + col
```

### <b>Loop-Based vs Grid-Based</b>

**Loop-Based (Part 10a)**:
```
Pros:
✓ Fewer blocks → less overhead
✓ Better occupancy with limited threads
✓ Good for small workloads

Cons:
✗ More complex indexing
✗ Each thread does more work
✗ Harder to debug
```

**Grid-Based (Part 10b)**:
```
Pros:
✓ Simple 1:1 mapping
✓ Easier to reason about
✓ Better scalability
✓ More parallelism

Cons:
✗ More blocks → higher overhead
✗ May underutilize GPU
✗ Synchronization across blocks harder
```

### <b>Memory Coalescing in 2D</b>

**Coalesced Access** (Good):
```python
# Threads access consecutive columns
for row in range(my_rows):
    for col_offset in range(0, width, num_threads):
        col = thread_idx + col_offset
        access(row, col)  # Adjacent threads → adjacent memory
```

**Strided Access** (Bad):
```python
# Threads access consecutive rows
for col in range(my_cols):
    for row_offset in range(0, height, num_threads):
        row = thread_idy + row_offset
        access(row, col)  # Adjacent threads → strided memory
```

**Why Coalescing Matters**:
- GPU loads 128-byte cache lines
- 32 threads in a warp should access consecutive addresses
- Coalesced: 1 transaction per 32 threads
- Uncoalesced: Up to 32 transactions per 32 threads

### <b>Bounds Checking</b>

**Always Check Bounds**:
```python
if idx < total_size:
    process(idx)
```

**Why It's Critical**:
```
Example: 256 elements, 128 threads
- Thread 0 handles: 0, 128 ✓
- Thread 127 handles: 127, 255 ✓
- Without check: Thread 0 would access 256 (out of bounds!)
```

**2D Bounds Checking**:
```python
if row < height and col < width:
    shared[row, col] = gmem[row, col]
```

### <b>Synchronization</b>

**Why Sync After Load**:
```python
shared[idx] = gmem[idx]  # Load phase
cute.arch.sync_threads()  # Wait for all threads
result = shared[idx]      # Use phase
```

**Without Sync**:
- Thread 0 might read before Thread 127 writes
- Race condition → incorrect results
- Undefined behavior

**Sync Points**:
1. After loading to shared memory
2. Before reading from shared memory
3. Between phases of computation

### <b>Performance Considerations</b>

**Shared Memory Size**:
```python
# Per block limit: ~48KB
tile_bytes = TILE_M * TILE_N * sizeof(element)
# Must fit in shared memory!
```

**Occupancy**:
```python
# More threads → better hiding latency
# But limited by:
# - Shared memory usage
# - Register usage
# - Thread block size limits (1024 threads)
```

**Work Distribution**:
```python
# Balance:
# - Too many elements/thread → serial bottleneck
# - Too few elements/thread → launch overhead
# Sweet spot: ~4-8 elements per thread
```

### <b>Tips</b>
- Use loop-based for < 1024 elements per block
- Use grid-based for large datasets
- Always ensure coalesced access in innermost loop
- Bound check even with "perfect" division
- Synchronize between load and compute phases
- Calculate `smem` parameter: `elements × bytes_per_element`
- For 2D, make thread_idx the fast-moving dimension
- Test with non-power-of-2 sizes to catch edge cases