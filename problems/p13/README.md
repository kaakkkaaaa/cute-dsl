## <b>Puzzle 13: Partitioning</b>

### <b>Learning Objective</b>
Master partitioning - dividing data among threads where each thread processes multiple elements. Learn both manual loop-based partitioning and CuTe's `zipped_divide` for structured data access.

### <b>Challenge</b>
Process a 128-element array by distributing 4 elements to each of 32 threads. Implement both manual partitioning and CuTe's `zipped_divide` approach.

```py
def p13a():
    """Loop-based partitioning"""
    
    @cute.kernel
    def partition_kernel(tensor: cute.Tensor):
        tidx, _, _ = cute.arch.thread_idx()
        
        # TODO: Allocate shared memory for partial sums
        # TODO: Loop over VALUES_PER_THREAD elements
        # TODO: Calculate global index: tidx * VALUES_PER_THREAD + i
        # TODO: Accumulate local sum
        # TODO: Store in shared memory and reduce
        pass
    
    @cute.jit
    def run_kernel(data: cute.Tensor):
        # TODO: Launch with THREADS_PER_BLOCK threads
        pass
    
    return run_kernel

def p13b():
    """Zipped Divide partitioning"""
    
    @cute.kernel
    def partition_kernel(tensor: cute.Tensor):
        tidx, _, _ = cute.arch.thread_idx()
        
        # TODO: Use zipped_divide to create 2D view
        # TODO: Access elements with simple 2D indexing
        pass
    
    return run_kernel
```

**Configuration**:
```
Total Size: 128 elements
Threads per Block: 32 threads
Values per Thread: 4 elements
Total Work: 32 × 4 = 128 elements
Layout: Row-major (32,4):(4,1)
```

### <b>Key Concepts</b>

**Partitioning vs. Tiling**: <br>
Partitioning divides work logically among threads (each thread processes multiple elements). Tiling divides data spatially into blocks.

**Loop-Based Partitioning**: <br>
```python
for i in range(VALUES_PER_THREAD):
    idx = tidx * VALUES_PER_THREAD + i
    local_sum += tensor[idx]
```

**Zipped Divide**: <br>
```python
tiled = cute.zipped_divide(tensor, (THREADS_PER_BLOCK,))
# Creates (32,4):(4,1) layout - row-major
local_sum += tiled[(tidx, i)]  # Clean 2D indexing!
```

**Work Distribution**: <br>
Each thread owns a contiguous stripe of elements <br>
Thread 0: elements [0-3] <br>
Thread 1: elements [4-7] <br>
Thread 31: elements [124-127]

### <b>Partitioning Visualization</b>

**Work Assignment**:
```
128 elements divided among 32 threads:

Thread 0:  [0][1][2][3]
Thread 1:  [4][5][6][7]
Thread 2:  [8][9][10][11]
...
Thread 31: [124][125][126][127]

Each thread processes 4 consecutive elements
```

**Memory Layout with zipped_divide**:
```
Original 1D Tensor (128,):(1,)
         ↓ zipped_divide(tensor, (32,))
2D View (32,4):(4,1) - row-major

         Col 0  Col 1  Col 2  Col 3
Thread 0:  [0]    [1]    [2]    [3]
Thread 1:  [4]    [5]    [6]    [7]
Thread 2:  [8]    [9]   [10]   [11]
...
Thread 31:[124]  [125]  [126]  [127]

Access pattern: tiled[(thread_id, element_id)]
```

**Execution Flow**:
```
Step 1: Each thread computes local sum
  Thread 0: sum(0,1,2,3)
  Thread 1: sum(4,5,6,7)
  ...
  Thread 31: sum(124,125,126,127)

Step 2: Store partial sums in shared memory
  shared[0] = local_sum_0
  shared[1] = local_sum_1
  ...
  shared[31] = local_sum_31

Step 3: Thread 0 reduces all partial sums
  total = shared[0] + shared[1] + ... + shared[31]
```

### <b>Coordinate Calculation</b>

**Manual Loop-Based (p13a)**:
```python
# Calculate global index manually
idx = tidx * VALUES_PER_THREAD + i

# Example for Thread 5, iteration 2:
idx = 5 * 4 + 2 = 22
element = tensor[22]
```

**Zipped Divide (p13b)**:
```python
# Create structured 2D view
tiled = cute.zipped_divide(tensor, (32,))
# Shape: (32,4) - 32 threads, 4 elements each
# Stride: (4,1) - row-major, contiguous within thread

# Access with 2D coordinates
element = tiled[(tidx, i)]

# Example for Thread 5, iteration 2:
element = tiled[(5, 2)]  # Maps to tensor[22]
```

### <b>Comparison: p13a vs p13b</b>

| Aspect | p13a: Loop-Based | p13b: Zipped Divide |
|--------|------------------|---------------------|
| Index Calculation | Manual: `tidx * VALUES_PER_THREAD + i` | Automatic 2D indexing |
| Code Clarity | Explicit but verbose | Clean and structured |
| Layout Control | Manual | Handled by CuTe |
| Flexibility | Full control | Constrained to CuTe patterns |
| Error Prone | Easy to make indexing mistakes | Less error-prone |

### <b>Memory Access Pattern</b>

**Sequential Per Thread**:
```
Thread 0 accesses: [0] → [1] → [2] → [3] (sequential)
Thread 1 accesses: [4] → [5] → [6] → [7] (sequential)
...
All threads access their assigned elements sequentially
```

**Coalescing**:
```
At iteration i=0:
  Thread 0 → [0]
  Thread 1 → [4]
  Thread 2 → [8]
  ...
  NOT coalesced! Stride of 4 elements between threads

At iteration i=1:
  Thread 0 → [1]
  Thread 1 → [5]
  Thread 2 → [9]
  ...
  Still not coalesced
```

**Note**: This partitioning pattern prioritizes sequential access within each thread over coalesced access across threads. Good for reduction operations where each thread accumulates independently.

### <b>Expected Output</b>
```
Configuration:
Total Size: 128
Threads per Block: 32
Values per Thread: 4
Expected Sum: <sum of all 128 elements>

--- p13a: Loop-based partitioning ---
Total Sum (Loop-Based): <sum>
✅ p13a complete

--- p13b: zipped_divide ---
Sum: <sum>
✅ p13b complete
```

### <b>Real-World Applications</b>
- **Reduction Operations**: Each thread reduces its partition, then combine
- **Histogram Computation**: Partition data for parallel histogram building
- **Data Preprocessing**: Divide dataset for parallel normalization/transformation
- **Stream Processing**: Partition continuous streams for parallel processing

### <b>Key Takeaways</b>
1. **Partitioning** distributes multiple elements to each thread
2. **Manual indexing** gives full control but is error-prone
3. **`zipped_divide`** provides structured 2D views for cleaner code
4. **Row-major layout** (32,4):(4,1) keeps each thread's data contiguous
5. **Trade-off**: Sequential per-thread access vs. coalesced across-thread access