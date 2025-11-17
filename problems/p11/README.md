## <b>Puzzle 11: Vectorized Loads</b>

### <b>Learning Objective</b>
Master coalesced memory access patterns for optimal global memory bandwidth utilization.

<br>

### <b>Challenge</b>

Implement efficient memory copying with coalesced access patterns that work across different configurations.

```py
def p11(threads_per_block, num_blocks, tile_size):
    """Coalesced memory access with configurable dimensions"""
    
    @cute.kernel
    def coalesced_kernel(gmem_in: cute.Tensor, gmem_out: cute.Tensor):
        tidx, _, _ = cute.arch.thread_idx()
        bdimx, _, _ = cute.arch.block_dim()
        bidx, _, _ = cute.arch.block_idx()
        
        # TODO: Allocate shared memory
        # TODO: Implement coalesced load pattern
        # TODO: Synchronize threads
        # TODO: Write back with coalesced stores
        pass
    
    @cute.jit
    def run_kernel(data_in: cute.Tensor, data_out: cute.Tensor):
        # TODO: Launch with configuration parameters
        pass
    
    return run_kernel
```

**Test Configurations**:
```
Config 1: 32 threads/block, 1 block, 128 elements
Config 2: 64 threads/block, 1 block, 128 elements
Config 3: 32 threads/block, 2 blocks, 256 elements
Config 4: 128 threads/block, 1 block, 128 elements
```

<br>

### <b>Key Concepts</b>

**Coalesced Memory Access**: <br>
Threads in a warp (32 threads) access consecutive memory addresses in a single transaction.

**Access Pattern**: <br>
```python
# Good: Coalesced
Thread 0: address 0
Thread 1: address 1
Thread 2: address 2
...
Thread 31: address 31
→ 1 memory transaction

# Bad: Strided
Thread 0: address 0
Thread 1: address 32
Thread 2: address 64
...
→ 32 memory transactions
```

**Global Index Formula**: <br>
```python
global_idx = block_idx * tile_size + local_idx
```

**Loop Pattern for Coalescing**: <br>
```python
for i in range(iterations):
    idx = thread_idx + i * block_dim
    access(idx)  # Coalesced across threads
```

<br>

### <b>Memory Coalescing Explained</b>

**GPU Memory Access Granularity**:
- Cache lines: 128 bytes (32 × 4-byte floats)
- Warp size: 32 threads
- Ideal: 32 threads access 32 consecutive floats

**Coalesced Pattern**:
```
Memory: [0][1][2][3]...[31][32][33]...
         ↑  ↑  ↑  ↑       ↑   ↑   ↑
Thread:  0  1  2  3  ...  31  0   1  ...
         └─────────┘       └───────┘
         Warp 0, iter 0    Warp 0, iter 1

Result: 2 memory transactions for 64 elements
```

**Non-Coalesced Pattern (Strided)**:
```
Memory: [0]...[31][32]...[63]
         ↑       ↑   ↑       ↑
Thread:  0       1   0       1
         
Each thread accesses non-consecutive addresses
Result: Up to 32× more transactions!
```

<br>

### <b>Access Pattern Anatomy</b>

**Iteration 0**:
```
idx = thread_idx + 0 * block_dim
Thread 0:  idx = 0
Thread 1:  idx = 1
...
Thread 31: idx = 31
→ Perfect coalescing
```

**Iteration 1**:
```
idx = thread_idx + 1 * block_dim
Thread 0:  idx = 0 + 128 = 128
Thread 1:  idx = 1 + 128 = 129
...
Thread 31: idx = 31 + 128 = 159
→ Still coalesced!
```

<br>

### <b>Bandwidth Calculation</b>

**Theoretical Peak**:
- Modern GPUs: ~1-2 TB/s global memory bandwidth
- Requires perfect coalescing

**Effective Bandwidth**:
```python
bytes_transferred = size * sizeof(element) * 2  # Read + write
time_seconds = measured_time
bandwidth = bytes_transferred / time_seconds

# Example:
# 256 elements × 4 bytes × 2 = 2048 bytes
# Time: 0.01 ms = 10⁻⁵ seconds
# Bandwidth: 2048 / 10⁻⁵ = 204 MB/s
```

**Coalescing Efficiency**:
```python
efficiency = actual_bandwidth / theoretical_bandwidth

Good coalescing: >80% efficiency
Bad coalescing: <20% efficiency
```

<br>

### <b>Configuration Trade-offs</b>

**Few Threads, Many Iterations**:
```
32 threads × 4 iterations = 128 elements

Pros:
✓ Lower thread launch overhead
✓ Better register availability

Cons:
✗ Each thread does more work
✗ Serialization within thread
```

**Many Threads, Few Iterations**:
```
128 threads × 1 iteration = 128 elements

Pros:
✓ Maximum parallelism
✓ Simpler code (no loop)

Cons:
✗ Higher launch overhead
✗ More shared memory contention
```

**Multiple Blocks**:
```
32 threads × 2 blocks = 64 concurrent threads

Pros:
✓ Better SM utilization
✓ More work in flight

Cons:
✗ No cross-block synchronization
✗ Shared memory per-block overhead
```

<br>

### <b>Bounds Checking Importance</b>

**Local Bounds**:
```python
if idx < tile_size:
    # Within shared memory bounds
```

**Global Bounds**:
```python
if global_idx < total_size:
    # Within global memory bounds
```

**Why Both Are Needed**:
```
Tile size: 128
Total size: 256
Block 1:
  idx = 127 < 128 ✓ (local OK)
  global_idx = 127 < 256 ✓ (global OK)
  
Block 2:
  idx = 127 < 128 ✓ (local OK)
  global_idx = 255 < 256 ✓ (global OK)

Without global check:
  Block 2, idx=128 would access global[256] ✗
```

<br>

### <b>Dynamic Size Handling</b>

**Reading Tensor Size**:
```python
total_size = gmem_in.shape[0]  # Runtime size
```

**Why Dynamic is Better**:
- No hardcoded sizes
- Works with variable input
- Single kernel for multiple sizes
- More flexible and reusable

**Static vs Dynamic**:
```python
# Static (inflexible)
def kernel():
    SIZE = 256  # Compile-time constant
    
# Dynamic (flexible)
def kernel(tensor: cute.Tensor):
    size = tensor.shape[0]  # Runtime value
```

<br>

### <b>Shared Memory Usage</b>

**Allocation**:
```python
smem = cutlass.utils.SmemAllocator()
shared = smem.allocate_tensor(dtype, tile_size)
```

**Launch Parameter**:
```python
smem = tile_size * bytes_per_element
# For float32: tile_size * 4
```

**Per-Block Allocation**:
- Each block gets its own shared memory
- Total shared memory = tile_size × num_blocks
- Limited by hardware (typically 48KB-164KB per SM)

<br>

### <b>Performance Optimization Tips</b>

**Alignment**:
```python
# Align loads to cache line boundaries
# Use tile sizes that are multiples of 32
TILE_SIZE = 128  # 4 cache lines ✓
TILE_SIZE = 100  # Misaligned ✗
```

**Warp Efficiency**:
```python
# Use thread counts that are multiples of 32
THREADS = 128  # 4 warps ✓
THREADS = 100  # 3.125 warps, wastes threads ✗
```

**Minimize Divergence**:
```python
# Avoid conditional execution within warp
# Structure loops so all threads in warp execute same path
for i in range(fixed_iterations):  # ✓
    if global_idx < size:          # ✓ (uniform check)
        process()
```

<br>

### <b>Common Pitfalls</b>

**Strided Access**:
```python
# Bad: Thread i accesses element i*stride
idx = tidx * STRIDE  # ✗

# Good: Thread i accesses element i
idx = tidx + offset * block_dim  # ✓
```

**Missing Bounds Check**:
```python
# Dangerous
shared[idx] = gmem[global_idx]

# Safe
if idx < tile_size and global_idx < total_size:
    shared[idx] = gmem[global_idx]
```

**Incorrect Synchronization**:
```python
# Wrong: Sync inside conditional
if tidx == 0:
    cute.arch.sync_threads()  # ✗ Deadlock!

# Correct: All threads must sync
cute.arch.sync_threads()  # ✓
```

<br>

### <b>Tips</b>
- Always ensure coalesced access: adjacent threads → adjacent memory
- Use `thread_idx + offset * block_dim` pattern
- Check both local and global bounds
- Test with various configurations (threads, blocks, sizes)
- Measure bandwidth utilization to verify coalescing
- Align tile sizes to cache line boundaries (multiples of 32)
- Keep thread counts as multiples of 32 (warp size)
- Use dynamic sizing for flexibility