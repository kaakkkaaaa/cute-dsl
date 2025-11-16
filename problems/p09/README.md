## <b>Puzzle 9: Identity Tensors</b>

### <b>Learning Objective</b>
Learn to use identity tensors for debugging, visualizing memory layouts, and verifying tiling strategies.


### <b>Part 9a: Basic Identity Tensor</b>

Understand what an identity tensor is and how to create one.

```py
def p09a():
    @cute.jit
    def identity_kernel():
        # Create identity tensor for 4×8 matrix
        shape = (4, 8)
        identity = cute.make_identity_tensor(shape)
        
        # TODO: Print the identity tensor
        # TODO: Understand what values it contains
        # TODO: See the coordinate pattern
        pass
    
    return identity_kernel
```

**Identity Tensor Visualization**:
```
Shape: (4, 8)
Identity tensor stores coordinates at each position:

(0,0) (0,1) (0,2) (0,3) (0,4) (0,5) (0,6) (0,7)
(1,0) (1,1) (1,2) (1,3) (1,4) (1,5) (1,6) (1,7)
(2,0) (2,1) (2,2) (2,3) (2,4) (2,5) (2,6) (2,7)
(3,0) (3,1) (3,2) (3,3) (3,4) (3,5) (3,6) (3,7)

At position [i,j], the value is the coordinate (i,j)
```

<br>

### <b>Part 9b: Verifying Tiling Strategy</b>

**Challenge**: Use identity tensors to verify and debug tiling patterns.

```py
def p09b():
    @cute.jit
    def verify_tiling_kernel():
        # 8×8 matrix divided into 2×2 grid of 4×4 tiles
        full_shape = (8, 8)
        full_identity = cute.make_identity_tensor(full_shape)
        tile_size = 4
        
        # TODO: Print each tile separately
        # TODO: Verify tile boundaries
        # TODO: Confirm no overlap or gaps
        pass
    
    return verify_tiling_kernel
```

**Tiling Visualization**:
```
8×8 matrix divided into 4 tiles of 4×4:

Tile (0,0):          Tile (0,1):
(0,0) (0,1) (0,2) (0,3) | (0,4) (0,5) (0,6) (0,7)
(1,0) (1,1) (1,2) (1,3) | (1,4) (1,5) (1,6) (1,7)
(2,0) (2,1) (2,2) (2,3) | (2,4) (2,5) (2,6) (2,7)
(3,0) (3,1) (3,2) (3,3) | (3,4) (3,5) (3,6) (3,7)
--------------------|--------------------
Tile (1,0):          Tile (1,1):
(4,0) (4,1) (4,2) (4,3) | (4,4) (4,5) (4,6) (4,7)
(5,0) (5,1) (5,2) (5,3) | (5,4) (5,5) (5,6) (5,7)
(6,0) (6,1) (6,2) (6,3) | (6,4) (6,5) (6,6) (6,7)
(7,0) (7,1) (7,2) (7,3) | (7,4) (7,5) (7,6) (7,7)
```

<br>

### <b>Key Concepts</b>

**Identity Tensor**: <br>
`cute.make_identity_tensor(shape)`: Creates a tensor where each element stores its own coordinate <br>

**What It Contains**: <br>
```python
identity[i, j] = (i, j)  # The coordinate itself!
identity[2, 5] = (2, 5)
identity[0, 0] = (0, 0)
```

**Access Pattern**: <br>
```python
# Get coordinate at position [i, j]
coord = identity[i, j]
row = coord[0]    # i
col = coord[1]    # j
```

<br>

### <b>Why Identity Tensors?</b>

**1. Debugging Memory Layouts**:
```python
# See exactly which coordinate maps to which memory location
layout = make_layout(shape, stride=custom_stride)
identity = make_identity_tensor(shape)

# Apply layout transformation
transformed = apply_layout(identity, layout)
# Now you can see where each coordinate ended up!
```

**2. Visualizing Partitions**:
```python
# Check if your thread partitioning is correct
identity = make_identity_tensor(global_shape)
thread_partition = partition(identity, thread_layout)

# Print what each thread sees
print(thread_partition[thread_id])
```

**3. Verifying Tiling**:
```python
# Make sure tiles cover entire matrix without gaps/overlaps
identity = make_identity_tensor(matrix_shape)

# Extract each tile
for tile_i in range(num_tiles_m):
    for tile_j in range(num_tiles_n):
        tile = extract_tile(identity, tile_i, tile_j)
        print(f"Tile ({tile_i},{tile_j}):", tile)
```

**4. Understanding Transformations**:
```python
# See the effect of swizzling, permutations, etc.
original = make_identity_tensor(shape)
swizzled = apply_swizzle(original, swizzle_pattern)
# Compare original vs swizzled coordinates
```

<br>

### <b>Common Use Cases</b>

**Debugging Thread-to-Data Mapping**:
```python
# Which element does each thread access?
@cute.kernel
def debug_kernel(data: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    
    # Create identity to see coordinate mapping
    identity = cute.make_identity_tensor(data.shape)
    my_coord = identity[tidx]
    
    cute.printf("Thread {} accesses coordinate ({}, {})\n",
                tidx, my_coord[0], my_coord[1])
```

**Verifying Tile Coverage**:
```python
# Ensure all tiles together cover entire matrix
identity = make_identity_tensor((M, N))

for tile_id in range(num_tiles):
    tile_region = get_tile_region(identity, tile_id)
    print(f"Tile {tile_id} covers: {tile_region}")
```

**Understanding Layout Transformations**:
```python
# Original coordinates
original = make_identity_tensor((4, 8))

# After transpose layout
transposed_layout = make_layout((8, 4), stride=(1, 8))
transposed = apply_transform(original, transposed_layout)

# See which original coordinate maps where
```

<br>

### <b>Identity Tensor Properties</b>

**Self-Referential**:
```python
# The value at each position IS the position
identity[i, j] == (i, j)  # Always true!
```

**Coordinate Preservation**:
```python
# After transformations, you can trace back origins
transformed = apply_layout(identity, complex_layout)
original_coord = transformed[new_i, new_j]
# original_coord tells you where this came from!
```

**Lightweight**:
```python
# Identity tensors don't store actual data
# They compute coordinates on-the-fly
# Zero memory overhead for large tensors
```

<br>

### <b>Debugging Workflow</b>

**Step 1**: Create identity tensor matching your data shape
```python
identity = cute.make_identity_tensor(data_shape)
```

**Step 2**: Apply your layout transformation
```python
transformed = apply_your_transformation(identity)
```

**Step 3**: Print and inspect
```python
cute.print_tensor(transformed)
# See exactly which coordinates go where
```

**Step 4**: Verify correctness
```python
# Check for:
# - Complete coverage (no gaps)
# - No overlaps
# - Expected patterns (coalescing, etc.)
```

<br>

### <b>Tips</b>
- Identity tensors are pure metadata - no actual storage
- Use for debugging before implementing with real data
- Perfect for visualizing complex tiling strategies
- Helps catch off-by-one errors in index calculations
- Essential for understanding CuTe's advanced partitioning
- Print small regions to avoid overwhelming output
- Compare expected vs actual coordinate patterns