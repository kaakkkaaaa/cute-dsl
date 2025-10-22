## <b>Puzzle 7: Coordinate Systems</b>

### <b>Learning Objective</b>
Master hierarchical (nested) coordinates in CuTe - a powerful abstraction for tiling and blocking.

<br>

### <b>Part 7a: Hierarchical Coordinates Basics</b>

Understand how nested shapes represent blocked data structures.

```py
def p07a():
    """
    Hierarchical coordinates
    """
    @cute.jit
    def coord_kernel():
        # Hierarchical shape ((2, 3), 4)
        # This means: 4 blocks, each block has 2×3 elements
        shape = ((2, 3), 4)
        layout = cute.make_layout(shape)
        
        # TODO: Explore hierarchical coordinates
        # TODO: Map coordinates to indices
        # TODO: Map indices back to coordinates
        pass
    
    return coord_kernel
```

**Hierarchical Coordinate Structure**:
```
Shape: ((2, 3), 4)
       └─┬──┘  └┬┘
    Inner block  Outer (# of blocks)

Coordinates: ((inner_row, inner_col), block_idx)
Example: ((1, 2), 3) = row 1, col 2, in block 3
```

<br>

### <b>Part 7b: Hierarchical vs Flat Layouts</b>

**Challenge**: Compare hierarchical and flat representations of the same data.

```py
def p07b():
    """
    Understanding Hierarchical (Nested) Layout
    """
    @cute.jit
    def hierarchical_layout_kernel():
        # Hierarchical: 3 blocks of 2×4 each
        hier_shape = ((2, 4), 3)
        
        # Equivalent flat: 2×4×3 tensor
        flat_shape = (2, 4, 3)
        
        # TODO: Create both layouts
        # TODO: Fill with sequential data
        # TODO: Compare coordinate mappings
        pass
    
    return hierarchical_layout_kernel
```

<br>

### <b>Key Concepts</b>

**Hierarchical Shape Syntax**: <br>
```python
shape = ((inner_dims...), outer_dims...)
# ((2, 3), 4) = 4 blocks, each 2×3
# ((M, N), (P, Q)) = P×Q blocks, each M×N
```

**Coordinate Format**: <br>
```python
coord = ((inner_indices...), outer_indices...)
# ((1, 2), 3) = position (1,2) in block 3
```

**Layout Functions**: <br>
`cute.crd2idx(coord, layout)`: Convert coordinate to linear index <br>
`layout.get_hier_coord(idx)`: Convert linear index to hierarchical coordinate <br>
`layout.shape`: Get the shape tuple <br>
`layout.stride`: Get the stride tuple <br>

<br>

### <b>Why Hierarchical Coordinates?</b>

**Natural Tiling Representation**:
```
Matrix divided into tiles:
┌───────┬───────┬───────┐
│ Tile0 │ Tile1 │ Tile2 │  Each tile is 2×3
│  2×3  │  2×3  │  2×3  │  Total: 3 tiles
└───────┴───────┴───────┘

Hierarchical: ((2, 3), 3)
Flat: (2, 3, 3) or (2, 9) - less intuitive!
```

**Thread Block Mapping**:
```python
# Map thread blocks to tiles naturally
tile_shape = (16, 16)
num_tiles = (grid_m, grid_n)
global_shape = (tile_shape, num_tiles)
```

**Recursive Structure**:
```python
# Can nest arbitrarily deep
shape = (((2, 2), 3), 4)  # 4 super-blocks of (3 blocks of 2×2)
```

<br>

### <b>Coordinate Mapping Examples</b>

**Example Layout**: `((2, 3), 4)`
```
Block 0: positions 0-5    (rows 0-1, cols 0-2)
Block 1: positions 6-11   (rows 0-1, cols 0-2)
Block 2: positions 12-17  (rows 0-1, cols 0-2)
Block 3: positions 18-23  (rows 0-1, cols 0-2)
```

**Coordinate Mappings**:
```python
((0, 0), 0) → index 0   # First element of first block
((1, 2), 0) → index 5   # Last element of first block
((0, 0), 1) → index 6   # First element of second block
((1, 2), 3) → index 23  # Last element of last block
```

<br>

### <b>Hierarchical vs Flat</b>

**Semantically Different**:
```python
# Hierarchical: "3 blocks of 2×4"
hier = ((2, 4), 3)
coord_hier = ((row, col), block)

# Flat: "2 rows × 4 cols × 3 depth"
flat = (2, 4, 3)
coord_flat = (row, col, depth)
```

**Access Patterns**:
```python
# Hierarchical: natural for blocked algorithms
for block in range(num_blocks):
    for row in range(block_rows):
        for col in range(block_cols):
            access(((row, col), block))

# Flat: natural for dimension-wise traversal
for d in range(depth):
    for r in range(rows):
        for c in range(cols):
            access((r, c, d))
```

<br>

### <b>Use Cases</b>

**Tiled Matrix Multiplication**:
```python
# Divide M×N matrix into tiles
tile_size = 32
shape = ((tile_size, tile_size), (M//32, N//32))
```

**Hierarchical Thread Organization**:
```python
# Warp (32 threads) within thread block (256 threads)
shape = (32, 8)  # 8 warps per block
```

**Multi-level Blocking**:
```python
# L2 cache tiles → L1 cache tiles → registers
shape = (((reg_tile), l1_tiles), l2_tiles)
```

<br>

### <b>Tips</b>
- Hierarchical coordinates are purely logical - no data movement
- Use for algorithms that naturally work on tiles/blocks
- Inner dimensions = what each block looks like
- Outer dimensions = how many blocks
- Same total elements, different semantics
- Essential for understanding CuTe's advanced features (partitioning, tiling)
