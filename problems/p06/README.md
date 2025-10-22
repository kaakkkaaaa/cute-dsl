## <b>Puzzle 6: Layouts</b>

### <b>Learning Objective</b>
Master CuTe's layout system: understand how shape and stride define memory access patterns.

<br>

### <b>Part 6a: Row-Major vs Column-Major</b>

Compare different memory layouts for the same logical tensor.

```py
def p06a():
    """
    Compare column-major and row-major layouts
    """
    @cute.jit
    def layout_kernel():
        shape = (3, 4)  # 3 rows, 4 columns
        
        # TODO: Create column-major layout (default)
        # TODO: Create row-major layout (explicit stride)
        # TODO: Fill both with sequential data
        # TODO: Compare coordinate-to-index mapping
        pass
    
    return layout_kernel
```

**Memory Layouts**:
```
Logical view: 3×4 matrix
┌─────────────┐
│ 0  1  2  3  │
│ 4  5  6  7  │
│ 8  9 10 11  │
└─────────────┘

Column-major (stride=(1,3)): [0,4,8, 1,5,9, 2,6,10, 3,7,11]
Row-major (stride=(4,1)):    [0,1,2,3, 4,5,6,7, 8,9,10,11]
```

<br>

### <b>Part 6b: 3D Layouts</b>

**Challenge**: Explore 3D tensors with custom stride patterns.

```py
def p06b():
    """
    Explore 3D Layout with custom strides
    """
    @cute.jit
    def layout_kernel(source_tensor: cute.Tensor):
        shape = (2, 3, 4)  # 2×3×4 tensor
        
        # TODO: Create default 3D layout
        # TODO: Create custom 3D layout with specific strides
        # TODO: Fill tensor and test coordinate mapping
        pass
    
    return layout_kernel
```

<br>

### <b>Key Concepts</b>

**Layout = Shape + Stride**: <br>
```python
layout = cute.make_layout(shape, stride=stride)
```

**Shape**: Logical dimensions <br>
- `(4, 3)` → 4 rows, 3 columns
- `(2, 3, 4)` → 2 slices, 3 rows, 4 columns

**Stride**: Memory traversal pattern <br>
- How many elements to skip for each dimension
- `(3, 1)` → skip 3 for next row, 1 for next column

**Coordinate to Index**: <br>
`cute.crd2idx(coordinate, layout)`: Maps logical position to memory offset <br>

**Layout Functions**: <br>
`cute.make_layout(shape)`: Create layout (default stride) <br>
`cute.make_layout(shape, stride=...)`: Create with explicit stride <br>
`cute.print_tensor(tensor)`: Visualize tensor contents <br>

<br>

### <b>Column-Major vs Row-Major</b>

**Column-Major (Fortran/MATLAB style)**:
```python
stride = (1, num_rows)
# Memory: Column 0, Column 1, Column 2, ...
# Good for: Column-wise operations
```

**Row-Major (C/NumPy style)**:
```python
stride = (num_cols, 1)
# Memory: Row 0, Row 1, Row 2, ...
# Good for: Row-wise operations
```

**Index Calculation**:
```python
# For coordinate (row, col):
offset = row * stride[0] + col * stride[1]
```

<br>

### <b>3D Stride Patterns</b>

**Default 3D Layout** `(2, 3, 4)`:
```python
stride = (1, 2, 6)  # Column-major extended
# Vary fastest → slowest: dimension 0 → 1 → 2
```

**Custom 3D Layout**:
```python
shape = (2, 3, 4)
stride = (12, 4, 1)  # Row-major style
# Slice → Row → Column ordering
```

<br>

### <b>Why Layouts Matter?</b>

**Memory Coalescing**:
- Contiguous memory access is faster
- Layout determines which operations are efficient

**Algorithm Optimization**:
- Matrix transpose: just change layout!
- No data movement needed for transpose view

**Tiling & Blocking**:
- Advanced layouts enable efficient tiling
- Foundation for high-performance GEMM

**Hardware Alignment**:
- Match GPU memory access patterns
- Minimize wasted bandwidth

<br>

### <b>Tips</b>
- Default CuTe layouts are column-major
- Use explicit `stride=` for row-major layouts
- Visualize with `cute.print_tensor()` to understand layout
- Same data, different layout = different memory pattern
- Choose layout based on access pattern for best performance
