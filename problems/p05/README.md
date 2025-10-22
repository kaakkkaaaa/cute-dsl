## <b>Puzzle 5: Tensor Basics</b>

### <b>Learning Objective</b>
Understand CuTe's fundamental abstraction: tensors with shape and layout.

<br>

### <b>Challenge</b>
Learn how to create, inspect, and understand CuTe tensors.

```py
def p05():
    """
    Create and access tensors
    """
    @cute.kernel
    def tensor_demo():
        tidx, _, _ = cute.arch.thread_idx()
        if tidx == 0:
            # TODO: Create a tensor and print its properties
            # TODO: Understand shape and layout
            pass
    
    @cute.jit
    def run_kernel():
        # TODO: Launch kernel
        pass
    
    return run_kernel
```

<br>

### <b>Key Concepts</b>

**What is a Tensor?**: <br>
A tensor in CuTe is data + metadata:
- **Pointer**: Where the data lives in memory
- **Layout**: How to interpret the data (shape + stride)

**Tensor Properties**: <br>
`cute.size(tensor)`: Total number of elements <br>
`cute.size(tensor, mode=i)`: Size along dimension i <br>
`cute.shape(tensor)`: Complete shape tuple <br>
`cute.stride(tensor)`: Stride information <br>

**Creating Tensors**: <br>
```python
# From PyTorch
torch_tensor = torch.randn(4, 8, device='cuda')
cute_tensor = from_dlpack(torch_tensor)

# In kernel - accessing passed tensors
@cute.kernel
def my_kernel(tensor: cute.Tensor):
    # tensor is already a CuTe tensor
    value = tensor[i, j]  # Access elements
```

<br>

### <b>Tensor Indexing</b>

**1D Tensor**:
```python
tensor[i]  # Access element at position i
```

**2D Tensor**:
```python
tensor[row, col]  # Access element at (row, col)
```

**Multi-dimensional**:
```python
tensor[i, j, k, ...]  # Access with multiple indices
```

<br>

### <b>Shape vs Layout</b>

**Shape**: Logical dimensions
```python
shape = (4, 8)  # 4 rows, 8 columns
```

**Stride**: How to traverse memory
```python
# Row-major (C-style): [0,1,2,3,4,5,6,7, 8,9,10,...]
stride = (8, 1)  # Move 8 elements for next row, 1 for next column

# Column-major (Fortran-style): [0,4,8,12, 1,5,9,13, ...]
stride = (1, 4)  # Move 1 element for next row, 4 for next column
```

**Memory Address Calculation**:
```python
address = base + row * stride[0] + col * stride[1]
```

<br>

### <b>Why CuTe Tensors?</b>

**Flexibility**:
- Arbitrary shapes: `(4, 8, 16)`, `((2, 3), 4)`, etc.
- Custom layouts: Optimize memory access patterns
- Hierarchical coordinates: Natural tiling representation

**Type Safety**:
- Element type tracked: `Float32`, `Float16`, `Int32`, etc.
- Compile-time optimizations

**Composability**:
- Tensors can be partitioned, sliced, and tiled
- Layouts can be composed and transformed
- Foundation for advanced patterns (next puzzles!)

<br>

### <b>Tips</b>
- CuTe tensors are lightweight views - no data copying
- Use `from_dlpack()` to create CuTe tensors from PyTorch
- Always verify tensor shapes before launching kernels
- Understanding layouts is key to memory optimization
