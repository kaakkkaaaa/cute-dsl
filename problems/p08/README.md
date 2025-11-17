## <b>Puzzle 8: Layout Composition</b>

### <b>Learning Objective</b>
Master layout composition - a powerful technique for transforming coordinate spaces and creating complex access patterns.

### <b>Part 8a: Understanding Base Layouts</b>

Learn how layouts map coordinates to memory offsets.

```py
def p08a():
    """Basic composition with actual values to visualize memory layout"""
    @cute.jit
    def compose_kernel():
        # Base layout A: (6, 2) with stride (8, 2)
        base = cute.make_layout((6, 2), stride=(8, 2))
        
        # TODO: Understand offset calculation
        # offset = i * 8 + j * 2
        # TODO: Fill fragment with offset-based values
        # TODO: Visualize the memory pattern
        pass
    
    return compose_kernel
```

**Memory Offset Formula**:
```
Layout A: shape=(6, 2), stride=(8, 2)
offset(i, j) = i × 8 + j × 2

Examples:
(0,0) → 0×8 + 0×2 = 0
(1,0) → 1×8 + 0×2 = 8
(0,1) → 0×8 + 1×2 = 2
(1,1) → 1×8 + 1×2 = 10
```

### <b>Part 8b: Coordinate Transformers</b>

Understand how a second layout can transform coordinates.

```py
def p08b():
    """Understanding Layout B (coordinate transformer)"""
    @cute.jit
    def compose_kernel():
        # Layout B: (4, 3) with stride (3, 1)
        tiler = cute.make_layout((4, 3), stride=(3, 1))
        
        # TODO: Map coordinates through B
        # TODO: See how B transforms input coordinates
        # TODO: Understand the transformation pattern
        pass
    
    return compose_kernel
```

**Coordinate Transformation**:
```
Layout B: shape=(4, 3), stride=(3, 1)
B(i, j) = i × 3 + j × 1

Examples:
B(0,0) = 0    B(0,1) = 1    B(0,2) = 2
B(1,0) = 3    B(1,1) = 4    B(1,2) = 5
B(2,0) = 6    B(2,1) = 7    B(2,2) = 8
B(3,0) = 9    B(3,1) = 10   B(3,2) = 11
```

<br>

### <b>Part 8c: Composition</b>

**Challenge**: Compose two layouts to create a new coordinate mapping.

```py
def p08c():
    """Composition R = A ∘ B"""
    @cute.jit
    def compose_kernel():
        base = cute.make_layout((6, 2), stride=(8, 2))
        tiler = cute.make_layout((4, 3), stride=(3, 1))
        
        # TODO: Compose layouts: R = A ∘ B
        # TODO: R(c) = A(B(c)) for coordinate c
        # TODO: Fill and visualize composed layout
        pass
    
    return compose_kernel
```

### <b>Key Concepts</b>

**Layout Composition**: <br>
`R = cute.composition(A, B)`: Creates R where R(c) = A(B(c)) <br>

**Composition Function**: <br>
```python
# For coordinate c:
# Step 1: Apply B to get intermediate coordinate
intermediate = B(c)

# Step 2: Apply A to get final offset
result = A(intermediate)

# Equivalent to:
R(c) = A(B(c))
```

**Use Cases**: <br>
- Tiling: Transform tile coordinates to global memory
- Swizzling: Create permuted access patterns
- Blocking: Map thread blocks to memory regions

### <b>Composition Example</b>

**Given**:
```python
A: (6, 2) stride (8, 2)  # Base layout
B: (4, 3) stride (3, 1)  # Coordinate transformer
R = A ∘ B                # Composition
```

**How R works**:
```
For coordinate (1, 2):

Step 1: Apply B
B(1, 2) = 1×3 + 2×1 = 5

Step 2: Interpret 5 as coordinate for A
5 → (row=5÷2=2, col=5%2=1)  # Integer division

Step 3: Apply A
A(2, 1) = 2×8 + 1×2 = 18

Therefore: R(1, 2) = 18
```

**Complete Mapping Table**:
```
Input   | B result | A coord | A offset | R result
--------|----------|---------|----------|----------
(0,0)   | 0        | (0,0)   | 0        | 0
(0,1)   | 1        | (0,1)   | 2        | 2
(0,2)   | 2        | (1,0)   | 8        | 8
(1,0)   | 3        | (1,1)   | 10       | 10
(1,1)   | 4        | (2,0)   | 16       | 16
(1,2)   | 5        | (2,1)   | 18       | 18
...
```

### <b>Hierarchical Result</b>

**Composed Layout Properties**:
```python
R = composition(A, B)
R.shape = ((2, 2), 3)  # Hierarchical!

# This means:
# - Outer dimension: 3 (from B's columns)
# - Inner dimension: (2, 2) (from B's rows split)
```

**Hierarchical Coordinate Access**:
```python
# Can access R with hierarchical coordinates
R[((i0, i1), j)] = A(B(flatten((i0, i1), j)))
```

<br>

### <b>Why Composition Matters</b>

**Tiled Memory Access**:
```python
# Global layout: large matrix
global_layout = make_layout((M, N))

# Tile layout: how to divide into tiles
tile_layout = make_layout((tile_m, tile_n))

# Composed: tile coordinates → global memory
tiled_access = composition(global_layout, tile_layout)
```

**Thread-to-Memory Mapping**:
```python
# Thread layout: which thread handles which element
thread_layout = make_layout((threads_per_block,))

# Memory layout: where data lives
memory_layout = make_layout(data_shape, stride)

# Composed: thread ID → memory location
access_pattern = composition(memory_layout, thread_layout)
```

**Swizzling for Bank Conflict Avoidance**:
```python
# Base layout: shared memory
smem_layout = make_layout((rows, cols))

# Swizzle pattern: XOR-based permutation
swizzle = make_swizzle_layout(...)

# Composed: bank conflict-free access
swizzled_smem = composition(smem_layout, swizzle)
```

### <b>Mathematical Perspective</b>

**Function Composition**:
```
R = A ∘ B  (read as "A after B")

For any coordinate c:
R(c) = (A ∘ B)(c) = A(B(c))
```

**Associativity**:
```python
# Composition is associative
(A ∘ B) ∘ C = A ∘ (B ∘ C)

# Can build complex transformations incrementally
```

**Identity**:
```python
# Identity layout doesn't change coordinates
I = make_layout((n,), stride=(1,))
A ∘ I = I ∘ A = A
```

### <b>Tips</b>
- Composition applies transformations right-to-left: A ∘ B means "B first, then A"
- Use `cute.composition(A, B)` to create composed layouts
- Composed layouts often have hierarchical shapes
- Visualize each step: coordinate → B → intermediate → A → final offset
- Essential for understanding tiling, swizzling, and advanced memory patterns
- Think of B as "coordinate generator" and A as "memory mapper"