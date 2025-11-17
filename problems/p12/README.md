## <b>Puzzle 12: Tiling</b>

### <b>Learning Objective</b>
Master tiling - a fundamental technique for processing large matrices by dividing them into smaller, cache-friendly blocks.

### <b>Challenge</b>

Process a 64×64 matrix by dividing it into 16×16 tiles, with each tile processed by one thread block.

```py
def p12a():
    """Process in tiles"""
    
    @cute.kernel
    def tile_kernel(gmem: cute.Tensor):
        thread_x, thread_y, _ = cute.arch.thread_idx()
        block_x, block_y, _ = cute.arch.block_idx()
        
        # TODO: Allocate 16×16 shared memory tile
        # TODO: Map thread to global position
        # TODO: Load tile from global memory
        # TODO: Process data (add 1)
        # TODO: Synchronize
        # TODO: Write tile back to global memory
        pass
    
    @cute.jit
    def run_kernel(data: cute.Tensor):
        # TODO: Calculate grid dimensions
        # TODO: Launch with 2D grid and block
        pass
    
    return run_kernel
```

**Configuration**:
```
Matrix: 64×64 = 4096 elements
Tile size: 16×16 = 256 elements
Grid: 4×4 = 16 blocks
Threads per block: 16×16 = 256 threads

Each block processes one 16×16 tile
Each thread processes one element within its tile
```

### <b>Key Concepts</b>

**Tiling**: <br>
Dividing a large problem into smaller sub-problems (tiles) that fit in fast memory.

**2D Grid Launch**: <br>
```python
grid = (grid_x, grid_y, 1)
block = (block_x, block_y, 1)
```

**Global Coordinate Mapping**: <br>
```python
global_row = block_idx_x * tile_size + thread_idx_x
global_col = block_idx_y * tile_size + thread_idx_y
```

**Shared Memory Per Block**: <br>
Each block gets its own shared memory tile <br>
`smem_bytes = tile_size × tile_size × sizeof(element)` <br>

### <b>Tiling Visualization</b>

**Matrix Division**:
```
64×64 Matrix divided into 16 tiles (4×4 grid):

┌────────┬────────┬────────┬────────┐
│ Tile   │ Tile   │ Tile   │ Tile   │
│ (0,0)  │ (0,1)  │ (0,2)  │ (0,3)  │
│ 16×16  │ 16×16  │ 16×16  │ 16×16  │
├────────┼────────┼────────┼────────┤
│ Tile   │ Tile   │ Tile   │ Tile   │
│ (1,0)  │ (1,1)  │ (1,2)  │ (1,3)  │
│ 16×16  │ 16×16  │ 16×16  │ 16×16  │
├────────┼────────┼────────┼────────┤
│ Tile   │ Tile   │ Tile   │ Tile   │
│ (2,0)  │ (2,1)  │ (2,2)  │ (2,3)  │
│ 16×16  │ 16×16  │ 16×16  │ 16×16  │
├────────┼────────┼────────┼────────┤
│ Tile   │ Tile   │ Tile   │ Tile   │
│ (3,0)  │ (3,1)  │ (3,2)  │ (3,3)  │
│ 16×16  │ 16×16  │ 16×16  │ 16×16  │
└────────┴────────┴────────┴────────┘

Each tile processed independently by one block
```

**Thread-to-Element Mapping**:
```
Within Tile (0,0):
Block (0,0) with 16×16 threads

Thread (0,0) → Element [0,0]
Thread (0,1) → Element [0,1]
...
Thread (15,15) → Element [15,15]

Within Tile (2,1):
Block (2,1) with 16×16 threads

Thread (0,0) → Element [32,16]
Thread (0,1) → Element [32,17]
...
Thread (15,15) → Element [47,31]
```

### <b>Coordinate Calculation</b>

**Local to Global Mapping**:
```python
# Local: position within