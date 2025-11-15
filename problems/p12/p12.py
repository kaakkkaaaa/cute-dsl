import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

# Puzzle 12: Tiling

TILE_SIZE = 16;
THREADS_PER_BLOCK = 256;
NUM_ROWS, NUM_COLS = 64, 64;

def p12a():
    """Process in tiles."""

    @cute.kernel
    def tile_kernel(gmem: cute.Tensor):
        # Thread indices within the block
        thread_x, thread_y, _ = cute.arch.thread_idx();
        # Block indices within the grid
        block_x, block_y, _ = cute.arch.block_idx();
      
        # Allocate shared memory for this tile
        smem = cutlass.utils.SmemAllocator();
        smem_layout = cute.make_layout((TILE_SIZE, TILE_SIZE));
        shared = smem.allocate_tensor(cutlass.Float32, smem_layout);

        # Each thread handles one element in the tile
        # Thread organization: 16x16 threads per block
        if thread_x < TILE_SIZE and thread_y < TILE_SIZE:
            # Calculate global memory coordinates
            global_row = block_x * TILE_SIZE + thread_x;
            global_col = block_y * TILE_SIZE + thread_y;
        
            # Load from global memory to shared memory with computation
            shared[thread_x, thread_y] = gmem[global_row, global_col] + 1.0;
        
        # Synchronize all threads in the block
        cute.arch.sync_threads();

        # Write back from shared memory to global memory
        if thread_x < TILE_SIZE and thread_y < TILE_SIZE:
            global_row = block_x * TILE_SIZE + thread_x;
            global_col = block_y * TILE_SIZE + thread_y;
        
            gmem[global_row, global_col] = shared[thread_x, thread_y];

    @cute.jit
    def run_kernel(data: cute.Tensor):
        grid_x = NUM_ROWS // TILE_SIZE;
        grid_y = NUM_COLS // TILE_SIZE;
    
        block_x = TILE_SIZE;
        block_y = TILE_SIZE;
        threads_per_block = block_x * block_y;

        # Shared memory requirement (in bytes)
        smem_bytes = TILE_SIZE * TILE_SIZE * 4;
    
        tile_kernel(data).launch(
            grid=(grid_x, grid_y, 1),
            block=(block_x, block_y, 1),
            smem=smem_bytes
        );
    
    return run_kernel;

def main():
    print("\n" + "="*60);
    print("Puzzle 12: Tiling");
    print("=" * 60);
    
    cutlass.cuda.initialize_cuda_context();

    data = torch.arange(64*64, dtype=torch.float32, device="cuda").reshape(64, 64);
    expected = data.clone() + 1.0;
    kernel = p12a();
    kernel(from_dlpack(data));

    print("\nResult after kernel (first 8x8 corner):");
    print(data[:8, :8]);

    print("\nExpected result (first 8x8 corner):");
    print(expected[:8, :8]);

    print("\n🔍 Verification:");
    print(f"   Max difference: {(data - expected).abs().max().item()}");
    print(f"   All close: {torch.allclose(data, expected)}");

    assert torch.allclose(data, expected);
    print("✅ Passed!");

if __name__ == "__main__":
    main();