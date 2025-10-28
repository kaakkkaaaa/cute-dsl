import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

TILE_SIZE = 256;
THREADS_PER_BLOCK = (128, 1, 1);
BLOCKS_PER_GRID = (1, 1, 1);

def p10a():
    """Copy using block_dim to calculate work per thread"""

    @cute.kernel
    def copy_kernel(gmem_in: cute.Tensor, gmem_out: cute.Tensor):
        tidx, _, _ = cute.arch.thread_idx();
        bdimx, _, _ = cute.arch.block_dim();

        smem = cutlass.utils.SmemAllocator();
        smem_layout = cute.make_layout((TILE_SIZE,));
        shared = smem.allocate_tensor(cutlass.Float32, smem_layout);

        # Calculate how many elements each thread should handle
        elements_per_thread = (TILE_SIZE + bdimx - 1) // bdimx;

        for i in range(elements_per_thread):
            idx = tidx + i * bdimx;
            if idx < TILE_SIZE:
                shared[idx] = gmem_in[idx];

        cute.arch.sync_threads();

        if tidx == 0:
            cute.printf("Copied %d elements using %d threads (%d elem/thread)\n", 
                       TILE_SIZE, bdimx, elements_per_thread);

        # Write back to global memory
        for i in range(elements_per_thread):
            idx = tidx + bdimx * i;
            if idx < TILE_SIZE:
                gmem_out[idx] = shared[idx];
    
    @cute.jit
    def run_kernel(data_in: cute.Tensor, data_out: cute.Tensor):
        copy_kernel(data_in, data_out).launch(
            grid=BLOCKS_PER_GRID,
            block=THREADS_PER_BLOCK,
            smem=256*4
        );

    return run_kernel


def p10b():
    """Use enough blocks so each thread handles 1 element"""
    
    TOTAL_SIZE = 256;
    THREADS_PER_BLOCK = 128;

    @cute.kernel
    def copy_kernel(gmem_in: cute.Tensor, gmem_out: cute.Tensor, total_size: int):
        tidx, _, _ = cute.arch.thread_idx();
        bdimx, _, _ = cute.arch.block_dim();
        bidx, _, _ = cute.arch.block_idx();

        # Each block processes a tile
        smem = cutlass.utils.SmemAllocator();
        smem_layout = cute.make_layout((THREADS_PER_BLOCK, ));
        shared = smem.allocate_tensor(cutlass.Float32, smem_layout);

        # Calculate global index - each thread handles exactly 1 element
        global_idx = bidx * THREADS_PER_BLOCK + tidx;

        # Simple 1:1 mapping: thread i loads element i
        if tidx < THREADS_PER_BLOCK and global_idx < total_size:
            shared[tidx] = gmem_in[global_idx];

        cute.arch.sync_threads();

        if tidx == 0:
            cute.printf("Block %d: thread handles 1 element each\n", bidx);
    
        # Write back to global memory
        if tidx < THREADS_PER_BLOCK and global_idx < total_size:
            gmem_out[global_idx] = shared[tidx];

    @cute.jit
    def run_kernel(data_in: cute.Tensor, data_out: cute.Tensor):

        # Calculate number of blocks needed
        NUM_BLOCKS = (TOTAL_SIZE + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK;

        print(f"Grid-based approach:");
        print(f"  Total size: {TOTAL_SIZE}");
        print(f"  Threads per block: {THREADS_PER_BLOCK}");
        print(f"  Number of blocks: {NUM_BLOCKS}");
        print(f"  Elements per thread: 1");
        print();

        copy_kernel(data_in, data_out, TOTAL_SIZE).launch(
            grid=(NUM_BLOCKS, 1, 1),
            block=(THREADS_PER_BLOCK, 1, 1),
            smem=128*4
        );

    return run_kernel


def p10c():
    """Copy a 2D tile to shared memory with proper indexing."""

    @cute.kernel
    def copy_2d_kernel(gmem_in: cute.Tensor, gmem_out: cute.Tensor):
        tidx, tidy, _ = cute.arch.thread_idx();
        TILE_M, TILE_N = 16, 32;    # 16x32 tile

        smem = cutlass.utils.SmemAllocator();
        smem_layout = cute.make_layout((TILE_M, TILE_N));
        shared = smem.allocate_tensor(cutlass.Float32, smem_layout);

        # Each thread copies multiple elements in a coalesced pattern
        for m in range(TILE_M // 8):    # 8 threads in y-dimension
            for n in range(TILE_N // 16):   # 16 threads in x-dimension
                row = tidy + m * 8;
                col = tidx + n * 16;
                if row < TILE_M and col < TILE_N:
                    global_idx = row * TILE_N + col;
                    shared[row, col] = gmem_in[global_idx];

        cute.arch.sync_threads();
    
        # Write back to global memory for verification
        for m in range(TILE_M // 8):
            for n in range(TILE_N // 16):
                row = tidy + m * 8;
                col = tidx + n * 16;
                if row < TILE_M and col < TILE_N:
                    global_idx = row * TILE_N + col;
                    gmem_out[global_idx] = shared[row, col];

    @cute.jit
    def run_kernel(data_in: cute.Tensor, data_out: cute.Tensor):
        copy_2d_kernel(data_in, data_out).launch(
            grid=(1, 1, 1),
            block=(16, 8, 1),
            smem=16*32*4
        )
    
    return run_kernel

def main():
    print("\n" + "="*80);
    print("Puzzle 10: Copy Operations");
    print("="*80 + "\n");

    # Test p10a (Loop-based)
    print("--- Test p10a: Loop-Based Approach ---");
    inp_a = torch.arange(0, 256, dtype=torch.float32, device="cuda");
    out_a = torch.zeros(256, dtype=torch.float32, device="cuda");

    cutlass.cuda.initialize_cuda_context();
    kernel_a = p10a();
    kernel_a(from_dlpack(inp_a), from_dlpack(out_a));
    
    # Verify
    expected_a = torch.arange(0, 256, dtype=torch.float32, device="cuda");
    assert torch.allclose(out_a, expected_a)
    print("Output (first 8): ", out_a[:8].cpu().numpy());
    print("Output (last 8): ", out_a[-8:].cpu().numpy());
    print("Expected (first 8):", expected_a[:8].cpu().numpy());
    print("Expected (last 8): ", expected_a[-8:].cpu().numpy());

    # TEST P11b (Grid-based)
    print("\n--- Test p10b: Grid-Based Approach ---");
    inp_b = torch.arange(0, 256, dtype=torch.float32, device="cuda");
    out_b = torch.arange(256, dtype=torch.float32, device="cuda");

    kernel_b = p10b();
    kernel_b(from_dlpack(inp_b), from_dlpack(out_b));

    # Verify
    expected_b = torch.arange(0, 256, dtype=torch.float32, device="cuda");
    assert torch.allclose(out_b, expected_b), "Output doesn't match expected!";
    print("Output (first 8): ", out_b[:8].cpu().numpy());
    print("Output (last 8):  ", out_b[-8:].cpu().numpy());
    print("Expected (first 8):", expected_b[:8].cpu().numpy());
    print("Expected (last 8): ", expected_b[-8:].cpu().numpy());
    print("\n✅ p10b Passed!\n");
    

    print("\n--- Test p10c: 2D Tile Copy ---");
    TILE_M, TILE_N = 16, 32;
    TOTAL_SIZE = TILE_M * TILE_N;

    print(f"Tile dimensions: {TILE_M}x{TILE_N} = {TOTAL_SIZE} elements");
    print(f"Thread block: (16, 8, 1) = {16*8} threads");
    print();

    inp_c = torch.arange(0, TOTAL_SIZE, dtype=torch.float32, device="cuda");
    out_c = torch.zeros(TOTAL_SIZE, dtype=torch.float32, device="cuda");

    # Run kernel
    kernel = p10c();
    kernel(from_dlpack(inp_c), from_dlpack(out_c));

    # Verify full output
    expected = torch.arange(0, TOTAL_SIZE, dtype=torch.float32, device="cuda");
    assert torch.allclose(out_c, expected), "Output doesn't match expected!";

    # Reshape for 2D verification
    inp_2d = inp_c.reshape(TILE_M, TILE_N);
    out_2d = out_c.reshape(TILE_M, TILE_N);

    print(inp_2d);
    print();
    print(out_2d);

    print("\n" + "="*80);
    print("✅ Puzzle 10c Passed!");
    print("="*80);

if __name__ == "__main__":
    main();