import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import torch
import numpy as np
import time

# ANCHOR: p14a
def p14a():
    """
    Naive matrix addtion: one thread per element, no tilling.
    Used for comparison to show why tiling is matters.
    """

    @cute.kernel
    def naive_add_kernel(
        tensorA: cute.Tensor,
        tensorB: cute.Tensor,
        tensorC: cute.Tensor
    ):
        # Each thread computes one element
        tidx, tidy, _ = cute.arch.thread_idx();
        bidx, bidy, _ = cute.arch.block_idx();
        bdimx, bdimy, _ = cute.arch.block_dim();

        # Global indices
        row = bidx * bdimx + tidx;
        col = bidy * bdimy + tidy;

        # Bounds check
        M = cute.size(tensorA.shape[0]);
        N = cute.size(tensorB.shape[1]);

        if row < M and col < N:
            # Direct global memory access
            tensorC[(row, col)] = tensorA[(row, col)] + tensorB[(row, col)];

    @cute.jit
    def run_kernel(
        mA: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor
    ):
        M = cute.size(mA.shape[0]);
        N = cute.size(mA.shape[1]);
        
        # Launch with 16x16 thread blocks
        BLOCK_SIZE = 16;
        grid_m = (M + BLOCK_SIZE - 1) // BLOCK_SIZE;
        grid_n = (N + BLOCK_SIZE - 1) // BLOCK_SIZE;

        naive_add_kernel(
            mA, mB, mC
        ).launch(
            grid=(grid_m, grid_n, 1),
            block=(BLOCK_SIZE, BLOCK_SIZE, 1)
        );

    return run_kernel
# ANCHOR_END: p14a

def p14b():
    """
    Tiled matrix addition using zipped_divide.
    Each thread block processes one tile efficiently.
    """
    TILE_M = 16;
    TILE_N = 16;

    @cute.kernel
    def tiled_add_kernel(
        tensorA: cute.Tensor,
        tensorB: cute.Tensor,
        tensorC: cute.Tensor
    ):
        # Get thread and block coordinates
        tidx, tidy, _ = cute.arch.thread_idx();
        bidx, bidy, _ = cute.arch.block_idx();

        # Partition by tiles
        gA_tiled = cute.zipped_divide(tensorA, (TILE_M, TILE_N));
        gB_tiled = cute.zipped_divide(tensorB, (TILE_M, TILE_N));
        gC_tiled = cute.zipped_divide(tensorC, (TILE_M, TILE_N));

        # Create hierarchical coordinate
        # ((local position), (tile position))
        coord = ((tidy, tidx), (bidy, bidx));

        # Use the coordinate directly - no need to extract tiles!
        gC_tiled[coord] = gA_tiled[coord] + gB_tiled[coord];

    @cute.jit
    def run_kernel(
        mA: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor
    ):
        M = cute.size(mA.shape[0]);
        N = cute.size(mA.shape[1]);

        # Calculate number of tiles needed
        num_tiles_M = M // TILE_M;
        num_tiles_N = N // TILE_N;

        # Launch: one thread block per tile
        # Block dimensions match tile dimensions
        tiled_add_kernel(mA, mB, mC).launch(
            grid=(num_tiles_M, num_tiles_N, 1),
            block=(TILE_N, TILE_M, 1)
        );

    return run_kernel

# ANCHOR: p14c
def p14c():
    """
    Further optimized version with explicit bounds checking
    and support for non-divisible matrix sizes.
    """

    TILE_M = 16;
    TILE_N = 16;

    @cute.kernel
    def optimized_tiled_kernel(
        tensorA: cute.Tensor,
        tensorB: cute.Tensor,
        tensorC: cute.Tensor
    ):
        tidx, tidy, _ = cute.arch.thread_idx();
        bidx, bidy, _ = cute.arch.block_idx();

        # Calculate global position for bounds checking
        global_row = bidx * TILE_M + tidy;
        global_col = bidx * TILE_N + tidx;

        M = cute.size(tensorA.shape[0]);
        N = cute.size(tensorA.shape[1]);

        # Bounds check for non-divisible sizes
        if global_row < M and global_col < N:
            # Use zipped_divide for clean indexing
            gA_tiled = cute.zipped_divide(tensorA, (TILE_M, TILE_N));
            gB_tiled = cute.zipped_divide(tensorB, (TILE_M, TILE_N));
            gC_tiled = cute.zipped_divide(tensorC, (TILE_M, TILE_N));

            # Use hierarchical coordinates directly
            coord = ((tidy, tidx), (bidy, bidx));
            gC_tiled[coord] = gA_tiled[coord] + gB_tiled[coord];

    @cute.jit
    def run_kernel(
        mA: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor
    ):
        M = cute.size(mA.shape[0]);
        N = cute.size(mA.shape[1]);

        grid_m = (M + TILE_M - 1) // TILE_M;
        grid_n = (N + TILE_N - 1) // TILE_N;

        optimized_tiled_kernel(mA, mB, mC).launch(
            grid=(grid_m, grid_n, 1),
            block=(TILE_N, TILE_M, 1)
        );

    return run_kernel
# ANCHOR_END: p14c

def p14d():
    """
    Puzzle 15d: Shared Memory Optimization
    Demonstrates how to use shared memory with tiled matrix addition.
    """

    TILE_M = 16;
    TILE_N = 16;

    @cute.kernel
    def smem_tiled_add_kernel(
        tensorA: cute.Tensor,
        tensorB: cute.Tensor,
        tensorC: cute.Tensor
    ):
        # Get thread and block IDs
        tidx, tidy, _ = cute.arch.thread_idx();
        bidx, bidy, _ = cute.arch.block_idx();

        # Allocate shared memory
        # Shared memory is shared by all threads in a block
        smem = cutlass.utils.SmemAllocator();
        smem_layout = cute.make_layout((TILE_M, TILE_N));
        tile_A_smem = smem.allocate_tensor(cutlass.Float32, smem_layout);
        tile_B_smem = smem.allocate_tensor(cutlass.Float32, smem_layout);

        # Partition global tensors by tiles
        gA_tiled = cute.zipped_divide(tensorA, (TILE_M, TILE_N));
        gB_tiled = cute.zipped_divide(tensorB, (TILE_M, TILE_N));
        gC_tiled = cute.zipped_divide(tensorC, (TILE_M, TILE_N));

        # Hierarchical coordinate
        coord = ((tidy, tidx), (bidy, bidx));

        # Cooperatively load tile to shared memory
        # Each thread loads one element from global to shared
        tile_A_smem[(tidy, tidx)] = gA_tiled[coord];
        tile_B_smem[(tidy, tidx)] = gB_tiled[coord];

        # Synchronize threads
        # Wait for all threads to finish loading before computing
        cute.arch.sync_threads();

        # Compute using shared memory
        # Now access is from fast shared memory, not slow global memory
        result = tile_A_smem[(tidy, tidx)] + tile_B_smem[(tidy, tidx)];

        # Write result back to global memory
        gC_tiled[coord] = result;

    @cute.jit
    def run_kernel(
        mA: cute.Tensor, 
        mB: cute.Tensor, 
        mC: cute.Tensor
    ):
        M = cute.size(mA.shape[0]);
        N = cute.size(mA.shape[1]);

        num_tiles_m = M // TILE_M;
        num_tiles_n = N // TILE_N;

        smem_tiled_add_kernel(mA, mB, mC).launch(
            grid=(num_tiles_m, num_tiles_n, 1),
            block=(TILE_M, TILE_N, 1)
        );

    return run_kernel


def main():
    """
    Test and compare naive vs tiled implementations.
    """
    print("\n" + "="*80);
    print("Puzzle 15: Zipped Divide - Practical Use Case");
    print("="*80 + "\n");

    # Initialize CUDA context
    cutlass.cuda.initialize_cuda_context();

    # Test dimensions
    M, N = 1024, 1024;
    TILE_M, TILE_N = 16, 16;

    print(f"\nConfig:");
    print(f"  Matrix size: {M}×{N}");
    print(f"  Tile size: {TILE_M}×{TILE_N}");
    print(f"  Number of tiles: {M//TILE_M}×{N//TILE_N} = {(M//TILE_M)*(N//TILE_N)} tiles");
    print(f"  Threads per block: {TILE_M}×{TILE_N} = {TILE_M*TILE_N} threads");

    # Create test data
    print("\nCreating Test Tensors...");
    A_torch = torch.randn(M, N, dtype=torch.float32, device="cuda");
    B_torch = torch.randn(M, N, dtype=torch.float32, device="cuda");
    C_expected = A_torch + B_torch;

    # Create test data
    A_cute = from_dlpack(A_torch);
    B_cute = from_dlpack(B_torch);

    # ==== Test Naive Implementation ====
    print("\n" + "-"*80);
    print("Testing Naive Matrix Addition Kernel...");

    C_naive_torch = torch.zeros(M, N, dtype=torch.float32, device="cuda");
    C_naive = from_dlpack(C_naive_torch);

    naive_kernel = p14a();
    naive_kernel(A_cute, B_cute, C_naive);
    torch.cuda.synchronize();

    naive_correct = torch.allclose(C_naive_torch, C_expected, rtol=1e-4, atol=1e-4);
    print(f"Correctness: {'✅ PASS' if naive_correct else '✗ FAIL'}");

    if naive_correct:
        max_diff = torch.max(torch.abs(C_naive_torch - C_expected)).item();
        print(f"Max difference: {max_diff:e}");

    # ==== Test Tiled Implementation ====
    print("\n" + "-"*80);
    print("Testing Tiled Matrix Addition Kernel (Zipped Divide)...");
    print("-"*70);

    C_tiled_torch = torch.zeros(M, N, dtype=torch.float32, device="cuda");
    C_tiled = from_dlpack(C_tiled_torch);

    tiled_kernel = p14b();
    tiled_kernel(A_cute, B_cute, C_tiled);
    torch.cuda.synchronize();

    tiled_correct = torch.allclose(C_tiled_torch, C_expected, rtol=1e-4, atol=1e-4);
    print(f"Correctness: {'✅ PASS' if tiled_correct else '✗ FAIL'}");

    if tiled_correct:
        max_diff = torch.max(torch.abs(C_tiled_torch - C_expected)).item();
        print(f"Max difference: {max_diff:e}");

    # ===== Test Basic Shared Memory =====
    print("\n" + "-"*70);
    print("Basic Shared Memory Implementation");
    print("-"*70);

    C_mem = torch.zeros(M, N, dtype=torch.float32, device="cuda");
    C_mem_cute = from_dlpack(C_mem);

    smem_kernel = p14d();
    smem_kernel(A_cute, B_cute, C_mem_cute);
    torch.cuda.synchronize();

    correct = torch.allclose(C_mem, C_expected, rtol=1e-4, atol=1e-4);
    print(f"Correctness: {'✅ PASS' if correct else '✗ FAIL'}");
    if correct:
        max_diff = torch.max(torch.abs(C_mem - C_expected)).item();
        print(f"Max difference: {max_diff:.2e}");

if __name__ == "__main__":
    main();