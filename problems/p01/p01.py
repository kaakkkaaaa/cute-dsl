import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import torch
import numpy as np

# Puzzle 1a: Thread Indexing - Understanding GPU Parallelism
# Goal : Understand the GPU's hierarchical execution model: grids, blocks, and threads
# Concepts:
# - Grid: Collection of thread blocks
# - Block: Group of threads that can cooperate
# - Thread: Individual execution unit
# - cute.arch.thread_idx() - thread index within block
# - cute.arch.block_idx() - block index within grid

BLOCKS_PER_GRID = (3, 1, 1);
THREADS_PER_BLOCK = (4, 1, 1);

# ANCHOR: p01a
def p01a():
    @cute.kernel
    def thread_id_kernel():
        tidx, _, _ = cute.arch.thread_idx();
        bidx, _, _ = cute.arch.block_idx();
        bdim, _, _ = cute.arch.block_dim();
        # TODO: Copy input[global_id] to output[global_id]
        global_idx = bidx * bdim + tidx;
        cute.printf("Block %d, Thread %d ⟶ Global ID: %d\n", bidx, tidx, global_idx);

    @cute.jit
    def run_thread_id():
        thread_id_kernel().launch(
            grid=BLOCKS_PER_GRID,
            block=THREADS_PER_BLOCK
        );

    return run_thread_id
# ANCHOR_END: p01a

# Puzzle 1b: Block Indexing
# Goal: Use both thread and block indices
# Concepts: block_idx(), block_dim(), global thread ID

N = 32;
threads_per_block = 8;
a = torch.zeros(N, dtype=torch.int32, device="cuda");

# ANCHOR: p01b
def p01b():
    """
    Create a kernel that fills a 1D array with thread indices.
    Each thread should write its global thread index to the array.
    """
    @cute.kernel
    def fill_indices(arr: cute.Tensor):
        # TODO: Calculate global thread index
        # TODO: Write to array
        tidx, _, _ = cute.arch.thread_idx();
        bidx, _, _ = cute.arch.block_idx();
        bdim, _, _ = cute.arch.block_dim();
    
        global_idx = bidx * bdim + tidx;
        arr[global_idx] = global_idx;

    @cute.jit
    def run_kernel(arr: cute.Tensor):
        # TODO: Launch with correct grid/block dims
        num_blocks = cute.size(arr) // threads_per_block;

        fill_indices(arr).launch(
            grid=(num_blocks, 1, 1),
            block=(threads_per_block, 1, 1)
        );

    return run_kernel
# ANCHOR_END: p01b

# Puzzle 1c: 2D Indexing
# Goal: Access 2D tensor using row-major ordering
# Concepts: Multi-dimensional indexing, shape/stride

# ANCHOR: p01c
ROWS = 4;
COLS = 8;
THREAD_PER_BLOCK = (COLS, ROWS, 1);
BLOCKS_PER_GRID = (1, 1, 1);

def p01c():
    """
    Copy 2D tensor (4x8) from input to output using proper 2D indexing.

    Input: [[1, 2, 3, ...], [9, 10, 11, ...], ...]
    Output: Same as input
    """
    @cute.kernel
    def copy_2d_kernel(output: cute.Tensor, input: cute.Tensor):
        tidx, tidy, _ = cute.arch.thread_idx();

        # Assume 32 threads total, arrange as 4×8
        # TODO: Compute row and col from thread index
        # TODO: Copy input[row, col] to output[row, col]
        output[tidy, tidx] = input[tidy, tidx];
    
    @cute.jit
    def run_kernel(output: cute.Tensor, input: cute.Tensor):
        copy_2d_kernel(output, input).launch(
            grid=BLOCKS_PER_GRID,
            block=THREAD_PER_BLOCK
        );

    return run_kernel
# ANCHOR_END: p01c

def main():
    print("\n" + "=" * 60);
    print("Testing Puzzle 01a: Thread Indexing");
    print("="*60);

    cutlass.cuda.initialize_cuda_context();
    kernel = p01a();
    kernel();
    print("\n✅ Puzzle 01a passed!\n");

    print("\n" + "=" * 60);
    print("Testing Puzzle 01b: Block Indexing");
    print("="*60);

    a = torch.zeros(N, dtype=torch.int32, device="cuda");
    kernel = p01b();
    kernel(from_dlpack(a));
    print(f"\nOutput:\n{a}");

    expected = torch.arange(32, dtype=torch.int32, device="cuda");
    print(f"\nExpected:\n{expected}");
    print("\n" + "="*60);
    assert torch.allclose(a, expected), f"Expected {expected}, got {output}";
    print("\n✅ Puzzle 01b passed!");
    print("="*60);


    print("\n" + "=" * 60);
    print("Testing Puzzle 01c: 2D Indexing");
    print("="*60);

    input_data = torch.arange(32, dtype=torch.float32, device="cuda").reshape(4, 8);
    output = torch.zeros(4, 8, dtype=torch.float32, device="cuda");

    kernel = p01c();
    kernel(from_dlpack(a));
    print(f"\nOutput:\n{a}");

    expected = torch.arange(32, dtype=torch.int32, device="cuda");
    print(f"\nExpected:\n{expected}");

    print("\n" + "="*60);
    assert torch.allclose(a, expected), f"Expected {expected}, got {output}";
    print("\n✅ Puzzle 01c passed!");
    print("="*60);

if __name__ == "__main__":
    main();
