import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import torch
import numpy as np

# Puzzle 1: Block Indexing
# Goal: Use both thread and block indices
# Concepts: block_idx(), block_dim(), global thread ID

N = 32;
threads_per_block = 8;
a = torch.zeros(N, dtype=torch.int32, device="cuda");

def p01():
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

def main():
    cutlass.cuda.initialize_cuda_context();

    kernel = p01();
    kernel(from_dlpack(a));
    print(f"\nOutput:\n{a}");

    expected = torch.arange(32, dtype=torch.int32, device="cuda");
    print(f"\nExpected:\n{expected}");

    print("\n" + "="*60);
    assert torch.allclose(a, expected), f"Expected {expected}, got {output}";
    print("✅ Puzzle 01 passed!");
    print("="*60);

if __name__ == "__main__":
    main();