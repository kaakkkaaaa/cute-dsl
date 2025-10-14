import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import torch
import numpy as np

# Puzzle 1: 2D Indexing
# Goal: Access 2D tensor using row-major ordering
# Concepts: Multi-dimensional indexing, shape/stride

ROWS = 4;
COLS = 8;
THREAD_PER_BLOCK = (COLS, ROWS, 1);
BLOCKS_PER_GRID = (1, 1, 1);

def p02():
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

def main():
    input_data = torch.arange(32, dtype=torch.float32, device="cuda").reshape(4, 8);
    output = torch.zeros(4, 8, dtype=torch.float32, device="cuda");

    print("Input:", input_data);
    
    cutlass.cuda.initialize_cuda_context();
    kernel = p02();
    kernel(from_dlpack(output), from_dlpack(input_data))

    print("Output:", output);

    assert torch.allclose(output, input_data), "2D copy failed!";
    print("✅ Puzzle 02 passed!");

if __name__ == "__main__":
    main();