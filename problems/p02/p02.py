import torch
import numpy as np
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

SIZE = 128;
THREADS_PER_BLOCK = (32, 1, 1);
BLOCKS_PER_GRID = (4, 1, 1);

def p02():
    """
    Add 10 to each element in parallel.
    Input: [0, 1, 2, ..., 127]
    Output: [10, 11, 12, ..., 137]
    """
    @cute.kernel
    def add_10_kernel(tensor: cute.Tensor):
        tidx, _, _ = cute.arch.thread_idx();
        bidx, _, _ = cute.arch.block_idx();
        bdim, _, _ = cute.arch.block_dim();

        global_idx = bdim * bidx + tidx;
        if global_idx < cute.size(tensor):
            tensor[global_idx] = tensor[global_idx] + 10.0;

    @cute.jit
    def run_kernel(data: cute.Tensor):
        add_10_kernel(data).launch(
            grid=BLOCKS_PER_GRID,
            block=THREADS_PER_BLOCK
        );

    return run_kernel

def main():
    print("\n" + "=" * 60);
    print("Testing Puzzle 02: Global Memory Access");
    print("=" * 60);

    data = torch.arange(128, dtype=torch.float32, device="cuda");
    expected = data.clone() + 10.0;

    print(f"Input (first 10): {data[:10]}");

    kernel = p03();
    kernel(from_dlpack(data));

    print(f"Output (first 10): {data[:10]}");

    assert torch.allclose(data, expected), "Add 10 failed!"
    print("\n✅ Puzzle 02 passed!");


if __name__ == "__main__":
    main();
