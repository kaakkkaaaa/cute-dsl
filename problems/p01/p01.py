import torch
import numpy as np
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

# Puzzle 1: Thread Indexing
# Goal: Each thread writes its thread index to output
# Concepts: thread_idx(), basic kernel structure

def p01():
    """
    Fill in the kernel so each thread writes its x-coordinate to output.

    Input: None
    Output: [0, 1, 2, ..., 31] (32 threads)
    """
    @cute.kernel
    def thread_id_kernel(output: cute.Tensor):
        tidx, _, _ = cute.arch.thread_idx();
        # TODO: Write tidx to output[tidx]
        output[tidx] = tidx;

    @cute.jit
    def run_kernel(output: cute.Tensor):
        thread_id_kernel(output).launch(
            grid=(1, 1, 1),
            block=(32, 1, 1)
        );

    return run_kernel

def main():
    output = torch.zeros(32, dtype=torch.int32, device="cuda");
    cutlass.cuda.initialize_cuda_context();

    kernel = p01();
    kernel(from_dlpack(output));
    print(f"\nOutput:\n{output}");

    expected = torch.arange(32, dtype=torch.int32, device="cuda");
    print(f"\nExpected:\n{expected}");

    print("\n" + "="*60);
    assert torch.allclose(output, expected), f"Expected {expected}, got {output}";
    print("✅ Puzzle 01 passed!");
    print("="*60);

if __name__ == "__main__":
    main();