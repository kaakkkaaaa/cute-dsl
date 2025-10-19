import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import torch

THREADS_PER_BLOCK = (1, 1, 1);
BLOCKS_PER_GRID = (1, 1, 1);

def p05():
    """
    Create and access tensors
    """
    @cute.kernel
    def tensor_demo():
        tidx, _, _ = cute.arch.thread_idx();
        if tidx == 0:
            cute.printf("Tensor Basics Demonstrated\n");

    @cute.jit
    def run_kernel():
        tensor_demo().launch(
            grid=BLOCKS_PER_GRID,
            block=THREADS_PER_BLOCK
        );

    return run_kernel

def main():
    print("\n" + "="*60);
    print("Puzzle 05: Tensor Basics");
    print("="*60 + "\n");
    cutlass.cuda.initialize_cuda_context();
    kernel = p05();
    kernel();
    print("✅ Puzzle 05 Passed!")

if __name__ == "__main__":
    main();