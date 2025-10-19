import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import torch
import numpy as np

# ANCHOR: p00
def p00():
    """
    Write a kernel that prints "Hello from thread X" where X is the thread index.
    Launch it with 8 threads.
    """
    @cute.kernel
    def hello_kernel():
        # TODO: Get thread index
        # TODO: Print message with thread index
        tidx, _, _ = cute.arch.thread_idx();
        cute.printf("Hello from thread {}", tidx);

    @cute.jit
    def run_hello():
        # TODO: Launch kernel with 8 threads
        hello_kernel().launch(
            grid=(1, 1, 1),     # 1 thread block
            block=(8, 1, 1)     # 8 thread
        );

    return run_hello
# ANCHOR_END: p00

def main():
    print("="*60);
    print("Puzzle 0: Hello World from GPU Threads");
    print("="*60);
    print("\nLaunching kernel with 8 threads...\n");

    cutlass.cuda.initialize_cuda_context();
    kernel = p00();
    kernel();
    torch.cuda.synchronize();

    print("\n" + "="*60);
    print("✓ Puzzle 00 completed!");
    print("="*60);

if __name__ == "__main__":
    main();