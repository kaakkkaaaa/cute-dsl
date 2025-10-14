import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import torch
import numpy as np

# Puzzle 1: Thread Indexing - Understanding GPU Parallelism
# Goal : Understand the GPU's hierarchical execution model: grids, blocks, and threads
# Concepts:
# - Grid: Collection of thread blocks
# - Block: Group of threads that can cooperate
# - Thread: Individual execution unit
# - cute.arch.thread_idx() - thread index within block
# - cute.arch.block_idx() - block index within grid

BLOCKS_PER_GRID = (3, 1, 1);
THREADS_PER_BLOCK = (4, 1, 1);

def p02():
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

def main():
    print("\n" + "=" * 60);
    print("Testing Puzzle 02: Thread Indexing");
    print("="*60);

    cutlass.cuda.initialize_cuda_context();
    kernel = p02();
    kernel();
    print("\n✅ Puzzle 02 passed!\n");


if __name__ == "__main__":
    main();