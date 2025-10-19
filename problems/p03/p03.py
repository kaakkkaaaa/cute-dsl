import torch
import numpy as np
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

BLOCK_SIZE = 32;
THREADS_PER_BLOCK = (BLOCK_SIZE, 1, 1);
BLOCKS_PER_GRID = (4, 1, 1);

def p03():
    """
    Reverse array within each block using shared memory.
    Input: [0, 1, 2, ..., 127]
    Output: [31, 30, ..., 0, 63, 62, ..., 32, ...] (each block reversed)
    """
    @cute.kernel
    def reverse_block_kernel(input_t: cute.Tensor, output_t: cute.Tensor):
        tidx, _, _ = cute.arch.thread_idx();
        bidx, _, _ = cute.arch.block_idx();
        
        block_size = 32;
        
        # Create shared memory allocator and allocate tensor
        smem = cutlass.utils.SmemAllocator();
        shared = smem.allocate_tensor(input_t.element_type, block_size);
        
        global_idx = bidx * block_size + tidx;
        shared[tidx] = input_t[global_idx];
        
        # Synchronize threads - use cute.arch.sync_threads()
        cute.arch.sync_threads();
        
        reversed_idx = block_size - 1 - tidx;
        output_t[global_idx] = shared[reversed_idx];
    
    @cute.jit
    def run_kernel(input_data: cute.Tensor, output_data: cute.Tensor):
        reverse_block_kernel(input_data, output_data).launch(
            grid=BLOCKS_PER_GRID,
            block=THREADS_PER_BLOCK,
            smem=BLOCK_SIZE * 4  # 32 floats * 4 bytes per float
        );
    
    return run_kernel

def main():
    print("\n" + "="*60);
    print("Testing Puzzle 03: Shared Memory");
    print("="*60);
    
    inp = torch.arange(128, dtype=torch.float32, device='cuda');
    out = torch.zeros(128, dtype=torch.float32, device='cuda');
    
    print(f"Input (first block): {inp[:32]}");
    
    kernel = p03();
    kernel(from_dlpack(inp), from_dlpack(out));
    
    print(f"Output (first block): {out[:32]}");
    
    # Verify each block is reversed
    for block in range(4):
        start = block * 32;
        end = start + 32;
        expected_block = torch.flip(inp[start:end], [0]);
        assert torch.allclose(out[start:end], expected_block), f"Block {block} reversal failed!";
    
    print("✅ Puzzle 03 passed!");


if __name__ == "__main__":
    main();
