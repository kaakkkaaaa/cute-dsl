import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

# ANCHOR: p16a
def p16a():
    """
    Find minimum value using parallel reduction.
    """
    BLOCK_SIZE = 256;
    SMEM_BYTES = BLOCK_SIZE * 4;

    @cute.kernel
    def min_kernel(input_t: cute.Tensor, output_t: cute.Tensor):
        tidx, _, _ = cute.arch.thread_idx();
        bidx, _, _ = cute.arch.block_idx();

        # Allocate shared memory for min values
        smem = cutlass.utils.SmemAllocator();
        smem_layout = cute.make_layout((BLOCK_SIZE,));
        smin = smem.allocate_tensor(cutlass.Float32, smem_layout);

        # Load data with bounds checking
        idx = bidx * BLOCK_SIZE + tidx;
        if idx < cute.size(input_t):
            smin[tidx] = input_t[idx];
        else:
            smin[tidx] = 1e9;

        cute.arch.sync_threads();

        # Tree reduction for min
        stride = BLOCK_SIZE // 2;
        while stride > 0:
            if tidx < stride:
                if smin[tidx + stride] < smin[tidx]:
                    smin[tidx] = smin[tidx + stride];
            cute.arch.sync_threads();
            stride = stride // 2;

        # Thread 0 writes block result
        if tidx == 0:
            output_t[bidx] = smin[0];

    @cute.jit
    def run_kernel(data: cute.Tensor, partial_min: cute.Tensor, num_blocks: int):
        """Launch min reduction kernel"""
        min_kernel(data, partial_min).launch(
            grid=(num_blocks, 1, 1),
            block=(BLOCK_SIZE, 1, 1),
            smem=SMEM_BYTES
        );

    return run_kernel
# ANCHOR_END: p16a

# ANCHOR: p16b
def p16b():
    """
    Find maximum value using parallel reduction
    """

    BLOCK_SIZE = 256;
    SMEM_BYTES = BLOCK_SIZE * 4;

    @cute.kernel
    def max_kernel(input_t: cute.Tensor, output_t: cute.Tensor):
        tidx, _, _ = cute.arch.thread_idx();
        bidx, _, _ = cute.arch.block_idx();

        # Allocate shared memory for max values
        smem = cutlass.utils.SmemAllocator();
        layout_smem = cute.make_layout((BLOCK_SIZE,));
        smax = smem.allocate_tensor(cutlass.Float32, layout_smem);

        # Load data with bounds checking
        idx = bidx * BLOCK_SIZE + tidx;
        if idx < cute.size(input_t):
            smax[tidx] = input_t[idx];
        else:
            smax[tidx] = -1e9;

        cute.arch.sync_threads();

        # Tree reduction for max
        stride = BLOCK_SIZE // 2;
        while stride > 0:
            if tidx < stride:
                if smax[tidx + stride] > smax[tidx]:
                    smax[tidx] = smax[tidx + stride];
            cute.arch.sync_threads();
            stride = stride // 2;

        # Thread 0 writes block result
        if tidx == 0:
            output_t[bidx] = smax[0];          

    @cute.jit
    def run_kernel(data: cute.Tensor, partial_max: cute.Tensor, num_blocks: int):
        """Launch max reduction kernel"""
        max_kernel(data, partial_max).launch(
            grid=(num_blocks, 1, 1),
            block=(BLOCK_SIZE, 1, 1),
            smem=SMEM_BYTES
        );

    return run_kernel

def main():
    print("\n" + "="*60);
    print("Puzzle 16a: Min Reduction");
    print("="*60);

    cutlass.cuda.initialize_cuda_context();

    data = torch.randn(100000, dtype=torch.float32, device="cuda");

    # Allocate memory
    BLOCK_SIZE = 256;
    num_blocks = (data.numel() + BLOCK_SIZE - 1) // BLOCK_SIZE;
    partial_min = torch.zeros(num_blocks, dtype=torch.float32, device="cuda");

    # Run kernel
    kernel = p16a();
    kernel(from_dlpack(data), from_dlpack(partial_min), num_blocks);

    # Get final result
    min_val = partial_min.min().item();
    expected_min = data.min();

    print(f"Input size: {data.numel()}");
    print(f"Number of blocks: {num_blocks}");
    print(f"GPU min: {min_val:.6f}");
    print(f"Expected min: {expected_min:.6f}");

    assert torch.allclose(torch.tensor(min_val), expected_min);
    print("✅ Passed!");

    print("\n" + "="*60);
    print("Puzzle 16b: Max Reduction");
    print("="*60);
    
    partial_max = torch.zeros(num_blocks, dtype=torch.float32, device='cuda');

    # Run kernel
    kernel = p16b();
    kernel(from_dlpack(data), from_dlpack(partial_max), num_blocks);

    # Get final result
    max_val = partial_max.max().item();
    expected_max = data.max();

    print(f"Input size: {data.numel()}");
    print(f"Number of blocks: {num_blocks}");
    print(f"GPU max: {max_val:.6f}");
    print(f"Expected max: {expected_max:.6f}");
    
    assert torch.allclose(torch.tensor(max_val), expected_max);
    print("✅ Passed!");

if __name__ == "__main__":
    main();