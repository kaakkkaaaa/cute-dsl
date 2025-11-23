import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

# ANCHOR: p15a
def p15a():
    """
    Parallel Tree Reduction - No Separation Version
    All logic in one @cute.jit function, but caller handles memory allocation
    """
    BLOCK_SIZE = 256;
    SMEM_BYTES = BLOCK_SIZE * 4;

    @cute.kernel
    def reduce_kernel(input_t: cute.Tensor, output_t: cute.Tensor):
        tidx, _, _ = cute.arch.thread_idx();
        bidx, _, _ = cute.arch.block_idx();

        smem = cutlass.utils.SmemAllocator();
        smem_layout = cute.make_layout((BLOCK_SIZE,));
        shared = smem.allocate_tensor(cutlass.Float32, smem_layout);

        global_idx = bidx * BLOCK_SIZE + tidx;
        if global_idx < cute.size(input_t):
            shared[tidx] = input_t[global_idx];
        else:
            shared[tidx] = 0.0;

        cute.arch.sync_threads();

        stride = BLOCK_SIZE // 2;
        while stride > 0:
            if tidx < stride:
                shared[tidx] = shared[tidx] + shared[tidx + stride];
            cute.arch.sync_threads();
            stride = stride // 2;

        if tidx == 0:
            output_t[bidx] = shared[0];

    @cute.jit
    def run_kernel(data: cute.Tensor, partial_sums: cute.Tensor,
                   final_sum: cute.Tensor, num_blocks: int):
        """
        All-in-one jitted function
        Caller must be pre-allocate partial_sums and final_sum
        """
        # First reduction
        reduce_kernel(data, partial_sums).launch(
            grid=(num_blocks, 1, 1),
            block=(BLOCK_SIZE, 1, 1),
            smem=SMEM_BYTES
        );

        if num_blocks > 1:
            reduce_kernel(partial_sums, final_sum).launch(
                grid=(1, 1, 1),
                block=(BLOCK_SIZE, 1, 1),
                smem=SMEM_BYTES
            );

    return run_kernel
# ANCHOR_END: p15a

def main():
    print("\n" + "="*60);
    print("Puzzle 15: Parallel Tree Reduction");
    print("="*60);

    cutlass.cuda.initialize_cuda_context();

    # Test with various sizes
    data = torch.randn(1000, dtype=torch.float32, device="cuda");

    # Caller must handle ALL memory allocation
    BLOCK_SIZE = 256;
    num_blocks = (data.numel() + BLOCK_SIZE - 1) // BLOCK_SIZE;
    partial_sums = torch.zeros(num_blocks, dtype=torch.float32, device="cuda");
    final_sum = torch.zeros(1, dtype=torch.float32, device="cuda");

    # Call the kernel
    kernel = p15a();
    kernel(from_dlpack(data), from_dlpack(partial_sums), from_dlpack(final_sum), num_blocks);

    # Extract result based on num_blocks
    if num_blocks > 1:
        result = final_sum[0].item();
    else:
        result = partial_sums[0].item();

    expected = data.sum();

    print(f"Input size: {data.numel()}");
    print(f"Number of blocks: {num_blocks}");
    print(f"GPU result: {result:.6f}");
    print(f"Expected:   {expected:.6f}");
    print(f"Difference: {abs(result - expected.item()):.2e}");

    assert torch.allclose(torch.tensor(result), expected, rtol=1e-4);
    print("✅ Passed!");

if __name__ == "__main__":
    main();