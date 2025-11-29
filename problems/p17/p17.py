import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

# ANCHOR: p17
def p17():
    """Transpose with bank conflict avoidance."""

    @cute.kernel
    def transpose_kernel(input_t: cute.Tensor, output_t: cute.Tensor):
        tidx, tidy, _ = cute.arch.thread_idx();
        bidx, bidy, _ = cute.arch.block_idx();

        TILE = 32;
        smem = cutlass.utils.SmemAllocator();
        # Add padding: (32, 33) to avoid bank conflicts
        smem_layout = cute.make_layout((TILE, TILE + 1));
        shared = smem.allocate_tensor(cutlass.Float32, smem_layout);

        M = cute.size(input_t, [0]);
        N = cute.size(input_t, [1]);

        # Load from global memory to shared memory
        in_row = bidy * TILE + tidy;
        in_col = bidx * TILE + tidx;

        if in_row < M and in_col < N:
            shared[tidy, tidx] = input_t[in_row, in_col];

        # Synchronize to ensure all threads have finished loading
        cute.arch.sync_threads();

        # Store from shared memory to global memory (transposed)
        out_row = bidx * TILE + tidy;
        out_col = bidy * TILE + tidx;

        if out_row < N and out_col < M:
            output_t[out_row, out_col] = shared[tidx, tidy];

    @cute.jit
    def run_kernel(input_t: cute.Tensor, output_t: cute.Tensor):
        """Launch the transpose kernel"""
        TILE = 32;
        M = cute.size(input_t, [0]);
        N = cute.size(input_t, [1]);

        grid_x = (N + TILE - 1) // TILE;
        grid_y = (M + TILE - 1) // TILE;

        transpose_kernel(input_t, output_t).launch(
            grid=(grid_x, grid_y, 1),
            block=(TILE, TILE, 1),
            smem=TILE * (TILE + 1) * 4
        );
    
    return run_kernel
# ANCHOR_END: p17

def main():
    print("\n" + "="*60);
    print("Puzzle 17: Matrix Transpose");
    print("="*60);

    # Initialize CUDA context
    cutlass.cuda.initialize_cuda_context();

    # Create tensors OUTSIDE the JIT function
    matrix = torch.randn(1024, 2048, dtype=torch.float32, device="cuda");
    output = torch.zeros(2048, 1024, dtype=torch.float32, device="cuda");  # Transposed shape
    
    # Convert to CuTe tensors
    input_tensor = from_dlpack(matrix);
    output_tensor = from_dlpack(output);
    
    # Get the kernel function and compile it
    kernel = p17();
    kernel_compiled = cute.compile(kernel, input_tensor, output_tensor);
    
    # Run the kernel
    kernel_compiled(input_tensor, output_tensor);
    
    # Verify correctness
    expected = matrix.t();
    assert torch.allclose(output, expected);
    print("✅ Puzzle 18 Passed!");


if __name__ == "__main__":
    main();