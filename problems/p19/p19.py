import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

TPB = 256;

# ANCHOR: p19
def p19():
    """Generic binary operation with lambda."""

    @cute.kernel
    def binary_kernel(A: cute.Tensor, B: cute.Tensor, C: cute.Tensor,
                      op: cutlass.Constexpr):
        tidx, _, _ = cute.arch.thread_idx();
        bidx, _, _ = cute.arch.block_idx();

        idx = bidx * TPB + tidx;
        if idx < cute.size(A):
            C[idx] = op(A[idx], B[idx]);

    @cute.jit
    def run_kernel(A: cute.Tensor, B: cute.Tensor, C: cute.Tensor,
                   op: cutlass.Constexpr):
        N = cute.size(A);
        NUM_BLOCKS = (N + TPB - 1) // TPB;
        binary_kernel(A, B, C, op).launch(
            grid=(NUM_BLOCKS, 1, 1),
            block=(TPB, 1, 1)
        );

    return run_kernel
# ANCHOR_END: p19

def main():
    print("\n" + "="*60);
    print("Puzzle 19: Custom Binary Operations");
    print("="*60);

    # Initialize CUDA context
    cutlass.cuda.initialize_cuda_context();

    # Create tensors OUTSIDE the JIT function
    A = torch.randn(10000, dtype=torch.float32, device="cuda");
    B = torch.randn(10000, dtype=torch.float32, device="cuda");
    C = torch.zeros(10000, dtype=torch.float32, device='cuda');

    # Convert to CuTe tensors
    A_tensor = from_dlpack(A);
    B_tensor = from_dlpack(B);
    C_tensor = from_dlpack(C);

    # Get the kernel function
    kernel = p19();

    # Test addition
    add_op = lambda a, b: a + b;
    kernel_compiled = cute.compile(kernel, A_tensor, B_tensor, C_tensor, add_op);
    kernel_compiled(A_tensor, B_tensor, C_tensor);
    assert torch.allclose(C, A + B);
    print("✅ Addition test passed!");

    # Test multiplication
    C.zero_();  # Reset output tensor
    mul_op = lambda a, b: a * b;
    kernel_compiled = cute.compile(kernel, A_tensor, B_tensor, C_tensor, mul_op);
    kernel_compiled(A_tensor, B_tensor, C_tensor);
    assert torch.allclose(C, A * B);
    print("✅ Multiplication test passed!");

    print("✅ All tests passed!");


if __name__ == "__main__":
    main();