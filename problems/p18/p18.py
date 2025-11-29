import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


THREADS_PER_BLOCKS = 256;

def p18():
    """C = alpha*A + beta*B"""

    @cute.kernel
    def axpby_kernel(A: cute.Tensor, B: cute.Tensor, C: cute.Tensor,
                     alpha: cutlass.Constexpr, beta: cutlass.Constexpr):
        tidx, _, _ = cute.arch.thread_idx();
        bidx, _, _ = cute.arch.block_idx();

        idx = bidx * 256 + tidx;
        if idx < cute.size(A):
            C[idx] = alpha * A[idx] + beta * B[idx];
    
    @cute.jit
    def run_kernel(A: cute.Tensor, B: cute.Tensor, C: cute.Tensor,
                   alpha: cutlass.Constexpr, beta: cutlass.Constexpr):
        N = cute.size(A);
        NUM_BLOCKS = (N + THREADS_PER_BLOCKS - 1) // THREADS_PER_BLOCKS;

        axpby_kernel(A, B, C, alpha, beta).launch(
            grid=(NUM_BLOCKS, 1, 1),
            block=(THREADS_PER_BLOCKS, 1, 1)
        );

    return run_kernel;

def main():
    print("\n" + "="*60);
    print("Puzzle 18: Fused AXPBY");
    print("="*60);

    cutlass.cuda.initialize_cuda_context();

    N = 10000;
    alpha, beta = 2.5, -1.3;

    torch.manual_seed(42);
    A_torch = torch.randn(N, dtype=torch.float32, device="cuda");
    B_torch = torch.randn(N, dtype=torch.float32, device="cuda");
    C_torch = torch.randn(N, dtype=torch.float32, device="cuda");

    A = from_dlpack(A_torch);
    B = from_dlpack(B_torch);
    C = from_dlpack(C_torch);

    kernel = p18();
    kernel_compiled = cute.compile(kernel, A, B, C, alpha, beta);
    kernel_compiled(A, B, C);

    expected = alpha * A_torch + beta * B_torch;

    assert torch.allclose(C_torch, expected, rtol=1e-5, atol=1e-6);
    print("✅ Puzzle 18 Passed!");

if __name__ == "__main__":
    main();