import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

SIZE = 32;
THREADS_PER_BLOCK = (SIZE, 1, 1);
BLOCKS_PER_GRID = (1, 1, 1);
ELEMENTS_PER_THREAD = 8;

def p05():
    """
    Compute running sum using register memory.
    Input: [0, 1, 2, 3, 4, 5, 6, 7, ...]
    Output: [0, 1, 3, 6, 10, 15, 21, 28, ...] (running sum per thread)
    """
    @cute.kernel
    def running_sum_kernel(input_t: cute.Tensor, output_t: cute.Tensor):
        tidx, _, _ = cute.arch.thread_idx();
        bidx, _, _ = cute.arch.block_idx();
        bdim, _, _ = cute.arch.block_dim();

        reg_layout = cute.make_layout((ELEMENTS_PER_THREAD,));
        reg_buffer = cute.make_fragment(reg_layout, cutlass.Float32);

        base_idx = (bdim * bidx + tidx) * ELEMENTS_PER_THREAD;

        for i in range(ELEMENTS_PER_THREAD):
            reg_buffer[i] = input_t[base_idx + i];

        for i in range(1, ELEMENTS_PER_THREAD):
            reg_buffer[i] = reg_buffer[i] + reg_buffer[i - 1];

        for i in range(ELEMENTS_PER_THREAD):
            output_t[base_idx + i] = reg_buffer[i];

    @cute.jit
    def run_kernel(input_data: cute.Tensor, output_data: cute.Tensor):
        running_sum_kernel(
            input_data,
            output_data
        ).launch(
            grid=BLOCKS_PER_GRID,
            block=THREADS_PER_BLOCK
        );

    return run_kernel

def main():
    print("\n" + "="*80);
    print("Puzzle 05: Register Memory");
    print("="*80 + "\n");
    inp = torch.arange(0, SIZE * ELEMENTS_PER_THREAD, dtype=torch.float32, device="cuda");
    out = torch.zeros(256, dtype=torch.float32, device="cuda");

    cutlass.cuda.initialize_cuda_context();
    kernel = p05();
    kernel(from_dlpack(inp), from_dlpack(out));
    expected = torch.tensor([0, 1, 3, 6, 10, 15, 21, 28], dtype=torch.float32, device="cuda");
    assert torch.allclose(out[:8], expected);
    print("Output: ", out[:8].cpu().numpy());
    print("Expected: ", expected.cpu().numpy());
    print("✅ Puzzle 05 Passed!");

if __name__ == "__main__":
    main();