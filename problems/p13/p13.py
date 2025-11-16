import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

# Puzzle 13: Partitioning

TOTAL_SIZE = 128;
THREADS_PER_BLOCK = 32;
VALUES_PER_THREAD = 4;

def p13a():
    """p13a: Loop-based partitioning."""

    @cute.kernel
    def partition_kernel(tensor: cute.Tensor):
        tidx, _, _ = cute.arch.thread_idx();

        # Allocate shared memory for partial sums
        smem = cutlass.utils.SmemAllocator();
        smem_layout = cute.make_layout((THREADS_PER_BLOCK,));
        shared = smem.allocate_tensor(cutlass.Float32, smem_layout);

        # Each thread processes VALUES_PER_THREAD elements
        local_sum = 0.0;
        for i in range(VALUES_PER_THREAD):
            idx = tidx * VALUES_PER_THREAD + i;
            if idx < TOTAL_SIZE:
                local_sum += tensor[idx];

        # Store local sum in shared memory
        shared[tidx] = local_sum;
        cute.arch.sync_threads();

        # Thread 0 reduces all partial sums
        if tidx == 0:
            total_sum = 0.0;
            for i in range(THREADS_PER_BLOCK):
                total_sum += shared[i];
            cute.printf("Total Sum (Loop-Based): %f\n", total_sum);

    @cute.jit
    def run_kernel(data: cute.Tensor):
        partition_kernel(data).launch(
            grid=(1, 1, 1),
            block=(THREADS_PER_BLOCK, 1, 1),
            smem=THREADS_PER_BLOCK * 4
        );

    return run_kernel


def p13b():
    """p13b: Zipped Divide"""
    @cute.kernel
    def partition_kernel(tensor: cute.Tensor):
        tidx, _, _ = cute.arch.thread_idx();
        
        smem = cutlass.utils.SmemAllocator();
        smem_layout = cute.make_layout((THREADS_PER_BLOCK,));
        shared = smem.allocate_tensor(cutlass.Float32, smem_layout);
        
        # Use zipped_divide - creates row-major (32,4):(4,1)
        tiled = cute.zipped_divide(tensor, (THREADS_PER_BLOCK,));
        
        # Simple 2D indexing works!
        local_sum = 0.0;
        for i in range(VALUES_PER_THREAD):
            local_sum += tiled[(tidx, i)];
        
        shared[tidx] = local_sum;
        cute.arch.sync_threads();
        
        if tidx == 0:
            total_sum = 0.0;
            for i in range(THREADS_PER_BLOCK):
                total_sum += shared[i];
            cute.printf("Sum: %f\n", total_sum);
    
    @cute.jit
    def run_kernel(data: cute.Tensor):
        partition_kernel(data).launch(
            grid=(1, 1, 1),
            block=(THREADS_PER_BLOCK, 1, 1),
            smem=THREADS_PER_BLOCK*4
        );
    
    return run_kernel;

def print_config():
    """Print partitioning configuration."""
    print(f"\n{'='*60}");
    print("Configuration:");
    print(f"{'='*60}");
    print(f"Total Size: {TOTAL_SIZE}");
    print(f"Threads per Block: {THREADS_PER_BLOCK}");
    print(f"Values per Thread: {VALUES_PER_THREAD}");
    print(f"Total Threads: {THREADS_PER_BLOCK}");
    print(f"Total Processed Elements: {THREADS_PER_BLOCK * VALUES_PER_THREAD}");
    print();

def main():
    print("\n" + "="*60);
    print("Puzzle 13: Partitioning");
    print("=" * 60);
    
    cutlass.cuda.initialize_cuda_context();

    print_config();

    data = torch.randn(TOTAL_SIZE, dtype=torch.float32, device="cuda");
    expected_sum = data.sum().item();

    print(f"Expected Sum: {expected_sum:.6f}\n");

    # Test p13a: Loop-based partitioning
    print("--- p13a: Loop-based partitioning ---");
    kernel_a = p13a();
    kernel_a(from_dlpack(data));
    print("✅ p13a complete\n");

    # Test p13b: Zipped Divide
    print("--- p13b: zipped_divide ---");
    kernel_b = p13b();
    kernel_b(from_dlpack(data));
    print("✅ p13b complete\n");

if __name__ == "__main__":
    main();