import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

# Puzzle 11: Vectorized Loads
def p11(threads_per_block, num_blocks, tile_size):
    """Coalesced memory access with configurable dimensions"""
    
    @cute.kernel
    def coalesced_kernel(gmem_in: cute.Tensor, gmem_out: cute.Tensor):
        tidx, _, _ = cute.arch.thread_idx();
        bdimx, _, _ = cute.arch.block_dim();
        bidx, _, _ = cute.arch.block_idx();
        
        smem = cutlass.utils.SmemAllocator();
        smem_layout = cute.make_layout((tile_size,));
        shared = smem.allocate_tensor(cutlass.Float32, smem_layout);
        
        # Get size from tensor shape
        total_size = gmem_in.shape[0];
        
        # Coalesced access pattern
        for i in range((tile_size + bdimx - 1) // bdimx):
            idx = tidx + i * bdimx;
            if idx < tile_size:
                global_idx = bidx * tile_size + idx;
                if global_idx < total_size:
                    shared[idx] = gmem_in[global_idx];
        
        cute.arch.sync_threads();
        
        if tidx == 0:
            cute.printf("Block %d: %d threads\n", bidx, bdimx);
        
        # Write back
        for i in range((tile_size + bdimx - 1) // bdimx):
            idx = tidx + i * bdimx;
            if idx < tile_size:
                global_idx = bidx * tile_size + idx;
                if global_idx < total_size:
                    gmem_out[global_idx] = shared[idx];
    
    @cute.jit
    def run_kernel(data_in: cute.Tensor, data_out: cute.Tensor):  # ← Only 2 params
        coalesced_kernel(data_in, data_out).launch(
            grid=(num_blocks, 1, 1),
            block=(threads_per_block, 1, 1),
            smem=tile_size*4
        );
    
    return run_kernel;

def main():
    print("\n" + "="*80);
    print("Puzzle 11: Coalesced Memory Access");
    print("="*80 + "\n");
    
    cutlass.cuda.initialize_cuda_context();
    
    configs = [
        (32, 1, 128),
        (64, 1, 128),
        (32, 2, 256),
        (128, 1, 128),
    ];
    
    for threads, blocks, size in configs:
        print(f"\n--- Config: {threads} threads/block, {blocks} blocks, {size} elements ---");
        
        inp = torch.arange(size, dtype=torch.float32, device='cuda');
        out = torch.zeros(size, dtype=torch.float32, device='cuda');
        
        kernel = p11(threads_per_block=threads, num_blocks=blocks, tile_size=128);
        kernel(from_dlpack(inp), from_dlpack(out));  # ← NO size parameter!
        
        expected = torch.arange(size, dtype=torch.float32, device='cuda');
        assert torch.allclose(out, expected), f"Failed for config ({threads}, {blocks}, {size})";
        
        print(f"✅ Passed! First 8: {out[:8].cpu().numpy()}");
    
    print("\n" + "="*80);
    print("✅ All configurations passed!");
    print("="*80);


if __name__ == "__main__":
    main();