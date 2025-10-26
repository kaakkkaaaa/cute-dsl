import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

def p09a():
    @cute.jit
    def identity_kernel():
        # Example 1: Basic 2D Identity Tensor
        cute.printf("="*60 + "\n");
        cute.printf("Example 1: Basic 2D Identity Tensor\n");
        cute.printf("="*60 + "\n\n");
    
        shape = (4, 8);
        cute.printf("Creating a 2D identity tensor with shape: {}\n", shape);
    
        # Create identity layout
        identity = cute.make_identity_tensor(shape);
        cute.printf("\nIdentity tensor layout:\n");

        # Manually print identity tensor values by iterating
        for i in range(shape[0]):
            cute.printf("  ");
            for j in range(shape[1]):
                coord = identity[i, j];
                cute.printf("(%d,%d) ", coord[0], coord[1])
            cute.printf("\n");

    return identity_kernel

def p09b():
    @cute.jit
    def verify_tiling_kernel():
        # Example 2: Verify Tiling/Partition Strategy
        cute.printf("="*60 + "\n");
        cute.printf("Example 2: Verify Tiling\n");
        cute.printf("="*60 + "\n\n");

        cute.printf("Problem: Dividing 8x8 matrix into 2x2 tiles of 4x4 each\n");

        full_shape = (8, 8);
        full_identity = cute.make_identity_tensor(full_shape);
        tile_size = 4;

        cute.printf("Tile (0,0) - Top-Left:\n");
        for i in range(0, tile_size):
            cute.printf("  ");
            for j in range(0, tile_size):
                coord = full_identity[i, j];
                cute.printf("(%d,%d) ", coord[0], coord[1]);
            cute.printf("\n");

        cute.printf("\nTile (0,1) - Top-Right:\n");
        for i in range(0, tile_size):
            cute.printf("  ");
            for j in range(tile_size, 2*tile_size):
                coord = full_identity[i, j];
                cute.printf("(%d,%d) ", coord[0], coord[1]);
            cute.printf("\n");
        
        cute.printf("\nTile (1,0) - Bottom-Left:\n");
        for i in range(tile_size, 2*tile_size):
            cute.printf("  ");
            for j in range(0, tile_size):
                coord = full_identity[i, j];
                cute.printf("(%d,%d) ", coord[0], coord[1]);
            cute.printf("\n");

    return verify_tiling_kernel

def main():
    print("\n" + "="*80);
    print("Puzzle 09b: Identity Tensor");
    print("="*80 + "\n");

    cutlass.cuda.initialize_cuda_context();
    kernel = p09b();
    kernel();

if __name__ == "__main__":
    main();
