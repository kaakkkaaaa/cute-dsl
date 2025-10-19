import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

def p06():
    """
    Understand shape and stride
    """
    @cute.jit
    def layout_demo():
        shape = (4, 3);
        stride = (3, 1);
        layout = cute.make_layout(shape, stride=stride);
        cute.printf("Layout created: {}\n", layout)

    return layout_demo

def p06a():
    """
    Compare column-major and row-major layouts
    """
    @cute.jit
    def layout_kernel():
        shape = (3, 4); # 3 rows, 4 columns
        # Column-major (CuTe default)
        col_layout = cute.make_layout(shape);
        cute.printf("Column-major layout: {}\n", col_layout);

        # Row-major (explicit stride)
        row_layout = cute.make_layout(shape=shape, stride=(4, 1));
        cute.printf("Row-major layout: {}\n", row_layout);

        col_tensor = cute.make_fragment(col_layout, cutlass.Float32);
        row_tensor = cute.make_fragment(row_layout, cutlass.Float32);
    
        # Fill column-major tensor with sequential data (0-11)
        value = 0.0;
        for col in range(shape[1]):
            for row in range(shape[0]):
                idx = (row, col);
                col_tensor[idx] = value;
                value += 1.0;
    
        cute.printf("\nColumn-major tensor (filled column by column 0-11):\n");
        cute.print_tensor(col_tensor);
    

        # Fill column-major tensor with sequential data (0-11)
        value = 0.0;
        for row in range(shape[0]):
            for col in range(shape[1]):
                idx = (row, col);
                row_tensor[idx] = value;
                value += 1.0;

        cute.printf("\nRow-major tensor (filled row by row 0-11):\n");
        cute.print_tensor(row_tensor);
    
        # Test coordinate mapping
        coord = (1, 2);     # Row 1, Column 2
        col_idx = cute.crd2idx(coord, col_layout);
        row_idx = cute.crd2idx(coord, row_layout);
    
        cute.printf(f"\nCoordinate {coord} maps to:\n");
        cute.printf("   Column-major index: {} (value: {})\n", col_idx, col_tensor[coord]);
        cute.printf("   Row-major index: {} (value: {})\n", row_idx, row_tensor[coord]);

    return layout_kernel

def p06b():
    """
    Explore 3D Layout with custom strides
    """
    @cute.jit
    def layout_kernel(source_tensor: cute.Tensor):
        shape = (2, 3, 4);  # 2x3x4 tensor

        # Default 3D layout
        default_layout = cute.make_layout(shape);
        cute.printf("\n3D Default layout: {}\n", default_layout);
    
        # Custom 3D layout with specific stride pattern
        custom_stride = (1, 2, 6);
        custom_layout = cute.make_layout(shape, stride=custom_stride);
        cute.printf("3D Custom layout: {}\n", custom_layout);
    
        # Create tensor with custom layout
        custom_tensor = cute.make_fragment(custom_layout, cutlass.Float32);

        for i in range(2):
            for j in range(3):
                for k in range(4):
                    value = source_tensor[i, j, k];
                    custom_tensor[i, j, k] = value;
        
        cute.printf("Custom layout tensor filled with data:\n");
        cute.print_tensor(custom_tensor);

        # Test 3D coordinate
        coord_3d = (1, 2, 1);
        default_idx = cute.crd2idx(coord_3d, default_layout);
        custom_idx = cute.crd2idx(coord_3d, custom_layout);
    
        cute.printf("\n3D Coordinate {} maps to:\n", coord_3d);
        cute.printf("- Default: {}\n", default_idx);
        cute.printf("- Custom: {}\n", custom_idx);

    return layout_kernel

def main():
    print("\n" + "="*60);
    print("Puzzle 06: Layouts");
    print("="*60 + "\n");

    cutlass.cuda.initialize_cuda_context();

    torch_data = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4);
    print("Original PyTorch tensor:");
    print(torch_data);

    # Convert to CuTe tensor
    cute_tensor = from_dlpack(torch_data);
    kernel = p06b();
    kernel(cute_tensor);
    print("✅ Puzzle 06 Passed!");

if __name__ == "__main__":
    main();