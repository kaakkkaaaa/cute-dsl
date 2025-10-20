import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

def p07():
    """
    Hierarchical coordinates
    """
    @cute.jit
    def coord_kernel():
        shape = ((2, 3), 4);
        layout = cute.make_layout(shape);
        cute.printf("Hierarchical Layout: {}\n", layout);

    return coord_kernel

def p07a():
    """
    Hierarchical coordinates
    """
    @cute.jit
    def coord_kernel():
        # Hierarchical shape ((2, 3), 4)
        # This means: 4 blocks, each block has 2x3 elements
        shape = ((2, 3), 4);
        layout = cute.make_layout(shape);

        cute.printf("=== Hierarchical Layout Analysis ===\n");
        cute.printf("Shape: {}\n", shape);
        cute.printf("Layout: {}\n", layout);
        cute.printf("Layout Shape: {}\n", layout.shape);
        cute.printf("Layout Stride: {}\n", layout.stride);

        # Understanding the coordinate space
        cute.printf("\n=== Coordinate Examples ===\n");
    
        # Hierarchical coordinates: ((block_row, block_col), block_index)
        coord1 = ((0, 0), 0);
        coord2 = ((1, 2), 0);
        coord3 = ((0, 0), 1);
        coord4 = ((1, 2), 3);
    
        # Convert coordinates to linear indices
        idx1 = cute.crd2idx(coord1, layout);
        idx2 = cute.crd2idx(coord2, layout);
        idx3 = cute.crd2idx(coord3, layout);
        idx4 = cute.crd2idx(coord4, layout);
    
        cute.printf("Coord {} -> Index {}\n", coord1, idx1);
        cute.printf("Coord {} -> Index {}\n", coord2, idx2); 
        cute.printf("Coord {} -> Index {}\n", coord3, idx3);
        cute.printf("Coord {} -> Index {}\n", coord4, idx4);

        # Convert indices back to coordinates
        cute.printf("\n=== Index to Coordinate Mapping ===\n");
        back_coord1 = layout.get_hier_coord(idx1);
        back_coord2 = layout.get_hier_coord(idx2);
        back_coord3 = layout.get_hier_coord(idx3);
        back_coord4 = layout.get_hier_coord(idx4);
        
        cute.printf("Index {} -> Coord {}\n", idx1, back_coord1);
        cute.printf("Index {} -> Coord {}\n", idx2, back_coord2);
        cute.printf("Index {} -> Coord {}\n", idx3, back_coord3);
        cute.printf("Index {} -> Coord {}\n", idx4, back_coord4);
        
        # Show total size
        total_elements = 2 * 3 * 4;  # 24 total elements
        cute.printf("\nTotal elements: {}\n", total_elements);

    return coord_kernel

def p07b():
    """
    Understanding Hierarchical (Nested) Layout
    """
    @cute.jit
    def hierarchical_layout_kernel():
        # Hierarchical shape: 3 blocks of 2x4 each
        hier_shape = ((2, 4), 3);
        hier_layout = cute.make_layout(hier_shape);
        cute.printf("Hierarchical layout: {}\n", hier_layout);
        cute.printf("Shape: {}\n", hier_layout.shape);
        cute.printf("Stride: {}\n", hier_layout.stride);
    
        # Compare with equivalent flat layout
        flat_shape = (2, 4, 3);
        flat_layout = cute.make_layout(flat_shape);
        cute.printf("Equivalent flat layout: {}\n", flat_layout);

        hier_tensor = cute.make_fragment(hier_layout, cutlass.Float32);
        flat_tensor = cute.make_fragment(flat_layout, cutlass.Float32);
    
        # Fill hierarchical tensor: block by block, then row by row within each block
        value = 0.0;
        for block in range(hier_shape[1]):
            for row in range(hier_shape[0][0]):
                for col in range(hier_shape[0][1]):
                    coord = ((row, col), block);
                    hier_tensor[coord] = value;
                    value += 1.0;
    
        # Fill flat tensor
        value = 0.0;
        for dim2 in range(flat_shape[2]):
            for dim0 in range(flat_shape[0]):
                for dim1 in range(flat_shape[1]):
                    coord = (dim0, dim1, dim2);
                    flat_tensor[coord] = value;
                    value += 1.0;
    
        cute.printf("\nHierarchical tensor (3 blocks of 2x4):\n");
        cute.print_tensor(hier_tensor);
    
        cute.printf("\nFlat tensor (2x4x3):\n");
        cute.print_tensor(flat_tensor);

        # Test hierarchical coordinate
        hier_coord = ((1, 3), 2);
        flat_coord = (1, 3, 2);
        hier_idx = cute.crd2idx(hier_coord, hier_layout);
        flat_idx = cute.crd2idx(flat_coord, flat_layout);

        cute.printf("\nCoordinate mapping comparison:\n");
        cute.printf("   Hierarchical coord {} maps to index {} (value: {})\n",
                     hier_coord, hier_idx, hier_tensor[hier_coord]);
        cute.printf("   Flat coord {} maps to index {} (value: {})\n",
                     flat_coord, flat_idx, flat_tensor[flat_coord]);

    return hierarchical_layout_kernel

def main():
    print("\n" + "="*60);
    print("Puzzle 07: Coordinates System");
    print("="*60 + "\n");
    cutlass.cuda.initialize_cuda_context();
    kernel = p07a();
    kernel();
    print("✅ Puzzle 07 Passed!");


if __name__ == "__main__":
    main();