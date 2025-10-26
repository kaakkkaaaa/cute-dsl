import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

def p08a():
    """Basic composition with actual values to visualize memory layout"""
    @cute.jit
    def compose_kernel():
        cute.printf("="*60 + "\n");
        cute.printf("Example 1: Basic Composition with Values\n");
        cute.printf("="*60 + "\n\n");

        # ================================================================
        # Step 1: Create base layout A and fill with sequential values
        # ================================================================
        cute.printf("STEP 1: Create Layout A (base layout)\n");
        cute.printf("-" * 40 + "\n");

        base = cute.make_layout((6, 2), stride=(8, 2));
        cute.printf("Layout A: {}\n", base);
        cute.printf("  Shape: (6, 2) - 6 rows, 2 columns\n");
        cute.printf("  Stride: (8, 2) - row stride=8, col stride=2\n\n");
    
        cute.printf("How Layout A maps coordinates to memory offsets:\n");
        cute.printf("offset = i * 8 + j * 2\n\n");

        cute.printf("Memory offset table:\n");
        cute.printf("  Coord | Calculation | Offset\n");
        cute.printf("  ------|-------------|-------\n");

        for i in cutlass.range_constexpr(6):
            for j in cutlass.range_constexpr(2):
                offset = base((i, j));
                cute.printf("  ({},{})  | {}*8 + {}*2  | {}\n", i, j, i, j, offset);
        cute.printf("\n");

        # Create fragment and fill with values that show the pattern
        # We'll use the offset itself as the value to make it clear
        tensor_a = cute.make_fragment(base, cutlass.Float32);

        cute.printf("Filling Fragment A with offset-based values:\n");
        for i in cutlass.range_constexpr(6):
            for j in cutlass.range_constexpr(2):
                offset = base((i, j));
                tensor_a[i, j] = cutlass.Float32(offset * 100);

        cute.printf("\nFragment A (values = offset × 100):\n");
        cute.print_tensor(tensor_a);
        cute.printf("\n");
    
    return compose_kernel

def p08b():
    """
    Basic composition with actual values to visualize memory layout
    """
    @cute.jit
    def compose_kernel():
        cute.printf("="*60 + "\n");
        cute.printf("Example 2: Basic Composition with Values\n");
        cute.printf("="*60 + "\n\n");

        # ================================================================
        # Step 2: Create layout B (tiler)
        # ================================================================
        cute.printf("STEP 2: Understanding Layout B (coordinate transformer)\n");
        cute.printf("-" * 40 + "\n");

        tiler = cute.make_layout((4, 3), stride=(3, 1));
        cute.printf("Layout B: {}\n", tiler);
        cute.printf("  Shape: (4, 3) - 4 rows, 3 columns\n");
        cute.printf("  Stride: (3, 1) - row stride=3, col stride=1\n\n");

        cute.printf("How Layout B maps coordinates:\n");
        cute.printf("  offset = i * 3 + j * 1\n\n");

        cute.printf("Coordinate mapping table:\n");
        cute.printf("  Coord | Calculation | B(i,j)\n");
        cute.printf("  ------|-------------|-------\n");

        for i in cutlass.range_constexpr(4):
            for j in cutlass.range_constexpr(3):
                result = tiler((i, j));
                cute.printf("  ({},{})  | {}*3 + {}*1  | {}\n", i, j, i, j, result);
        cute.printf("\n");

        # Create fragment for B showing what coordinate it maps to
        frag_b = cute.make_fragment(tiler, cutlass.Int32);
        for i in cutlass.range_constexpr(4):
            for j in cutlass.range_constexpr(3):
                frag_b[i, j] = tiler((i, j));

        cute.printf("Fragment B (shows B(i, j) results):\n");
        cute.print_tensor(frag_b);
        cute.printf("\n");

    return compose_kernel

def p08c():
    """
    Basic composition with actual values to visualize memory layout
    """
    @cute.jit
    def compose_kernel():
        cute.printf("="*60 + "\n");
        cute.printf("STEP 3: Composition R = A ∘ B\n");
        cute.printf("=" * 60 + "\n\n");
        
        base = cute.make_layout((6, 2), stride=(8, 2));
        tiler = cute.make_layout((4, 3), stride=(3, 1));
        composed = cute.composition(base, tiler);
        cute.printf("Composed Layout R: {}\n", composed);
        cute.printf("  R(c) = A(B(c)) for each coordinate c\n\n");

        frag_a = cute.make_fragment(base, cutlass.Float32);
        
        # *** FIX 1: Fill Fragment A with values ***
        cute.printf("Filling Fragment A (value = offset × 100):\n");
        for i in cutlass.range_constexpr(6):
            for j in cutlass.range_constexpr(2):
                offset = base((i, j));
                value = cutlass.Float32(offset * 100);
                frag_a[i, j] = value;
                cute.printf("  frag_a[{},{}] = {}\n", i, j, value);
        cute.printf("\n");
        
        # Detailed mapping table
        cute.printf("Complete mapping:\n");
        cute.printf("B(i,j) | B result | A coord | A offset | Value    | R coord\n");
        cute.printf("-------|----------|---------|----------|----------|----------\n");
        
        for i in cutlass.range_constexpr(4):
            for j in cutlass.range_constexpr(3):
                b_result = tiler((i, j));
                i_a = b_result // 2;
                j_a = b_result % 2;
                a_offset = base((i_a, j_a));
                value = frag_a[i_a, j_a];
                
                # R's hierarchical coordinate
                i0 = i // 2;
                i1 = i % 2;
                
                # *** FIX 2: Correct format string ***
                cute.printf("({},{})  | {}        | ({},{})   | {}       | {}     | (({},{}),{})\n",
                           i, j, b_result, i_a, j_a, a_offset, value, i0, i1, j);
        cute.printf("\n");
        
        # Create and fill Fragment R
        frag_r = cute.make_fragment(composed, cutlass.Float32);
        
        cute.printf("Filling Fragment R:\n");
        for i in cutlass.range_constexpr(4):
            for j in cutlass.range_constexpr(3):
                b_result = tiler((i, j));
                i_a = b_result // 2;
                j_a = b_result % 2;
                value = frag_a[i_a, j_a];
                
                # R's hierarchical coordinate
                i0 = i // 2;
                i1 = i % 2;
                
                frag_r[(i0, i1), j] = value;
                cute.printf("  R(({},{}),{}) = A({},{}) = {}\n", i0, i1, j, i_a, j_a, value);
        cute.printf("\n");
        
        cute.printf("Fragment R visual:\n");
        cute.print_tensor(frag_r);
        cute.printf("\n");

    return compose_kernel

def main():
    print("\n" + "="*60);
    print("Puzzle 08: Layout Composition");
    print("="*60);
    kernel = p08c();
    kernel();
    print("✅ Puzzle 08 Passed!");


if __name__ == "__main__":
    main();