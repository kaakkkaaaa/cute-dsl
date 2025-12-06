# CuTeDSL GPU Puzzles 🧩 + Bonus 📦

A collection of GPU programming puzzles and examples using NVIDIA's CuTeDSL (Domain-Specific Language) for high-performance CUDA kernel development.

## What is CuTeDSL?

CuTeDSL is a Python-native interface for writing high-performance CUDA kernels based on CUTLASS and CuTe concepts. It enables:
- **Smooth learning curve** - Python syntax instead of deep C++ template metaprogramming
- **Fast iteration** - Orders of magnitude faster compile times than C++
- **Native DL framework integration** - No glue code needed for PyTorch/other frameworks
- **No performance compromises** - Full access to low-level GPU hardware

## Getting Started

### Installation

```bash
pip install nvidia-cutlass-dsl==4.2.1
```

Or install from requirements:
```bash
pip install -r requirements.txt
```

### Prerequisites

- NVIDIA GPU with CUDA support (Hopper/Ampere/Ada recommended)
- CUDA Toolkit matching your `cuda-python` version
- Python 3.8+

## Puzzles

40 progressive GPU programming challenges organized into 8 parts:

### Part 1: Foundations (Puzzles 0-4)
Basic GPU concepts - thread indexing, global memory, shared memory, registers

### Part 2: Tensors & Layouts (Puzzles 5-9)
CuTe tensor basics, layout composition, coordinate systems, identity tensors

### Part 3: Data Movement (Puzzles 10-14)
Copy operations, vectorized loads, tiling, partitioning, zipped division

### Part 4: Parallel Patterns (Puzzles 15-19)
Reductions (sum/min/max), matrix transpose, fused operations (AXPBY), custom binary ops

### Part 5: Advanced Techniques (Puzzles 20-24) - TBD
Pipelining, Tensor Cores, async copy, swizzling, basic GEMM

### Part 6: Reductions & Pooling (Puzzles 25-29) - TBD
Axis sum, warp reductions, max/avg pooling (1D/2D)

### Part 7: Convolutions (Puzzles 30-34) - TBD
Conv1D, Conv2D, separable/dilated/transposed convolutions

### Part 8: Activation & Attention (Puzzles 35-39) - TBD
Softmax, LayerNorm, scaled dot-product attention, multi-head attention, Flash Attention

## Key Concepts

Through these puzzles, you'll learn:
- GPU kernel programming with `@cute.kernel` decorator
- Thread hierarchy and memory layouts
- Coalesced memory access patterns
- Vectorization techniques (e.g., loading 8 contiguous elements)
- Layout algebra (`logical_divide`, `zipped_divide`, tiling)
- Performance optimization and benchmarking

## Usage

Run puzzle scripts directly:
```bash
python puzzle_01.py  # Hello GPU
python puzzle_16.py  # Parallel Reduction
python puzzle_40.py  # Flash Attention
```

Each puzzle includes:
- Clear problem description and learning objectives
- Complete working implementation
- Test verification with PyTorch reference
- Progressive difficulty from basics to advanced

## Documentation

- [NVIDIA CUTLASS Documentation](https://docs.nvidia.com/cutlass/)
- [CuTe DSL API Reference](https://nvidia.github.io/cutlass/)

## License

Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.  
SPDX-License-Identifier: LicenseRef-NvidiaProprietary

## Acknowledgments

Built on NVIDIA's CUTLASS library and CuTe tensor algebra framework.
