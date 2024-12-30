# 3D-Poisson-Solver

## Overview
This project implements a high-performance 3D Poisson equation solver using C++ with MPI, OpenMP, and CUDA. The solver achieves significant parallelism and performance improvements, leveraging multi-node, multi-core, and GPU acceleration.

## Key Features
MPI: Distributed memory parallelism across multiple nodes.

OpenMP: Multi-threading on shared-memory systems for additional parallelism.

CUDA: GPU acceleration for faster computation on supported hardware.

CUDA-aware MPI: Optimized halo exchange for efficient data transfer between nodes, reducing communication overhead.
Solvers: Red-Black & Jacobi solvers.
