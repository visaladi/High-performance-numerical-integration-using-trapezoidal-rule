# High-performance-numerical-integration-using-trapezoidal-rule

This project applies **High Performance Computing (HPC)** techniques to accelerate the numerical integration of a complex mathematical function using the **trapezoidal rule**. It demonstrates how parallel computing models—OpenMP, MPI, CUDA, and Hybrid CUDA+OpenMP—can drastically improve the speed and efficiency of numerical computations.

> 📘 **Course**: EC7207 - High Performance Computing  
> 🎓 **University**: Faculty of Engineering, University of Ruhuna  
> 👤 **Author**: Adikari A.M.V.S.  
> 🗓️ **Date**: July 2025

---

## 🧮 Function Integrated

The integration is done over the interval `[0, 1]` for the following function:

## math
f(x) = x² + sin(x) + log(1 + x)
💡 Implementations
Method	Technology	Description
Serial	C++	Baseline version
OpenMP	C++	Shared-memory CPU parallelism
MPI	C++	Distributed-memory parallelism
CUDA	C++/CUDA	GPU-based parallelism
Hybrid	CUDA+OpenMP	Heterogeneous CPU-GPU parallelism

## 🚀 Performance Summary
Model	Time (s)	Speedup
Serial	13.461	1.00×
OpenMP (16 threads)	1.224	11.00×
MPI (18 processes)	0.4379	30.73×
Optimized CUDA	0.0776	173.49×
Hybrid CUDA+OpenMP	0.0742	181.40×

## All implementations maintain the same result accuracy:
Integral ≈ 1.17933

## 📁 Project Structure
bash

├── serial_parallel_openmp.cpp           # Serial + OpenMP
├── more_optimized_openmp.cpp            # Optimized OpenMP
├── traprzoidhal_mpi.cpp                 # MPI (Lightweight)
├── heavy_integral_mpi.cpp               # MPI (Heavy)
├── serial_parallel_cuda.cu              # Basic CUDA
├── more_optimzed_cuda.cu                # Optimized CUDA
├── serial_parallel_openmp_cuda.cu       # Hybrid model
├── more_optimized_cuda_openmp.cu        # Optimized Hybrid
├── /plots                               # MATLAB plot scripts
└── README.md                            # Project documentation
## 📊 Plots & Visualization
Execution time and speedup graphs are generated using MATLAB:

plot_openmp.m

plot_mpi.m

plot_cuda.m

plot_hybrid.m

combined_plot.m

## ⚙️ Build Instructions
###🖥️ Serial & OpenMP

g++ serial_parallel_openmp.cpp -fopenmp -o openmp_integral
./openmp_integral
###🧪 MPI

mpic++ traprzoidhal_mpi.cpp -o mpi_integral
mpiexec -n 4 ./mpi_integral
###⚡ CUDA

nvcc more_optimzed_cuda.cu -o cuda_integral
./cuda_integral
### 🔀 Hybrid CUDA + OpenMP

nvcc -Xcompiler -fopenmp more_optimized_cuda_openmp.cu -o hybrid_integral
./hybrid_integral

###🖥️ Execution Environment
CPU: [e.g., AMD Ryzen 7 5800H]

GPU: [e.g., NVIDIA RTX 3060 6GB]

OS: [e.g., Windows 11 + WSL2 / Ubuntu 22.04]

RAM: 16 GB

Compilers: g++, nvcc, mpic++

Tools: MATLAB R2023b
