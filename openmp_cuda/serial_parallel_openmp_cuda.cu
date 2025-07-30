#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <omp.h>

// Define function usable by both host and device
__host__ __device__ double f(double x) {
    return x * x + sin(x) + log(1.0 + x);
}

// CUDA kernel to compute partial values
__global__ void trapezoid_kernel(double a, double h, int n, double* d_partial) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 1 && idx < n) {
        double x = a + idx * h;
        d_partial[idx] = f(x);
    }
}

__host__ int main() {
    const int n = 100000000;
    const double a = 0.0, b = 1.0;
    const double h = (b - a) / n;

    std::vector<int> block_sizes = {128, 256, 512, 1024};

    std::cout << "Mode\tBlockSize\tTime (s)\tIntegral\n";
    std::cout << "----\t---------\t--------\t--------\n";

    // Allocate device memory
    double* d_partial;
    cudaMalloc(&d_partial, n * sizeof(double));

    // Host memory for results
    std::vector<double> h_partial(n, 0.0);

    for (int blockSize : block_sizes) {
        int gridSize = (n + blockSize - 1) / blockSize;

        // Clear GPU memory
        cudaMemset(d_partial, 0, n * sizeof(double));

        // CUDA timing
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        // Launch kernel
        trapezoid_kernel<<<gridSize, blockSize>>>(a, h, n, d_partial);
        cudaDeviceSynchronize();

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        // Copy GPU results to CPU
        cudaMemcpy(h_partial.data(), d_partial, n * sizeof(double), cudaMemcpyDeviceToHost);

        // Use OpenMP to parallelize the final summation
        double sum = 0.0;
        #pragma omp parallel for reduction(+:sum)
        for (int i = 1; i < n; ++i) {
            sum += h_partial[i];
        }

        // Add endpoints
        sum += (f(a) + f(b)) / 2.0;
        double result = h * sum;

        std::cout << "Hybrid\t" << blockSize << "\t\t" << milliseconds / 1000.0 << "\t" << result << "\n";

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    cudaFree(d_partial);
    return 0;
}
