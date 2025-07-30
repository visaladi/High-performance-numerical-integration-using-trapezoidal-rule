#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

// CUDA device function for heavy integrand
__host__ __device__ double f(double x) {
    return x * x + sin(x) + log(1.0 + x);
}

// CUDA kernel for trapezoid integration
__global__ void trapezoid_kernel(double a, double h, int n, double* d_partial) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 1 && idx < n) {
        double x = a + idx * h;
        d_partial[idx] = f(x);
    }
}

// Host version of the function (for serial baseline)
double f_host(double x) {
    return x * x + std::sin(x) + std::log(1.0 + x);
}

__host__ int main() {
    const int n = 100000000;
    const double a = 0.0, b = 1.0;
    const double h = (b - a) / n;

    std::vector<int> block_sizes = {128, 256, 512, 1024};

    std::cout << "Mode\tBlockSize\tTime (s)\tIntegral\n";
    std::cout << "----\t---------\t--------\t--------\n";

    // --- Serial Baseline ---
    {
        double sum = 0.0;
        double t0 = clock() / (double)CLOCKS_PER_SEC;

        for (int i = 1; i < n; ++i) {
            double x = a + i * h;
            sum += f_host(x);
        }
        sum += (f_host(a) + f_host(b)) / 2.0;
        double result = h * sum;

        double t1 = clock() / (double)CLOCKS_PER_SEC;
        std::cout << "Serial\t-\t\t" << (t1 - t0) << "\t" << result << "\n";
    }

    // Allocate GPU memory
    double* d_partial;
    cudaMalloc(&d_partial, n * sizeof(double));

    std::vector<double> h_partial(n, 0.0);

    for (int blockSize : block_sizes) {
        int gridSize = (n + blockSize - 1) / blockSize;

        cudaMemset(d_partial, 0, n * sizeof(double));

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        trapezoid_kernel<<<gridSize, blockSize>>>(a, h, n, d_partial);
        cudaDeviceSynchronize();

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);  // milliseconds

        // Copy results to CPU
        cudaMemcpy(h_partial.data(), d_partial, n * sizeof(double), cudaMemcpyDeviceToHost);

        // Sum on CPU
        double sum = 0.0;
        for (int i = 1; i < n; ++i)
            sum += h_partial[i];

        sum += (f_host(a) + f_host(b)) / 2.0;
        double result = h * sum;

        std::cout << "CUDA\t" << blockSize << "\t\t" << ms / 1000.0 << "\t" << result << "\n";

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    cudaFree(d_partial);
    return 0;
}
