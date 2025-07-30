#include <iostream>
#include <cuda_runtime.h>

__host__ __device__ double f(double x) {
    return x * x;  // function to integrate
}

__global__ void integrate_kernel(double a, double h, int n, double* partial_sums) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 1 && i < n) {
        double x = a + i * h;
        partial_sums[i] = f(x);
    }
}

int main() {
    int n = 1000000;
    double a = 0.0, b = 1.0;
    double h = (b - a) / n;

    double* d_partial_sums;
    double* h_partial_sums = new double[n];

    // Allocate memory on GPU
    cudaMalloc(&d_partial_sums, n * sizeof(double));
    cudaMemset(d_partial_sums, 0, n * sizeof(double));

    // Launch CUDA kernel
    int threadsPerBlock = 256;
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    integrate_kernel<<<blocks, threadsPerBlock>>>(a, h, n, d_partial_sums);
    cudaDeviceSynchronize();

    // Copy results back to CPU
    cudaMemcpy(h_partial_sums, d_partial_sums, n * sizeof(double), cudaMemcpyDeviceToHost);

    // Final summation on CPU
    double sum = 0.0;
    for (int i = 1; i < n; i++) {
        sum += h_partial_sums[i];
    }

    sum += (f(a) + f(b)) / 2.0;
    double result = h * sum;

    std::cout << "Integral estimate (CUDA) = " << result << std::endl;

    // Cleanup
    delete[] h_partial_sums;
    cudaFree(d_partial_sums);

    return 0;
}
