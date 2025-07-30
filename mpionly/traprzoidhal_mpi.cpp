#include <iostream>
#include <mpi.h>

double f(double x) {
    return x * x;
}

int main(int argc, char** argv) {
    int n = 1000000;
    double a = 0.0, b = 1.0;
    double h = (b - a) / n;
    double local_sum = 0.0, total_sum = 0.0;

    int rank, size;
    MPI_Init(&argc, &argv);               // Initialize the MPI environment
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Get the rank of the process
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Get the total number of processes

    // Determine local interval for each process
    int local_n = n / size;
    int start = rank * local_n + 1;
    int end = (rank == size - 1) ? n - 1 : (start + local_n - 1); // handle remainder in last process

    for (int i = start; i <= end; ++i) {
        double x = a + i * h;
        local_sum += f(x);
    }

    // Add f(a) and f(b) only once by rank 0
    if (rank == 0) {
        local_sum += (f(a) + f(b)) / 2.0;
    }

    // Reduce all local sums into total_sum at root process (rank 0)
    MPI_Reduce(&local_sum, &total_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double result = h * total_sum;
        std::cout << "Integral estimate (MPI) = " << result << std::endl;
    }

    MPI_Finalize(); // Clean up the MPI environment
    return 0;
}
