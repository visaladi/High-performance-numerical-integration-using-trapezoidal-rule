#include <iostream>
#include <vector>
#include <cmath>
#include <mpi.h>

// A heavier integrand
double f(double x) {
    return x * x + std::sin(x) + std::log(1.0 + x);
}

int main(int argc, char** argv) {
    const int n = 100000000;  // Number of intervals
    const double a = 0.0, b = 1.0;
    const double h = (b - a) / n;

    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Timing start
    double t_start = MPI_Wtime();

    // Each process calculates a range of indices
    int local_n = n / size;
    int start = rank * local_n + 1;
    int end = (rank == size - 1) ? n - 1 : (start + local_n - 1);  // last process takes remainder

    // Local integral
    double local_sum = 0.0;
    for (int i = start; i <= end; ++i) {
        double x = a + i * h;
        local_sum += f(x);
    }

    // Add edge values only in rank 0
    if (rank == 0) {
        local_sum += (f(a) + f(b)) / 2.0;
    }

    // Reduce all local sums into global sum
    double total_sum = 0.0;
    MPI_Reduce(&local_sum, &total_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    double t_end = MPI_Wtime();

    if (rank == 0) {
        double result = h * total_sum;
        std::cout << "Mode\tProcs\tTime (s)\tIntegral\n";
        std::cout << "----\t-----\t--------\t--------\n";
        std::cout << "MPI\t" << size << "\t" << (t_end - t_start) << "\t" << result << "\n";
    }

    MPI_Finalize();
    return 0;
}
