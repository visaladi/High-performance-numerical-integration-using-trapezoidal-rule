#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>

// A heavier integrand:
double f(double x) {
    return x*x + std::sin(x) + std::log(1.0 + x);
}

int main() {
    const int n = 100'000'000;       // number of intervals
    const double a = 0.0, b = 1.0;
    const double h = (b - a) / n;

    // how many threads to test
    std::vector<int> thread_counts =  {1, 2,3, 4,5,6,7, 8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25};

    std::cout << "Mode\tThreads\tTime (s)\tIntegral\n";
    std::cout << "----\t-------\t---------\t--------\n";

    // --- Serial version (no OpenMP) ---
    {
        double sum = 0.0;
        double t0 = omp_get_wtime();

        for (int i = 1; i < n; ++i) {
            double x = a + i * h;
            sum += f(x);
        }
        sum += (f(a) + f(b)) / 2.0;
        double result = h * sum;

        double t1 = omp_get_wtime();
        std::cout
            << "Serial\t"
            << "-" << "\t"
            << (t1 - t0) << "\t"
            << result << "\n";
    }

    // --- OpenMP versions ---
    for (int t : thread_counts) {
        omp_set_num_threads(t);

        double sum = 0.0;
        double t_start = omp_get_wtime();

        #pragma omp parallel for reduction(+:sum)
        for (int i = 1; i < n; ++i) {
            double x = a + i * h;
            sum += f(x);
        }
        sum += (f(a) + f(b)) / 2.0;
        double result = h * sum;

        double t_end = omp_get_wtime();
        std::cout
            << "OpenMP\t"
            << t << "\t"
            << (t_end - t_start) << "\t"
            << result << "\n";
    }

    return 0;
}
