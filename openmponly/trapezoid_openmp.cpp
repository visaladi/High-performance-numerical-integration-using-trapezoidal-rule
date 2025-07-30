#include <iostream>
#include <omp.h>

double f(double x) {
    return x * x;
}

int main() {
    int n = 1000000;
    double a = 0.0, b = 1.0;
    double h = (b - a) / n;
    double sum = 0.0;

    #pragma omp parallel for reduction(+:sum)
    for (int i = 1; i < n; i++) {
        double x = a + i * h;
        sum += f(x);
    }

    sum += (f(a) + f(b)) / 2.0;
    double result = h * sum;

    std::cout << "Integral estimate (OpenMP) = " << result << std::endl;
    return 0;
}
