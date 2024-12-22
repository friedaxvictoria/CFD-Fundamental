#include <vector>
#include <algorithm>
#include <iostream>
#include <cmath>
#include <chrono>
#include <omp.h>
using namespace std::chrono;

////////////////////////////////////////////////////////////
// Step 5: 2D Linear Convection
////////////////////////////////////////////////////////////

const int X = 25000;                         // Number of points along X-axis
const int Y = 25000;                         // Number of points along Y-axis
static float x[X], y[X];
static float nX[X][Y], nY[X][Y], u[X][Y], un[X][Y];

//std::vector<std::vector<double>> u(X, std::vector<double>(Y)), un(X, std::vector<double>(Y));
//std::vector<std::vector<double>> nX(X, std::vector<double>(Y)), nY(X, std::vector<double>(Y));

int main() {
    // Define simulation parameters

    const int T = 100;                         // Total number of time steps

    const double  c = 1.;                     // Convection coefficient
    const double dx = 2. / (X - 1);           // Step size in the X direction
    const double dy = 2. / (Y - 1);           // Step size in the Y direction
    const double dt = 0.2 * dx;               // Time step size

    int sum_values = 0;
    int num_rounds = 1;


for (int round = 0; round < num_rounds; round++) {
    auto start = high_resolution_clock::now();

    #ifdef PARALLEL
    // Create spatial grids
    #pragma omp parallel for simd
    for (int i = 0; i < X; i++)
        x[i] = (2 * i) / (X - 1.0);

    #pragma omp parallel for simd
    for (int i = 0; i < Y; i++)
        y[i] = (2 * i) / (Y - 1.0);

    #pragma omp parallel for simd collapse(2) 
    for (int i = 0; i < X; ++i) {
        for (int j = 0; j < Y; ++j) {
            nX[i][j] = x[i];
            nY[i][j] = y[j];
        }
    }

    // no simd bc of i-else
    #pragma omp parallel for collapse(2) 
    for (int i = 0; i < X; i++) {
        for (int j = 0; j < Y; j++)
            u[i][j] = ((x[i] >= 0.5 && x[i] <= 1) && (y[j] >= 0.5 && y[j] <= 1)) ? 2.0 : 1.0;
    }


    // Time-stepping loop
    for (int n = 0; n < T; n++) {
        std::copy(&u[0][0], &u[0][0] + X * Y, &un[0][0]);
        //std::copy(std::begin(u), std::end(u), std::begin(un));

        #pragma omp parallel for simd collapse(2) 
        for (int i = 1; i < X - 1; i++) {
            for (int j = 1; j < Y - 1; j++)
                u[i][j] = un[i][j] - c * (un[i][j] - un[i - 1][j]) * dt / dx - c * (un[i][j] - un[i][j - 1]) * dt / dx;
        }

        // Boundary conditions
        #pragma omp parallel for simd
        for (int i = 0; i < Y; i++) u[i][0] = 1.;
        #pragma omp parallel for simd
        for (int i = 0; i < X; i++) u[0][i] = 1.;

        #pragma omp parallel for simd
        for (int i = 0; i < X; i++) u[i][Y - 1] = 1.;
        #pragma omp parallel for simd
        for (int i = 0; i < Y; i++) u[X - 1][i] = 1.;
    }

    #else
    // Create spatial grids
    for (int i = 0; i < X; i++)
        x[i] = (2 * i) / (X - 1.0);

    for (int i = 0; i < Y; i++)
        y[i] = (2 * i) / (Y - 1.0);

    for (int i = 0; i < X; ++i) {
        for (int j = 0; j < Y; ++j) {
            nX[i][j] = x[i];
            nY[i][j] = y[j];
        }
    }

    for (int i = 0; i < X; i++) {
        for (int j = 0; j < Y; j++)
            u[i][j] = ((x[i] >= 0.5 && x[i] <= 1) && (y[j] >= 0.5 && y[j] <= 1)) ? 2.0 : 1.0;
    }


    // Time-stepping loop
    for (int n = 0; n < T; n++) {
        std::copy(&u[0][0], &u[0][0] + X * Y, &un[0][0]);

        for (int i = 1; i < X - 1; i++) {
            for (int j = 1; j < Y - 1; j++)
                u[i][j] = un[i][j] - c * (un[i][j] - un[i - 1][j]) * dt / dx - c * (un[i][j] - un[i][j - 1]) * dt / dx;
        }

        // Boundary conditions
        for (int i = 0; i < Y; i++) u[i][0] = 1.;
        for (int i = 0; i < X; i++) u[0][i] = 1.;

        for (int i = 0; i < X; i++) u[i][Y - 1] = 1.;
        for (int i = 0; i < Y; i++) u[X - 1][i] = 1.;
    }
    #endif
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    std::cout << "microseconds: " << duration.count() << std::endl;
    auto duration_sec = duration_cast<seconds>(stop - start);
    std::cout << "seconds: " << duration_sec.count() << std::endl;
    sum_values += duration.count();
    }
    std::cout << "average microseconds: " << sum_values / num_rounds << std::endl;
    return 0;
}
