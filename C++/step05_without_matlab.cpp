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

//test with X=Y=10000, T=200...X=Y=7000, T=400...X=Y=14000, T=100
const int X = 10;                         // Number of points along X-axis
const int Y = 10;                         // Number of points along Y-axis
//static float x[X], y[X];
//static float nX[X][Y], nY[X][Y], u[X][Y], un[X][Y];

int main() {
    // Define simulation parameters
    float* x = (float*)malloc(X * sizeof(float));
    float* y = (float*)malloc(X * sizeof(float));
    float* nX = (float*)malloc(X*Y * sizeof(float));
    float* nY = (float*)malloc(X*Y * sizeof(float));
    float* u = (float*)malloc(X*Y * sizeof(float));
    float* un = (float*)malloc(X*Y * sizeof(float));
    float* tmp = (float*)malloc(X*Y * sizeof(float));

    const int T = 10;                         // Total number of time steps

    const double  c = 1.;                     // Convection coefficient
    const double dx = 2. / (X - 1);           // Step size in the X direction
    const double dy = 2. / (Y - 1);           // Step size in the Y direction
    const double dt = 0.2 * dx;               // Time step size

    int sum_values = 0;
    int num_rounds = 10;

    int chunk_size = 0;

    #ifdef PARALLEL
    #pragma omp parallel
    {
        #pragma omp single
        {
            int num_threads = omp_get_num_threads(); // Number of threads
            chunk_size = std::max(1,X / num_threads); // Calculate chunk size
        }
    }
    #endif


for (int round = 0; round < num_rounds; round++) {
    auto start = high_resolution_clock::now();

    #ifdef PARALLEL
    // Create spatial grids
    #pragma omp parallel for simd schedule(static,chunk_size)
    for (int i = 0; i < X; i++)
        x[i] = (2 * i) / (X - 1.0);

    #pragma omp parallel for simd schedule(static,chunk_size)
    for (int i = 0; i < Y; i++)
        y[i] = (2 * i) / (Y - 1.0);

    #pragma omp parallel for schedule(static,chunk_size)
    for (int i = 0; i < X; ++i) {
        #pragma omp simd
        for (int j = 0; j < Y; ++j) {
            int idx = i*X+j;
            nX[idx] = x[i];
            nY[idx] = y[j];
        }
    }

    // no simd bc of if-else
    #pragma omp parallel for collapse(2) schedule(guided)
    for (int i = 0; i < X; i++) {
        for (int j = 0; j < Y; j++){
            int idx = i*X+j;
            u[idx] = ((x[i] >= 0.5 && x[i] <= 1) && (y[j] >= 0.5 && y[j] <= 1)) ? 2.0 : 1.0;
        }
    }

    // Time-stepping loop
    for (int n = 0; n < T; n++) {
        tmp = un;
        un = u;
        u = tmp;

        /*
        #pragma omp parallel for simd schedule(static,chunk_size)
        for (int i = 0; i < X; i++) u[i*X] = un[i*X];
        #pragma omp parallel for simd schedule(static,chunk_size)
        for (int i = 0; i < Y; i++) u[i] = un[i*X];
        */

        #pragma omp parallel for schedule(static,chunk_size)
        for (int i = 1; i < X - 1; i++) {
            #pragma omp simd
            for (int j = 1; j < Y - 1; j++){
                int idx = i*X+j;
                u[idx] = un[idx] - c * (un[idx] - un[(i-1)*X+j]) * dt / dx - c * (un[idx] - un[idx-1]) * dt / dx;
            }
        }
    }
    // Boundary conditions
    #pragma omp parallel for simd schedule(static,chunk_size)
    for (int i = 0; i < X; i++) {
        u[i*X] = 1.;
        u[i*X+(Y - 1)] = 1.;
    }
    #pragma omp parallel for simd schedule(static,chunk_size)
    for (int i = 0; i < Y; i++){
        u[i] = 1.;
        u[X*(X - 1)+i] = 1.;
    } 


        std::vector<double> x2(X), y2(Y);

    for (int i = 0; i < X; i++)
        x2[i] = (2 * i) / (X - 1.0);

    for (int i = 0; i < Y; i++)
        y2[i] = (2 * i) / (Y - 1.0);

    std::vector<std::vector<double>> nX2(X, std::vector<double>(Y)), nY2(X, std::vector<double>(Y));
    for (int i = 0; i < X; ++i) {
        for (int j = 0; j < Y; ++j) {
            nX2[i][j] = x2[i]; 
            nY2[i][j] = y2[j]; 
        }
    }

    std::vector<std::vector<double>> u2(X, std::vector<double>(Y)), un2(X, std::vector<double>(Y));
    for (int i = 0; i < X; i++){
        for (int j = 0; j < Y; j++)
            u2[i][j] = ((x2[i] >= 0.5 && x2[i] <= 1) && (y2[j] >= 0.5 && y2[j] <= 1)) ? 2.0 : 1.0;        
    }

 
    // Time-stepping loop
    for (int n = 0; n < T; n++) {
        un2 = u2;

        for (int i = 1; i < X-1; i++) {
            for (int j = 1; j < Y-1; j++)
            u2[i][j] = un2[i][j] - c * (un2[i][j] - un2[i-1][j]) * dt / dx - c * (un2[i][j] - un2[i][j-1]) * dt / dx;
        }

        // Boundary conditions
        for (int i = 0; i < Y; i++) u2[i][0] = 1.;
        for (int i = 0; i < X; i++) u2[0][i] = 1.;

        for (int i = 0; i < X; i++) u2[i][Y-1] = 1.;
        for (int i = 0; i < Y; i++) u2[X-1][i] = 1.;
    }
    for (int i = 0; i < X; i++){
        for (int j = 0; j < Y; j++){
            std::cout<<u2[i][j]<<std::endl;  
            int idx = i*X+j;  
            std::cout<<u[idx]<<std::endl; 
        }           
    }
    #else
    // Create spatial grids
    for (int i = 0; i < X; i++)
        x[i] = (2 * i) / (X - 1.0);

    for (int i = 0; i < Y; i++)
        y[i] = (2 * i) / (Y - 1.0);

    for (int i = 0; i < X; ++i) {
        for (int j = 0; j < Y; ++j) {
            int idx = i*X+j;
            nX[idx] = x[i];
            nY[idx] = y[j];
        }
    }

    for (int i = 0; i < X; i++) {
        for (int j = 0; j < Y; j++){
            int idx = i*X+j;
            u[idx] = ((x[i] >= 0.5 && x[i] <= 1) && (y[j] >= 0.5 && y[j] <= 1)) ? 2.0 : 1.0;
        }
    }


    // Time-stepping loop
    for (int n = 0; n < T; n++) {
        tmp = un;
        un = u;
        u = tmp;

        for (int i = 0; i < X; i++) u[i*X] = un[i*X];
        for (int i = 0; i < Y; i++) u[i] = un[i*X];

        for (int i = 1; i < X - 1; i++) {
            for (int j = 1; j < Y - 1; j++){
                int idx = i*X+j;
                u[idx] = un[idx] - c * (un[idx] - un[(i-1)*X+j]) * dt / dx - c * (un[idx] - un[idx-1]) * dt / dx;
            }
        }

        // Boundary conditions
        for (int i = 0; i < X; i++) {
            u[i*X] = 1.;
            u[i*X+(Y - 1)] = 1.;
        }
        for (int i = 0; i < Y; i++){
            u[i] = 1.;
            u[X*(X - 1)+i] = 1.;
        } 
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
