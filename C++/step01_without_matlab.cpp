#include <cmath>
#include <vector>
#include <algorithm>
#include <iostream>
#include <chrono>
#include <omp.h>
#include <cstring>
using namespace std::chrono;

////////////////////////////////////////////////////////////
// Step 1: 1D Linear Convection
////////////////////////////////////////////////////////////
//test with X=800'000'000, T=50...X=400'000'000, T=100...X=200'000'000, T=200
const int X = 200'000'000;                    // Number of spatial points

//static float x[X], u[X], un[X];
int main() {
    // Simulation parameters
    const int T = 200;                    // Number of time steps
    const int c = 1;                     // Wave speed

    const float dx = 2.0 / (X - 1);     // Spatial step size
    const float dt = 0.025;             // Time step size

    // Initialize spatial grid and initial condition
    float* x = (float*)malloc(X * sizeof(float));
    float* u = (float*)malloc(X * sizeof(float));
    float* un = (float*)malloc(X * sizeof(float));
    float* tmp = (float*)malloc(X * sizeof(float));
    
    //std::vector<double> x2(X), u2(X), un2(X);

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
            #pragma omp parallel for simd schedule(static, chunk_size)
            for (int i = 0; i < X; i++) {
                x[i] = (5.0 * i) / (X - 1);
            }

            #pragma omp parallel for schedule(guided)
            for (int i = 0; i < X; i++) {
                u[i] = (x[i] >= 0.5 && x[i] <= 1) ? 2 : 1;
            }

            un[0] = u[0];
            for (int n = 0; n < T; n++) {
                tmp = un;
                un = u;
                u = tmp;
                
                #pragma omp parallel for simd schedule(static, chunk_size)
                for (int i = 1; i < X; i++) {
                    u[i] = un[i] - c * (un[i] - un[i - 1]) * dt / dx;
                }
            }

            #else
            for (int i = 0; i < X; i++) {
                x[i] = (5.0 * i) / (X - 1);
            }

            for (int i = 0; i < X; i++) {
                u[i] = (x[i] >= 0.5 && x[i] <= 1) ? 2 : 1;
            }

            u[0] = un[0];
            for (int n = 0; n < T; n++) {
                //std::copy(std::begin(u), std::end(u), std::begin(un));
                //std::copy(u, u + X, un);
                tmp = un;
                un = u;
                u = tmp;
                for (int i = 1; i < X; i++) {
                    u[i] = un[i] - c * (un[i] - un[i - 1]) * dt / dx;
                }
            }
            #endif


        //std::copy(std::begin(testu), std::end(testu), std::begin(u));

            /*
            for (int n = 0; n < T; n++) {
                std::copy(std::begin(testu), std::end(testu), std::begin(testun));
                for (int i = 1; i < X; i++) {
                    testu[i] = testun[i] - c * (testun[i] - testun[i - 1]) * dt / dx;
                }
            }
*/
             

            auto stop = high_resolution_clock::now();
            auto duration = duration_cast<microseconds>(stop - start);
            std::cout << "microseconds: " << duration.count() << std::endl;
            auto duration_sec = duration_cast<seconds>(stop - start);
            std::cout << "seconds: " << duration_sec.count() << std::endl;
            sum_values += duration.count();
        
    }
    std::cout << "average microseconds: " << sum_values/num_rounds << std::endl;

    return 0;
}