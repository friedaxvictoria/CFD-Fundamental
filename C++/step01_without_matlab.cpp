#ifdef MATPLOTLIB
#include "matplotlibcpp.h"
#endif
#include <cmath>
#include <vector>
#include <algorithm>
#include <iostream>
#include <chrono>
#include <omp.h>
using namespace std::chrono;

#ifdef MATPLOTLIB
namespace plt = matplotlibcpp;
#endif

////////////////////////////////////////////////////////////
// Step 1: 1D Linear Convection
////////////////////////////////////////////////////////////

int main() {
    // Simulation parameters
    //const int X = 40;
    //const int T = 41;
    const int X = 900000;                    // Number of spatial points
    const int T = 100000;                    // Number of time steps
    const int c = 1;                     // Wave speed

    const double dx = 2.0 / (X - 1);     // Spatial step size
    const double dt = 0.025;             // Time step size

    // Initialize spatial grid and initial condition
    //std::vector<double> x(X), u(X), un(X);
    double x[X], u[X], un[X];
    double testun[X], testu[X];

    int sum_values = 0;
    int num_rounds = 10;

    for (int round = 0; round < num_rounds; round++) {

        auto start = high_resolution_clock::now();

            //not good with simd bc of implied if statement? should you split into two loops?
            #pragma omp parallel for simd
            for (int i = 0; i < X; i++) {
                x[i] = (5.0 * i) / (X - 1);
            }

            #pragma omp parallel for
            for (int i = 0; i < X; i++) {
                u[i] = (x[i] >= 0.5 && x[i] <= 1) ? 2 : 1;
            }


        //std::copy(std::begin(testu), std::end(testu), std::begin(u));

        #ifdef MATPLOTLIB
            plt::ion();
            plt::Plot plot;
        #endif
            // Time-stepping loop

            for (int n = 0; n < T; n++) {
                std::copy(std::begin(u), std::end(u), std::begin(un));
                //un = u;
                #pragma omp parallel for simd
                for (int i = 1; i < X; i++) {
                    u[i] = un[i] - c * (un[i] - un[i - 1]) * dt / dx;
                }


        #ifdef MATPLOTLIB
                plot.update(x, u);
                plt::xlim(0, 2);
                plt::ylim(0.5, 2.5);
                plt::pause(0.1);
        #endif
            }

            /*
            for (int n = 0; n < T; n++) {
                std::copy(std::begin(testu), std::end(testu), std::begin(testun));
                for (int i = 1; i < X; i++) {
                    testu[i] = testun[i] - c * (testun[i] - testun[i - 1]) * dt / dx;
                }
            }

            for (int i = 0; i < X; i++) {
                std::cout << u[i] << std::endl;
                std::cout << testu[i] << std::endl;
                if (u[i] != testu[i])
                    std::cout << "u is unequal" << std::endl;
            }
             */

            auto stop = high_resolution_clock::now();
            auto duration = duration_cast<microseconds>(stop - start);
            std::cout << "microseconds: " << duration.count() << std::endl;
            auto duration_sec = duration_cast<seconds>(stop - start);
            std::cout << "seconds: " << duration_sec.count() << std::endl;
            sum_values += duration.count();
        #ifdef MATPLOTLIB
            plt::show();
        #endif
        }
    std::cout << "average microseconds: " << sum_values/num_rounds << std::endl;

    return 0;
}