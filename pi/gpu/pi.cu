// Approximation of Pi using a simple, and not optimized, CUDA program
// Copyleft Alessandro Re
//
// GCC 6.x not supported by CUDA 8, I used compat version
//
// nvcc -std=c++11 -ccbin=gcc5 pigreco.cu -c
// g++5 pigreco.o -lcudart -L/usr/local/cuda/lib64 -o pigreco
//
// This code is basically equivalent to the following Python code:
//
// def pigreco(NUM):
//     from random import random as rand
//     def sqrad():
//         x, y = rand(), rand()
//         return x*x + y*y
//     return 4 * sum(1 - int(test()) for _ in range(NUM)) / NUM
//
// Python version takes, on this machine, 3.5 seconds to compute 10M tests
// CUDA version takes, on this machine, 1.6 seconds to compute 20.48G tests
//
#include <iostream>
#include <limits>
#include <numeric>
#include <cuda.h>
#include <curand_kernel.h>
#include <math.h>

#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>


#include "timer.h"
#include "number_with_commas.h"

using std::cout;
using std::endl;

typedef uint64_t Count;
typedef std::numeric_limits<double> DblLim;

const Count WARP_SIZE = 32; // Warp size 32
const Count NBLOCKS = 6144; // Number of total cuda cores on my GPU
const Count ITERATIONS = 1000000; // Number of points to generate (each thread)
const Count LOOPS = 10; // Number of sims to run on GPUs


/*
 * Results:
 * GeForce RTX 3080.
 * NBLOCKS=6144, ITERATIONS=1,000,000, LOOPS=10.
 * Approximated PI using 1,966,080,000,000 random tests
 * Time taken 6.279254 seconds
 * PI err ~= -8.7864675935023229e-07
 */

// This kernel is
__global__ void picount(Count *totals) {
    // Define some shared memory: all threads in this block
    __shared__ Count counter[WARP_SIZE];

    // Unique ID of the thread
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Initialize RNG
    curandState_t rng;
    curand_init(clock64(), tid, 0, &rng);

    // Initialize the counter
    counter[threadIdx.x] = 0;

    // Computation loop
    for (int i = 0; i < ITERATIONS; i++) {
        float x = curand_uniform(&rng); // Random x position in [0,1]
        float y = curand_uniform(&rng); // Random y position in [0,1]
#if 1
        // Use int cast to avoid branching
        // Time taken 0.647842 seconds
        counter[threadIdx.x] += 1 - int(x * x + y * y); // Hit test
#else
        // Slower branching logic for comparison.  Time taken 0.678691 seconds with 1m iterations
        if (x*x+y*y < 1) {
            counter[threadIdx.x] += 1;
        }
#endif
    }

    // The first thread in *every block* should sum the results
    if (threadIdx.x == 0) {
        // Reset count for this block
        totals[blockIdx.x] = 0;
        // Accumulate results
        for (int i = 0; i < WARP_SIZE; i++) {
            totals[blockIdx.x] += counter[i];
        }
    }
}

int main(int argc, char **argv) {
    int numDev;
    cudaGetDeviceCount(&numDev);
    if (numDev < 1) {
        cout << "CUDA device missing! Do you need to set it up?\n";
        return 1;
    }
    cout << "Starting simulation with " << NBLOCKS << " blocks, " << WARP_SIZE << " threads, and " <<
         numberFormatWithCommas(ITERATIONS) << " iterations\n";

    // Allocate device memory to store the counters
    thrust::device_vector<Count> Out(NBLOCKS);
    Count *dOut = thrust::raw_pointer_cast(Out.data());

    uint64_t timer;
    start_time(timer);

    Count total = 0;

    for (Count i=0; i<LOOPS; ++i) {
        // Launch kernel
        picount<<<NBLOCKS, WARP_SIZE>>>(dOut);

        // Sum values on GPU
        total += thrust::reduce(Out.begin(), Out.end(), Count(0));
    }

    stop_time(timer);


    Count tests = NBLOCKS * ITERATIONS * WARP_SIZE * LOOPS;
    double mc_pi = 4.0 * (double)total/(double)tests;
    double pi_err = mc_pi - M_PI;

    cout << "Approximated PI using " << numberFormatWithCommas(tests) << " random tests\n";

    // Set maximum precision for decimal printing
    cout.precision(DblLim::max_digits10);
    cout << "PI  ~= " <<  mc_pi << endl;
    cout << "M_PI = " <<  M_PI << endl;
    cout << "PI err ~= " <<  pi_err << endl;

    return 0;
}