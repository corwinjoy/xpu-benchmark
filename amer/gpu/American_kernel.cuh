/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/* American option code adapted from:
 * Optimizing Cox, Ross and Rubinstein in C++
 * https://sites.google.com/view/vinegarhill-financelabs/binomial-lattice-framework/cox-ross-and-rubinstein/optimizing-cox-ross-and-rubinstein
 * Example code 2:
 * C++ code based on Espen Haug and Broadie and Detemple (1996) Dynamic Memory design.
 */

#include <math.h>

// Function for binomial tree
__device__ inline float Binomial(float S, float X, float R, float Q, float V, float T, char PutCall, char OpStyle) {
    const unsigned int STEPS = 63;
    float OptionValue[STEPS+1];
    int i, j;
    float dt, u, d, p;
    int z;

    // Quantities for the tree
    dt = T / STEPS;
    u = exp(V * sqrt(dt));
    d = 1.0 / u;
    p = (exp((R - Q) * dt) - d) / (u - d);

    if (PutCall == 'C')
    {
        z = 1;
    }
    else if (PutCall == 'P')
    {
        z = -1;
    }

    // Initialize terminal exercise values
    for (i = 0; i <= STEPS; i++) {
        OptionValue[i] = fmaxf(z*(S*pow(u, i)*pow(d, STEPS - i) - X), 0.0);
    }
    
    // Backward recursion through the tree
    for (j = STEPS - 1; j >= 0; j--) {
        for (i = 0; i <= j; i++) {
            if (OpStyle == 'E')
                OptionValue[i] = exp(-R * dt) * (p * (OptionValue[i + 1]) + (1.0 - p) * (OptionValue[i]));
            else {
                float CurrentExercise = z*(S*pow(u, i)*pow(d, j - i) - X);
                float FutureExercise = exp(-R * dt) * (p * (OptionValue[i + 1]) + (1.0 - p) * (OptionValue[i]));
                OptionValue[i] = fmaxf(CurrentExercise, FutureExercise);
            }
        }
    }

    // Return the option price
    return OptionValue[0];
}

////////////////////////////////////////////////////////////////////////////////
// American Call Option, priced via Binomial Tree
////////////////////////////////////////////////////////////////////////////////
__device__ inline void AmerBodyGPU(float &CallResult,
                                   float S,  // Stock price
                                   float X,  // Option strike
                                   float T,  // Option years
                                   float R,  // Riskless rate
                                   float V  // Volatility rate
                                   ) {
  const char PutCall = 'C';
  const char OpStyle = 'A';

  CallResult = Binomial(S, X, R, 0.0, V, T, PutCall, OpStyle);
}

////////////////////////////////////////////////////////////////////////////////
// Process an array of optN options on GPU
////////////////////////////////////////////////////////////////////////////////
__global__
    void AmerGPU(float *d_CallResult,
                 float *d_StockPrice,
                 float *d_OptionStrike,
                 float *d_OptionYears, float Riskfree,
                 float Volatility, int optN) {
  const int tid = blockDim.x * blockIdx.x + threadIdx.x; //Thread index
  if (tid < optN) {
      float callResult;
      AmerBodyGPU(callResult, d_StockPrice[tid],
                  d_OptionStrike[tid], d_OptionYears[tid], Riskfree,
                  Volatility);
      d_CallResult[tid] = callResult;
  }
}
