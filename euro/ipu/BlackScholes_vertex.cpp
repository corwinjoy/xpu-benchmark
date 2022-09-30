// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <poplar/Vertex.hpp>
#include <climits>
#include <print.h>
#include <math.h>

#include "count_type.h"

#ifdef SIMULATED_IPU
    #include <random>
    std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution(0.0,1.0);
#else
    #include <ipudef.h>
#endif

using namespace poplar;

////////////////////////////////////////////////////////////////////////////////
// Polynomial approximation of cumulative normal distribution function
////////////////////////////////////////////////////////////////////////////////
inline float cndGPU(float d) {
    const float A1 = 0.31938153f;
    const float A2 = -0.356563782f;
    const float A3 = 1.781477937f;
    const float A4 = -1.821255978f;
    const float A5 = 1.330274429f;
    const float RSQRT2PI = 0.39894228040143267793994605993438f;

    float K = 1.0f / (1.0f + 0.2316419f * fabsf(d));

    float cnd = RSQRT2PI * __expf(-0.5f * d * d) *
                (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))));

    if (d > 0) cnd = 1.0f - cnd;

    return cnd;
}

////////////////////////////////////////////////////////////////////////////////
// Black-Scholes formula for both call and put
////////////////////////////////////////////////////////////////////////////////
inline void BlackScholesBodyGPU(float &CallResult, float &PutResult,
                                           float S,  // Stock price
                                           float X,  // Option strike
                                           float T,  // Option years
                                           float R,  // Riskless rate
                                           float V  // Volatility rate
) {
    float sqrtT, expRT;
    float d1, d2, CNDD1, CNDD2;

    sqrtT = 1.0F / sqrt(T);
    d1 = (__logf(S / X) + (R + 0.5f * V * V) * T) / (V * sqrtT);
    d2 = d1 - V * sqrtT;

    CNDD1 = cndGPU(d1);
    CNDD2 = cndGPU(d2);

    // Calculate Call and Put simultaneously
    expRT = __expf(-R * T);
    CallResult = S * CNDD1 - X * expRT * CNDD2;
    PutResult = X * expRT * (1.0f - CNDD2) - S * (1.0f - CNDD1);
};

////////////////////////////////////////////////////////////////////////////////
// Vertex to price option
////////////////////////////////////////////////////////////////////////////////
class BlackScholesVertex : public Vertex {
public:
    Input<float> StockPrice, OptionStrike, OptionYears, Riskfree, Volatility;
    Output<float> CallResult, PutResult;

    void compute() {
        BlackScholesBodyGPU(*CallResult, *PutResult, StockPrice,
                            OptionStrike, OptionYears, Riskfree,
                            Volatility);
    }
};