//
// Created by cjoy on 9/30/22.
//

#pragma once
#include <climits>
#include <math.h>
// #include <ipudef.h>

////////////////////////////////////////////////////////////////////////////////
// Polynomial approximation of cumulative normal distribution function
////////////////////////////////////////////////////////////////////////////////
inline float cndGPU(double d) {
    const double A1 = 0.31938153;
    const double A2 = -0.356563782;
    const double A3 = 1.781477937;
    const double A4 = -1.821255978;
    const double A5 = 1.330274429;
    const double RSQRT2PI = 0.39894228040143267793994605993438;

    double K = 1.0 / (1.0 + 0.2316419 * fabs(d));

    double cnd = RSQRT2PI * exp(-0.5 * d * d) *
                 (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))));

    if (d > 0.0) cnd = 1.0 - cnd;

    return cnd;
}

////////////////////////////////////////////////////////////////////////////////
// Black-Scholes formula for both call and put
////////////////////////////////////////////////////////////////////////////////
inline void BlackScholesBodyGPU(float *CallResult, float *PutResult,
                                float Sf,  // Stock price
                                float Xf,  // Option strike
                                float Tf,  // Option years
                                float Rf,  // Riskless rate
                                float Vf  // Volatility rate
) {
    double S = Sf, X = Xf, T = Tf, R = Rf, V = Vf;

    double sqrtT = sqrt(T);
    double d1 = (log(S / X) + (R + 0.5 * V * V) * T) / (V * sqrtT);
    double d2 = d1 - V * sqrtT;
    double CNDD1 = cndGPU(d1);
    double CNDD2 = cndGPU(d2);

    // Calculate Call and Put simultaneously
    double expRT = exp(-R * T);
    *CallResult = (float)(S * CNDD1 - X * expRT * CNDD2);
    *PutResult = (float)(X * expRT * (1.0 - CNDD2) - S * (1.0 - CNDD1));
}
