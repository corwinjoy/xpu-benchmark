//
// Created by cjoy on 9/30/22.
//

#pragma once
#include <climits>
#include <stdint.h>
// #include <ipudef.h>

double fabs(double x) {
    if (x < 0.0) {
        return -x;
    }
    return x;
}

// Returns approximate value of e^x
// using sum of first n terms of Taylor Series
double fm_exp(double x)
{
    int n = 15;
    double sum = 1.0; // initialize sum of series

    for (int i = n - 1; i > 0; --i )
        sum = 1.0 + x * sum / i;

    return sum;
}

double bad_log(double x)
{
    double epsilon = 0.0001;
    double yn = x - 1.0; // using the first term of the taylor series as initial-value
    double yn1 = yn;
    int cnt = 0;
    do
    {
        yn = yn1;
        yn1 = yn + 2 * (x - fm_exp(yn)) / (x + fm_exp(yn));
        cnt += 1;
    } while (fabs(yn - yn1) > epsilon && cnt < 20);

    return yn1;
}

float flt_log(float y)
{
    // Algo from: https://stackoverflow.com/a/18454010
    // Accurate between (1 / scaling factor) < y < (2^32  / scaling factor). Read comments below for more info on how to extend this range

    float divisor, x, result;
    const float LN_2 = 0.69314718; //pre calculated constant used in calculations
    uint32_t log2 = 0;


    //handle if input is less than zero
    if (y <= 0)
    {
        return -1000.0f;
    }

    //scaling factor. The polynomial below is accurate when the input y>1, therefore using a scaling factor of 256 (aka 2^8) extends this to 1/256 or ~0.04. Given use of uint32_t, the input y must stay below 2^24 or 16777216 (aka 2^(32-8)), otherwise uint_y used below will overflow. Increasing the scaing factor will reduce the lower accuracy bound and also reduce the upper overflow bound. If you need the range to be wider, consider changing uint_y to a uint64_t
    const uint32_t SCALING_FACTOR = 256;
    const float LN_SCALING_FACTOR = 5.545177444; //this is the natural log of the scaling factor and needs to be precalculated

    y = y * SCALING_FACTOR;

    uint32_t uint_y = (uint32_t)y;
    while (uint_y >>= 1) // Convert the number to an integer and then find the location of the MSB. This is the integer portion of Log2(y). See: https://stackoverflow.com/a/4970859/6630230
    {
        log2++;
    }

    divisor = (float)(1 << log2);
    x = y / divisor;    // FInd the remainder value between [1.0, 2.0] then calculate the natural log of this remainder using a polynomial approximation
    result = -1.7417939 + (2.8212026 + (-1.4699568 + (0.44717955 - 0.056570851 * x) * x) * x) * x; //This polynomial approximates ln(x) between [1,2]

    result = result + ((float)log2) * LN_2 - LN_SCALING_FACTOR; // Using the log product rule Log(A) + Log(B) = Log(AB) and the log base change rule log_x(A) = log_y(A)/Log_y(x), calculate all the components in base e and then sum them: = Ln(x_remainder) + (log_2(x_integer) * ln(2)) - ln(SCALING_FACTOR)

    return result;
}

double fm_log(double x) {
    float y = flt_log((float)x);
    return (double)y;
}
double fm_sqrt(double x) {
    double half_log = 0.5 * fm_log(x);
    return fm_exp(half_log);
}


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

    double cnd = RSQRT2PI * fm_exp(-0.5 * d * d) *
                 (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))));

    if (d > 0) cnd = 1.0 - cnd;

    return cnd;
}

////////////////////////////////////////////////////////////////////////////////
// Black-Scholes formula for both call and put
////////////////////////////////////////////////////////////////////////////////
inline void BlackScholesBodyGPU(float &CallResult, float &PutResult,
                                float Sf,  // Stock price
                                float Xf,  // Option strike
                                float Tf,  // Option years
                                float Rf,  // Riskless rate
                                float Vf  // Volatility rate
) {
    double S = Sf, X = Xf, T = Tf, R = Rf, V = Vf;

    double sqrtT = fm_sqrt(T);
    double d1 = (fm_log(S / X) + (R + 0.5 * V * V) * T) / (V * sqrtT);
    double d2 = d1 - V * sqrtT;
    double CNDD1 = cndGPU(d1);
    double CNDD2 = cndGPU(d2);

    // Calculate Call and Put simultaneously
    double expRT = fm_exp(-R * T);
    CallResult = (float)(S * CNDD1 - X * expRT * CNDD2);
    PutResult = (float)(X * expRT * (1.0 - CNDD2) - S * (1.0 - CNDD1));
}
