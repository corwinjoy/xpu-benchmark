//
// Created by cjoy on 9/30/22.
//

#pragma once
#include <climits>
#include <math.h>

// Function for binomial tree
inline float Binomial(float S, float X, float R, float Q, float V, float T, char PutCall, char OpStyle) {
    const int STEPS = 63;
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
inline void AmerBodyIPU(float *CallResult,
                                   float S,  // Stock price
                                   float X,  // Option strike
                                   float T,  // Option years
                                   float R,  // Riskless rate
                                   float V  // Volatility rate
) {
    const char PutCall = 'C';
    const char OpStyle = 'A';

    *CallResult = Binomial(S, X, R, 0.0, V, T, PutCall, OpStyle);
}

