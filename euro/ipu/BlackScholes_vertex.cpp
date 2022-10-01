// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include "BlackScholes_vertex.h"
#include <poplar/Vertex.hpp>

using namespace poplar;

////////////////////////////////////////////////////////////////////////////////
// Vertex to price option
////////////////////////////////////////////////////////////////////////////////
class BlackScholesVertex : public Vertex {
public:
    Input<float> StockPrice, OptionStrike, OptionYears;
    Output<float> CallResult, PutResult;
    float RiskFree, Volatility;

    bool compute() {
        float cr, pr;

        BlackScholesBodyGPU(cr, pr, StockPrice,
                            OptionStrike, OptionYears, RiskFree,
                            Volatility);
        *CallResult = cr;
        *PutResult = pr;
        return true;
    }
};