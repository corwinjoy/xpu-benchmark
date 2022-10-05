// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include "American_vertex.h"
#include <poplar/Vertex.hpp>

using namespace poplar;

////////////////////////////////////////////////////////////////////////////////
// Vertex to price option
////////////////////////////////////////////////////////////////////////////////
class AmericanVertex : public Vertex {
public:
    Input<float> StockPrice, OptionStrike, OptionYears;
    Output<float> CallResult;
    float RiskFree, Volatility;

    bool compute() {
        float cr;

        AmerBodyIPU(&cr, StockPrice,
                            OptionStrike, OptionYears, RiskFree,
                            Volatility);

        *CallResult = cr;

        return true;
    }
};