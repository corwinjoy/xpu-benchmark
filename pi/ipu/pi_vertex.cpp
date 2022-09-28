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

// MultiVertex runs multiple (6) workers on each tile
class PiVertex : public MultiVertex {

public:
    Output<Vector<Count>> hits;
    int iterations;

    auto compute(unsigned i) -> bool {
        int count = 0;
        for (auto i = 0; i < iterations; i++) {
#ifdef SIMULATED_IPU
            float x = distribution(generator);
            float y = distribution(generator);
#else
            float x = (float)__builtin_ipu_urand32() / (float)UINT_MAX;
            float y = (float)__builtin_ipu_urand32() / (float)UINT_MAX;
#endif

            auto val = x * x + y * y;
            count +=  val < 1.f;
        }

        hits[i] = count;
        return true;
    }
};

// A vertex type to sum up a set of inputs.
class ReduceVertex : public Vertex {
public:
    Input<Vector<Count>> in;
    Output<Count> out;

    bool compute() {
        Count sum = 0;
        for (unsigned i = 0; i < in.size(); ++i) {
            sum += in[i];
        }
        *out = sum;
        return true;
    }
};
