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

/*
 * This sample evaluates fair call and put prices for a
 * given set of European options by Black-Scholes formula.
 * See supplied whitepaper for more explanations.
 */

#include <iostream>
#include <cstdlib>
#include <algorithm>
#include <iomanip>
#include <fstream>
#include <map>
#include <random>
#include <chrono>
#include <optional>

#include <poplar/Engine.hpp>
#include <poplar/Graph.hpp>
#include <poplar/IPUModel.hpp>
#include <poputil/TileMapping.hpp>
#include <poplar/DeviceManager.hpp>
#include <poplar/Program.hpp>
#include <poplar/Type.hpp>

#include "number_with_commas.h"

//#define DEBUG_VERTEX
#ifdef DEBUG_VERTEX
#include "BlackScholes_vertex.h"
#endif

using ::std::optional;

using namespace poplar;
using namespace poplar::program;


////////////////////////////////////////////////////////////////////////////////
// Process an array of optN options on CPU
////////////////////////////////////////////////////////////////////////////////
extern "C" void BlackScholesCPU(float *h_CallResult, float *h_PutResult,
                                float *h_StockPrice, float *h_OptionStrike,
                                float *h_OptionYears, float Riskfree,
                                float Volatility, int optN);


////////////////////////////////////////////////////////////////////////////////
// Helper function, returning uniformly distributed
// random float in [low, high] range
////////////////////////////////////////////////////////////////////////////////
float RandFloat(float low, float high) {
    float t = (float) rand() / (float) RAND_MAX;
    return (1.0f - t) * low + t * high;
}

////////////////////////////////////////////////////////////////////////////////
// Data configuration
////////////////////////////////////////////////////////////////////////////////
//const int OPT_N = 4000000;
//const int OPT_N = 6144*32;
const int OPT_N = 1000;
const int NUM_ITERATIONS = 512;
// const int NUM_ITERATIONS = 1;
const int OPT_SZ = OPT_N * sizeof(float);
const float RISKFREE = 0.02f;
const float VOLATILITY = 0.30f;

#define DIV_UP(a, b) (((a) + (b)-1) / (b))

////////////////////////////////////////////////////////////////////////////////
// IPU Utilities
////////////////////////////////////////////////////////////////////////////////

// Find hardware IPU
auto getIpuDevice(const unsigned int numIpus = 1) -> optional <Device> {
    DeviceManager manager = DeviceManager::createDeviceManager();
    optional <Device> device = std::nullopt;
    for (auto &d: manager.getDevices(TargetType::IPU, numIpus)) {
        std::cout << "Trying to attach to IPU " << d.getId();
        if (d.attach()) {
            std::cout << " - attached" << std::endl;
            device = {std::move(d)};
            break;
        } else {
            std::cout << std::endl << "Error attaching to device" << std::endl;
        }
    }
    return device;
}

// Return simulated IPU
auto getIpuModel(const unsigned int numIpus = 1, int tilesPerIpu = 2) -> optional <Device> {
    optional <Device> device = std::nullopt;
    poplar::IPUModel ipuModel;
    ipuModel.numIPUs = numIpus;
    ipuModel.tilesPerIPU = tilesPerIpu;
    device = ipuModel.createDevice();
    return device;
}

auto createGraphAndAddCodelets(const optional <Device> &device) -> Graph {
    auto graph = poplar::Graph(device->getTarget());
#ifdef SIMULATED_IPU
    graph.addCodelets({"../BlackScholes_vertex.cpp"}, "-O3 -DSIMULATED_IPU -I..");
#else
    graph.addCodelets({"../BlackScholes_vertex.cpp"}, "-O3 -I..");
#endif
    return graph;
}

auto serializeGraph(const Graph &graph) {
    std::ofstream graphSerOfs;
    graphSerOfs.open("serialized_graph.capnp", std::ofstream::out | std::ofstream::trunc);

    graph.serialize(graphSerOfs, poplar::SerializationFormat::Binary);
}

auto captureProfileInfo(Engine &engine) {
    std::ofstream graphOfs;
    graphOfs.open("graph.json", std::ofstream::out | std::ofstream::trunc);

    std::ofstream executionOfs;
    executionOfs.open("execution.json", std::ofstream::out | std::ofstream::trunc);
}

////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
    // Start logs
    printf("[%s] - Starting...\n", argv[0]);

    //'h_' prefix - CPU (host) memory space
    float
    // Results calculated by CPU for reference
    *h_CallResultCPU, *h_PutResultCPU,
    // CPU instance of input data
    *h_StockPrice, *h_OptionStrike, *h_OptionYears;

    double delta, ref, sum_delta, sum_ref, max_delta, L1norm;
    int i;

    std::cout << "STEP 1: Connecting to an IPU device" << std::endl;
#ifdef SIMULATED_IPU
    auto device = getIpuModel(1, OPT_N);  // Simulated IPU
#else
    auto device = getIpuDevice(1);
#endif

    if (!device.has_value()) {
        std::cerr << "Could not attach to an IPU device. Aborting" << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "STEP 2: Create graph and compile codelets" << std::endl;
    auto graph = createGraphAndAddCodelets(device);

    printf("Initializing data...\n");
    printf("...allocating CPU memory for options.\n");
    h_CallResultCPU = (float *) malloc(OPT_SZ);
    h_PutResultCPU = (float *) malloc(OPT_SZ);
    h_StockPrice = (float *) malloc(OPT_SZ);
    h_OptionStrike = (float *) malloc(OPT_SZ);
    h_OptionYears = (float *) malloc(OPT_SZ);

    // Generate option inputs
    srand(5347);
    for (i = 0; i < OPT_N; i++) {
        h_CallResultCPU[i] = 0.0f;
        h_PutResultCPU[i] = -1.0f;
        h_StockPrice[i] = RandFloat(5.0f, 30.0f);
        h_OptionStrike[i] = RandFloat(1.0f, 100.0f);
        h_OptionYears[i] = RandFloat(0.25f, 10.0f);
    }


#ifdef DEBUG_VERTEX
    {
        float cr, pr;
        BlackScholesBodyGPU(&cr, &pr, h_StockPrice[0],
                            h_OptionStrike[0], h_OptionYears[0], RISKFREE,
                            VOLATILITY);
        std::cout << cr << std::endl;
        BlackScholesCPU(h_CallResultCPU, h_PutResultCPU, h_StockPrice, h_OptionStrike,
                        h_OptionYears, RISKFREE, VOLATILITY, 1);
        std::cout << h_CallResultCPU[0] << std::endl;
    };
#endif

    // Initialize parameter vectors
    std::vector<float> CallResult(OPT_N);
    std::vector<float> PutResult(OPT_N);
    std::vector<float> StockPrice(OPT_N);
    std::vector<float> OptionStrike(OPT_N);
    std::vector<float> OptionYears(OPT_N);

    // Copy CPU data to vectors
    CallResult.assign(h_CallResultCPU, h_CallResultCPU + OPT_N);
    PutResult.assign(h_PutResultCPU, h_PutResultCPU + OPT_N);
    StockPrice.assign(h_StockPrice, h_StockPrice + OPT_N);
    OptionStrike.assign(h_OptionStrike, h_OptionStrike + OPT_N);
    OptionYears.assign(h_OptionYears, h_OptionYears + OPT_N);

    // Create tensors in the graph.
    Tensor CallResult_t = graph.addVariable(FLOAT, {OPT_N}, "CallResult");
    poputil::mapTensorLinearly(graph, CallResult_t);

    Tensor PutResult_t = graph.addVariable(FLOAT, {OPT_N}, "PutResult");
    poputil::mapTensorLinearly(graph, PutResult_t);

    Tensor StockPrice_t = graph.addVariable(FLOAT, {OPT_N}, "StockPrice");
    poputil::mapTensorLinearly(graph, StockPrice_t);

    Tensor OptionStrike_t = graph.addVariable(FLOAT, {OPT_N}, "OptionStrike");
    poputil::mapTensorLinearly(graph, OptionStrike_t);

    Tensor OptionYears_t = graph.addVariable(FLOAT, {OPT_N}, "OptionYears");
    poputil::mapTensorLinearly(graph, OptionYears_t);

    // Map input values
    auto inStockPrice = graph.addHostToDeviceFIFO("TO_IPU_STOCK_PRICE", FLOAT, OPT_N);
    auto inOptionStrike = graph.addHostToDeviceFIFO("TO_IPU_OPTION_STRIKE", FLOAT, OPT_N);
    auto inOptionYears = graph.addHostToDeviceFIFO("TO_IPU_OPTION_YEARS", FLOAT, OPT_N);

    // Map return values
    auto fromIpuCallResult = graph.addDeviceToHostFIFO("FROM_IPU_CALL_RESULT", FLOAT, OPT_N);
    auto fromIpuPutResult = graph.addDeviceToHostFIFO("FROM_IPU_PUT_RESULT", FLOAT, OPT_N);

    // Create a control program that is a sequence of steps
    Sequence prog;

    ComputeSet computeSet = graph.addComputeSet("computeSet");
    for (i = 0; i < OPT_N; ++i) {
        VertexRef vtx = graph.addVertex(computeSet, "BlackScholesVertex");
        graph.connect(vtx["CallResult"], CallResult_t[i]);
        graph.connect(vtx["PutResult"], PutResult_t[i]);
        graph.connect(vtx["StockPrice"], StockPrice_t[i]);
        graph.connect(vtx["OptionStrike"], OptionStrike_t[i]);
        graph.connect(vtx["OptionYears"], OptionYears_t[i]);

        graph.setInitialValue(vtx["RiskFree"], RISKFREE);
        graph.setInitialValue(vtx["Volatility"], VOLATILITY);
        graph.setTileMapping(vtx, i);
        graph.setPerfEstimate(vtx, 200);
    }

    // Get data in
    prog.add(Copy(inStockPrice, CallResult_t));
    prog.add(Copy(inOptionStrike, OptionStrike_t));
    prog.add(Copy(inOptionYears, OptionYears_t));

    // Add step to execute the compute set
    prog.add(Execute(computeSet));

    // Get data out
    prog.add(Copy(CallResult_t, fromIpuCallResult));
    prog.add(Copy(PutResult_t, fromIpuPutResult));

#define IPU_PROFILE
#ifdef IPU_PROFILE
    auto ENGINE_OPTIONS = OptionFlags{
            {"target.saveArchive",                "archive.a"},
            {"debug.instrument",                  "true"},
            {"debug.instrumentCompute",           "true"},
            {"debug.loweredVarDumpFile",          "vars.capnp"},
            {"debug.instrumentControlFlow",       "true"},
            {"debug.computeInstrumentationLevel", "tile"},
            {"debug.outputAllSymbols",            "true"},
            {"autoReport.all",                    "true"},
            {"autoReport.outputSerializedGraph",  "true"},
            {"debug.retainDebugInformation",      "true"}
    };
#else
    auto ENGINE_OPTIONS = OptionFlags{};
#endif

    auto engine = Engine(graph, prog, ENGINE_OPTIONS);

    engine.load(*device);
#ifdef IPU_PROFILE
    engine.enableExecutionProfiling();
#endif

    engine.connectStream("TO_IPU_STOCK_PRICE", StockPrice.data());
    engine.connectStream("TO_IPU_OPTION_STRIKE", OptionStrike.data());
    engine.connectStream("TO_IPU_OPTION_YEARS", OptionYears.data());
    engine.connectStream("FROM_IPU_CALL_RESULT", CallResult.data());
    engine.connectStream("FROM_IPU_PUT_RESULT", PutResult.data());

    auto start = std::chrono::steady_clock::now();
    for (i = 0; i < NUM_ITERATIONS; i++) {
        engine.run(0);
    }
    auto stop = std::chrono::steady_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
    duration_ms /= NUM_ITERATIONS;
    double duration_s = duration_ms / pow(10, 6);

    std::cout << "Options Count = " << numberFormatWithCommas(OPT_N) << " took " << numberFormatWithCommas(duration_ms) << " microseconds";
    std::cout << ", or " << duration_s << " seconds" << std::endl;


    printf("Checking the results...\n");
    printf("...running CPU calculations.\n\n");
    // Calculate options values on CPU


    BlackScholesCPU(h_CallResultCPU, h_PutResultCPU, h_StockPrice, h_OptionStrike,
                    h_OptionYears, RISKFREE, VOLATILITY, OPT_N);

    printf("Comparing the results...\n");
    // Calculate max absolute difference and L1 distance
    // between CPU and GPU results
    sum_delta = 0;
    sum_ref = 0;
    max_delta = 0;

    for (i = 0; i < OPT_N; i++) {
        ref = h_CallResultCPU[i];
        delta = fabs(ref - CallResult[i]);

        if (delta > max_delta) {
            max_delta = delta;
        }

        sum_delta += delta;
        sum_ref += fabs(ref);
    }

    L1norm = sum_delta / sum_ref;
    printf("L1 norm: %E\n", L1norm);
    printf("Max absolute error: %E\n\n", max_delta);

    printf("Shutting down...\n");

    printf("...releasing CPU memory.\n");
    free(h_OptionYears);
    free(h_OptionStrike);
    free(h_StockPrice);
    free(h_PutResultCPU);
    free(h_CallResultCPU);
    printf("Shutdown done.\n");

    printf("\n[BlackScholes] - Test Summary\n");

    if (L1norm > 1e-6) {
        printf("Test failed!\n");
        exit(EXIT_FAILURE);
    }

    printf("Test passed\n");
    exit(EXIT_SUCCESS);
}
