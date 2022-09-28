// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <iostream>
#include <cstdlib>
#include <algorithm>
#include <iomanip>
#include <fstream>
#include <map>
#include <random>
#include <chrono>

#include <poplar/Engine.hpp>
#include <poplar/IPUModel.hpp>
#include <poputil/TileMapping.hpp>
#include <poplar/DeviceManager.hpp>
#include <poplar/Program.hpp>
#include <poplar/Type.hpp>

#include "number_with_commas.h"
#include "count_type.h"
auto CountVertex = ::poplar::equivalent_device_type<Count>().value;

using ::std::map;
using ::std::vector;
using ::std::string;
using ::std::optional;

using ::poplar::OptionFlags;
using ::poplar::Tensor;
using ::poplar::Graph;
using ::poplar::Engine;
using ::poplar::Device;
using ::poplar::DeviceManager;
using ::poplar::TargetType;
using ::poplar::program::Program;
using ::poplar::program::Sequence;
using ::poplar::program::Copy;
using ::poplar::program::Repeat;
using ::poplar::program::Execute;

static const auto MAX_TENSOR_SIZE = 55000000ul;



// Find hardware IPU
auto getIpuDevice(const unsigned int numIpus = 1) -> optional<Device> {
    DeviceManager manager = DeviceManager::createDeviceManager();
    optional<Device> device = std::nullopt;
    for (auto &d : manager.getDevices(TargetType::IPU, numIpus)) {
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
auto getIpuModel(const unsigned int numIpus = 1, int tilesPerIpu = 2) -> optional<Device> {
    optional<Device> device = std::nullopt;
    poplar::IPUModel ipuModel;
    ipuModel.numIPUs = numIpus;
    ipuModel.tilesPerIPU = tilesPerIpu;
    device = ipuModel.createDevice();
    return device;
}

auto createGraphAndAddCodelets(const optional<Device> &device) -> Graph {
    auto graph = poplar::Graph(device->getTarget());
#ifdef SIMULATED_IPU
    graph.addCodelets({"../pi_vertex.cpp"}, "-O3 -DSIMULATED_IPU -I..");
#else
    graph.addCodelets({"../pi_vertex.cpp"}, "-O3 -I..");
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


int main(int argc, char *argv[]) {
    struct pi_options {
        unsigned long iterations = 196608000000; // Match GPU calculation
//unsigned long iterations = 30000000; // Test calculation
        unsigned int num_ipus = 1;
        int precision = 10;
    } options;

    auto iterations = options.iterations;

    std::cout << "STEP 1: Connecting to an IPU device" << std::endl;
#ifdef SIMULATED_IPU
    auto device = getIpuModel(options.num_ipus);  // Simulated IPU
#else
    auto device = getIpuDevice(options.num_ipus);
#endif

    if (!device.has_value()) {
        std::cerr << "Could not attach to an IPU device. Aborting" << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "STEP 2: Create graph and compile codelets" << std::endl;
    auto graph = createGraphAndAddCodelets(device);

    std::cout << "STEP 4: Define data streams" << std::endl;
    size_t numTiles = device->getTarget().getNumTiles();
    auto fromIpuStream = graph.addDeviceToHostFIFO("FROM_IPU", CountVertex, 1);

    std::cout << "STEP 3: Building the compute graph" << std::endl;
    auto counts = graph.addVariable(CountVertex, {numTiles * 6}, "counts");
     poputil::mapTensorLinearly(graph, counts);

    // Each tile has 6 hardware worker threads that are scheduled in a round-robin (barrel) schedule
    // with one instruction being executed per context in turn
    const auto NumElemsPerTile = iterations / (numTiles * 6);
    auto cs = graph.addComputeSet("loopBody");
    std::cout << "numTiles = " << numTiles << std::endl;
    for (auto tileNum = 0u; tileNum < numTiles; tileNum++) {
        const auto sliceStart = tileNum * 6;
        const auto sliceEnd = (tileNum + 1) * 6; 

        auto v = graph.addVertex(cs, "PiVertex", {
                {"hits", counts.slice(sliceStart, sliceEnd)}
        });
        graph.setInitialValue(v["iterations"], NumElemsPerTile);
        graph.setPerfEstimate(v, 10); // Ideally you'd get this as right as possible
        graph.setTileMapping(v, tileNum);
    }


    // Create a compute set to reduce the hits from each tile to a single sum
    auto sum_counts = graph.addVariable(CountVertex, {1}, "sum_counts");
    auto reduceCS = graph.addComputeSet("reduceCS");

    // Reduce the output set. Just use one tile.
    for (unsigned tile = 0; tile < 1; ++tile) {
        auto v = graph.addVertex(reduceCS, "ReduceVertex",
                                 {{"in", counts}, {"out", sum_counts[tile]}});
        graph.setTileMapping(v, tile);
        graph.setTileMapping(sum_counts, tile);
        graph.setPerfEstimate(v, 5);
    }

    // The program consists of executing the simulation followed by
    // summing all the counts with a reduction

    Sequence prog({Execute(cs), Execute(reduceCS), Copy(sum_counts, fromIpuStream)});


    std::cout << "STEP 5: Create engine and compile graph" << std::endl;
// #define IPU_PROFILE
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
        
    std::cout << "STEP 6: Load compiled graph onto the IPU tiles" << std::endl;
    engine.load(*device);
#ifdef IPU_PROFILE
    engine.enableExecutionProfiling();
#endif

    std::cout << "STEP 7: Attach data streams" << std::endl;
    auto results = std::vector<Count>(1);
    engine.connectStream("FROM_IPU", results.data(), results.data() + results.size());

    std::cout << "STEP 8: Run programs" << std::endl;

    auto hits = 0ull;
    const int LOOPS = 10;
    auto start = std::chrono::steady_clock::now();
    for (size_t l = 0; l < LOOPS; ++l) {
        engine.run(0, "main"); // Main program
        for (size_t i = 0; i < results.size(); i++) {
            hits += results[i];
        }
    }
    auto stop = std::chrono::steady_clock::now();

    auto duration_ms = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
    double duration_s = duration_ms / pow(10, 6);

    double mc_pi = 4. * hits/(iterations*LOOPS);
    double pi_err = mc_pi - M_PI;

    std::cout << "Results size: " << results.size() << std::endl;
    std::cout << "STEP 9: Capture debug and profile info" << std::endl;
    serializeGraph(graph);
    captureProfileInfo(engine);
    engine.printProfileSummary(std::cout,
                               OptionFlags{{"showExecutionSteps", "false"}});
    std::cout << std::endl;

    typedef std::numeric_limits<double> DblLim;
    std::cout.precision(DblLim::max_digits10);

    std::cout << "chunk_size = " << numTiles * 6 << ", repeats = " << numberFormatWithCommas(iterations / numTiles * 6) << std::endl;
    std::cout << "tests = " << numberFormatWithCommas(iterations*LOOPS) << " took " << numberFormatWithCommas(duration_ms) << " microseconds";
    std::cout << ", or " << duration_s << " seconds" << std::endl;
    std::cout << "PI  ~= " <<  mc_pi << std::endl;
    std::cout << "M_PI = " <<  M_PI << std::endl;
    std::cout << "PI err ~= " <<  pi_err << std::endl;

    return EXIT_SUCCESS;
}
