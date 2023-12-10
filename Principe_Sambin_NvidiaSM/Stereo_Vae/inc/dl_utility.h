#ifndef __DL_UTILITY_H__
#define __DL_UTILITY_H__

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <fstream>
#include "host_utility.h"
#include <opencv2/core.hpp>

using namespace nvinfer1;

/*class Logger : public nvinfer1::ILogger
{
public:
    void log(nvinfer1::ILogger::Severity severity, const char* msg) override
    {
        // Skip info (verbose) messages.
        // if (severity == Severity::kINFO)
        //     return;

        switch (severity)
        {
            case Severity::kINTERNAL_ERROR: std::cerr << "TRT INTERNAL_ERROR: "; break;
            case Severity::kERROR:          std::cerr << "TRT ERROR: "; break;
            case Severity::kWARNING:        std::cerr << "TRT WARNING: "; break;
            case Severity::kINFO:           std::cerr << "TRT INFO: "; break;
            default:                        std::cerr << "TRT UNKNOWN: "; break;
        }
        std::cerr << msg << std::endl;
    }
};*/
class Logger : public ILogger
{
public:
    void log(Severity severity, const char* msg) noexcept override {
        // remove this 'if' if you need more logged info
        if ((severity == Severity::kERROR) || (severity == Severity::kINTERNAL_ERROR)) {
            std::cout << msg << std::endl;
        }
    }
};

struct InferDeleter
{
    template <typename T>
    void operator()(T* obj) const
    {
        delete obj;
        //obj->destroy();

    }
};

std::vector<float> fromImgToTensor(cv::Mat img);

std::vector<float> createOutputTensor(Dims size);

// Retrieve the Binding Index of a given Input or Output Parameters of the Network
int getBindingParameterIndex(std::unique_ptr<ICudaEngine, InferDeleter>& engine, char const * name);

// Create the engine for the ONNX model inference
ICudaEngine* createCudaEngine(std::string const& onnxModelPath, int batchSize, Logger gLogger);

// Load the evaluated engine of the model selected
ICudaEngine* getCudaEngine(std::string const& onnxModelPath, int batchSize, Logger gLogger);

// Run Inference of the TensorRT model loaded
std::tuple<double, double> launchInference(std::unique_ptr<IExecutionContext, InferDeleter>& context, cudaStream_t stream, std::vector<float>& inputTensor, std::vector<float>& outputTensor, void** bindings, int batchSize, int out_idx,  int in_idx_disp);

// Apply Inference on the model loaded in a certain number of iteration for averaging the throughput and time of execution
void doInference(std::unique_ptr<IExecutionContext, InferDeleter>& context, cudaStream_t stream, std::vector<float>& inputTensor, std::vector<float>& outputTensor, void** bindings, int batchSize, int out_idx,  int in_idx_disp);

#endif // !__DL_UTILITY_H__
