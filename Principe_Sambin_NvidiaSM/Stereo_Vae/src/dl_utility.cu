#include "dl_utility.h"

std::vector<float> fromImgToTensor(cv::Mat img)
{
    int h = img.size().height;
    int w = img.size().width;
    // 1. Normalize. 
    img.convertTo(img, CV_32F, 1.0, 0.0);
    // 2. Convert HWC -> CHW.
    cv::Mat res = img.reshape(1, w * h).t();
    // 3 Compute the maximum 
    double min_v;
    double max_v;
    cv::minMaxLoc(img, &min_v, &max_v);
    // 4. Scale.
    res /= max_v;
   return std::vector<float>(res.ptr<float>(0), res.ptr<float>(0) + w * h * 1);
}

std::vector<float> createOutputTensor(Dims size)
{
    std::vector<float> o_tensor(size.d[2]*size.d[3]);
    return o_tensor;
}

int getBindingParameterIndex(std::unique_ptr<ICudaEngine, InferDeleter>& engine, char const * name)
{
    return engine->getBindingIndex(name);
}

ICudaEngine* createCudaEngine(std::string const& onnxModelPath, int batchSize, Logger gLogger)
{
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH); 
    std::unique_ptr<IBuilder, InferDeleter> builder{createInferBuilder(gLogger)};
    std::unique_ptr<INetworkDefinition, InferDeleter> network{builder->createNetworkV2(explicitBatch)};
    std::unique_ptr<nvonnxparser::IParser, InferDeleter> parser{nvonnxparser::createParser(*network, gLogger)};
    std::unique_ptr<nvinfer1::IBuilderConfig, InferDeleter> config{builder->createBuilderConfig()};

    if (!parser->parseFromFile(onnxModelPath.c_str(), static_cast<int>(ILogger::Severity::kINFO)))
    {
        std::cout << "ERROR: could not parse input engine." << std::endl;
        return nullptr;
    }
    // use FP16 mode if possible
    if (builder->platformHasFastFp16()){
       config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }
    // Define the maximum amount of memory that TensorRT can allocate
    constexpr size_t MAX_WORKSPACE_SIZE = 1ULL << 30;
    // set builder flag 
    config->setMaxWorkspaceSize(MAX_WORKSPACE_SIZE);
    // Define the maximum Batch size of the model : Just 2 images per time
    builder->setMaxBatchSize(batchSize);
    //builder->buildEngineWithConfig(*network, *config);
    IHostMemory* engine_plan = builder->buildSerializedNetwork(*network, *config);
    IRuntime* runtime = createInferRuntime(gLogger);
    return  runtime->deserializeCudaEngine(engine_plan->data() , engine_plan ->size());// build and return TensorRT engine
}

ICudaEngine* getCudaEngine(std::string const& onnxModelPath, int batchSize, Logger gLogger)
{
    ICudaEngine* engine{nullptr};

    std::ifstream trt_plan(onnxModelPath, std::ios::binary); //  readBuffer(enginePath);   
    if (trt_plan.good())
    {
        // try to deserialize engine
        std::stringstream buffer;
        buffer << trt_plan.rdbuf();
        buffer.seekg(0, buffer.beg);
        const auto& model_final = buffer.str(); 
        std::unique_ptr<IRuntime, InferDeleter> runtime{createInferRuntime(gLogger)};
        engine = runtime->deserializeCudaEngine(model_final.c_str(), model_final.size()); 
    }

    if (!engine)
    {
        // Fallback to creating engine from scratch
        engine = createCudaEngine(onnxModelPath, batchSize,gLogger);

        if (engine)
        {
            std::unique_ptr<IHostMemory, InferDeleter> engine_plan{engine->serialize()};
            // try to save engine for future uses
            std::ofstream trt_plan_out( onnxModelPath, std::ios::binary);
            trt_plan_out.write((const char*)engine_plan->data(), engine_plan->size());
        }
    }
    return engine;
}

std::tuple<double, double> launchInference(std::unique_ptr<IExecutionContext, InferDeleter>& context, cudaStream_t stream, std::vector<float>& inputTensor ,std::vector<float>& outputTensor, void** bindings, int batchSize, int out_idx,   int in_idx_disp)
{
    // Measure time it takes to copy input to GPU, run inference and move output back to CPU
    cudaEvent_t start;
    cudaEvent_t end;
    float elapsedTime;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start, 0);
    cudaMemcpyAsync(bindings[in_idx_disp], inputTensor.data(), inputTensor.size() * sizeof(float), cudaMemcpyHostToDevice, 0);
    context->enqueue(batchSize, bindings, 0, nullptr);
    CUDA_CHECK(cudaMemcpyAsync(outputTensor.data(), bindings[out_idx], outputTensor.size() * sizeof(float), cudaMemcpyDeviceToHost, 0));
    cudaEventRecord(end, 0);
    // wait until the work is finished
    cudaStreamSynchronize(0);
    cudaEventElapsedTime(&elapsedTime, start, end);
    const double fps = 1e6 / elapsedTime;
    //std::cout <<  "Test StereoVae model Execution Time no Multiplication  :  " << elapsedTime << "[ms]" << std::endl;
    std::cout << "Inference batch size " << batchSize << " StereoVae model Execution Time :  " << 1e-3 * elapsedTime << "[ms]  " << fps << "  [fps]" << std::endl;
    std::tuple<double, double> time(elapsedTime, fps);
    return time;
}

void doInference(std::unique_ptr<IExecutionContext, InferDeleter>& context, cudaStream_t stream, std::vector<float>& inputTensor , std::vector<float>& outputTensor, void** bindings, int batchSize, int out_idx,  int in_idx_disp)
{
    cudaEvent_t start;
    cudaEvent_t end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    double totalTime = 0.0;
    // Number of times we run inference to calculate average time
    int ITERATIONS = 10;
 

    for (int i = 0; i < ITERATIONS; ++i)
    {
        float elapsedTime;

        // Measure time it takes to copy input to GPU, run inference and move output back to CPU
        cudaEventRecord(start, stream);
        launchInference(context, stream, inputTensor, outputTensor, bindings, batchSize, out_idx, in_idx_disp);
        cudaEventRecord(end, stream);

        // wait until the work is finished
        cudaStreamSynchronize(stream);
        cudaEventElapsedTime(&elapsedTime, start, end);

        totalTime += elapsedTime;
    }

    std::cout << "Inference batch size " << batchSize << " average over " << ITERATIONS << " runs is " << totalTime / ITERATIONS << "ms" << std::endl;
}
