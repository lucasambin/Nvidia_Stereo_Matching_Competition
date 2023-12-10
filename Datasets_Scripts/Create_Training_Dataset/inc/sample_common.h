#ifndef __SAMPLE_COMMON_H__
#define __SAMPLE_COMMON_H__

#include <opencv2/core.hpp>
#include <cuda_runtime.h>

#define ASSERT_MSG(expr, msg) \
if (!(expr)) { \
	std::cerr << msg << std::endl; \
	std::exit(EXIT_FAILURE); \
} \

struct device_buffer
{
	device_buffer() : data(nullptr), size(0) {}
	device_buffer(size_t count) : device_buffer() { allocate(count); }
	~device_buffer() { cudaFree(data); }

	void allocate(size_t count) { cudaMalloc(&data, count); size = count; }
	void upload(const void* h_data) { cudaMemcpy(data, h_data, size, cudaMemcpyHostToDevice); }
	void download(void* h_data) { cudaMemcpy(h_data, data, size, cudaMemcpyDeviceToHost); }

	void* data;
	size_t size;
};

void colorize_disparity(const cv::Mat& src, cv::Mat& dst, int disp_size, cv::InputArray mask = cv::noArray());

#endif // !__SAMPLE_COMMON_H__
