#include "internal.h"

#include <cuda_runtime.h>

#include "constants.h"
#include "host_utility.h"

namespace
{

template<typename SRC_T, typename DST_T>
__global__ void check_consistency_kernel(DST_T* dispL, const DST_T* dispR, const SRC_T* srcL, int width, int height, int src_pitch, int dst_pitch, bool subpixel, int LR_max_diff)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= width || y >= height)
		return;

	// left-right consistency check, only on leftDisp, but could be done for rightDisp too

	SRC_T mask = srcL[y * src_pitch + x];
	DST_T org = dispL[y * dst_pitch + x];
	int d = org;
	if (subpixel) {
		d >>= sgm::StereoSGM::SUBPIXEL_SHIFT;
	}
	const int k = x - d;
	if (mask == 0 || org == sgm::INVALID_DISP || (k >= 0 && k < width && LR_max_diff >= 0 && abs(dispR[y * dst_pitch + k] - d) > LR_max_diff)) {
		// masked or left-right inconsistent pixel -> invalid
		dispL[y * dst_pitch + x] = static_cast<DST_T>(sgm::INVALID_DISP);
	}
}

} // namespace

namespace sgm
{
namespace details
{

void check_consistency(DeviceImage& dispL, const DeviceImage& dispR, const DeviceImage& srcL, bool subpixel, int LR_max_diff)
{
	SGM_ASSERT(dispL.type == SGM_16U && dispR.type == SGM_16U, "");

	const int w = srcL.cols;
	const int h = srcL.rows;

	const dim3 block(16, 16);
	const dim3 grid(divUp(w, block.x), divUp(h, block.y));

	if (srcL.type == SGM_8U) {
		using SRC_T = uint8_t;
		check_consistency_kernel<SRC_T><<<grid, block>>>(dispL.ptr<uint16_t>(), dispR.ptr<uint16_t>(),
			srcL.ptr<SRC_T>(), w, h, srcL.step, dispL.step, subpixel, LR_max_diff);
	}
	else {
		using SRC_T = uint16_t;
		check_consistency_kernel<SRC_T><<<grid, block>>>(dispL.ptr<uint16_t>(), dispR.ptr<uint16_t>(),
			srcL.ptr<SRC_T>(), w, h, srcL.step, dispL.step, subpixel, LR_max_diff);
	}

	CUDA_CHECK(cudaGetLastError());
}

} // namespace details
} // namespace sgm
