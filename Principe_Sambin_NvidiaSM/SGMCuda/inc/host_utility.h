#ifndef __HOST_UTILITY_H__
#define __HOST_UTILITY_H__

#include <cstdio>
#include <stdexcept>

#define CUDA_CHECK(err) \
do {\
	if (err != cudaSuccess) { \
		printf("[CUDA Error] %s (code: %d) at %s:%d\n", cudaGetErrorString(err), err, __FILE__, __LINE__); \
	} \
} while (0)

#define SGM_ASSERT(expr, msg) \
if (!(expr)) { \
	throw std::logic_error(msg); \
} \

namespace sgm
{

static inline int divUp(int total, int grain)
{
	return (total + grain - 1) / grain;
}

} // namespace sgm

#endif // !__HOST_UTILITY_H__
