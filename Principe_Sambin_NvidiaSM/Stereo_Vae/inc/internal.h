#ifndef __INTERNAL_H__
#define __INTERNAL_H__

#include "sgm.h"
#include "device_image.h"

namespace sgm
{
namespace details
{

void census_transform(const DeviceImage& src, DeviceImage& dst, CensusType type);

void cost_aggregation(const DeviceImage& srcL, const DeviceImage& srcR, DeviceImage& dst,
	int disp_size, int P1, int P2, PathType path_type, int min_disp);

void winner_takes_all(const DeviceImage& src, DeviceImage& dstL, DeviceImage& dstR,
	int disp_size, float uniqueness, bool subpixel, PathType path_type);

void median_filter(const DeviceImage& src, DeviceImage& dst);

void check_consistency(DeviceImage& dispL, const DeviceImage& dispR, const DeviceImage& srcL, bool subpixel, int LR_max_diff);

void correct_disparity_range(DeviceImage& disp, bool subpixel, int min_disp);

void cast_16bit_to_8bit(const DeviceImage& src, DeviceImage& dst);
void cast_8bit_to_16bit(const DeviceImage& src, DeviceImage& dst);

} // namespace details
} // namespace sgm

#endif // !__INTERNAL_H__
