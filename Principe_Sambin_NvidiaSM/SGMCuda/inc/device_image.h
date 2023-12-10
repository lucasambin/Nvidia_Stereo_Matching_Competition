#ifndef __DEVICE_IMAGE_H__
#define __DEVICE_IMAGE_H__

#include "device_allocator.h"

namespace sgm
{

enum ImageType
{
	SGM_8U,
	SGM_16U,
	SGM_32U,
	SGM_64U,
};

class DeviceImage
{
public:

	DeviceImage();
	DeviceImage(int rows, int cols, ImageType type, int step = -1);
	DeviceImage(void* data, int rows, int cols, ImageType type, int step = -1);

	void create(int rows, int cols, ImageType type, int step = -1);
	void create(void* data, int rows, int cols, ImageType type, int step = -1);

	void upload(const void* data);
	void download(void* data) const;
	void fill_zero();

	template <typename T> T* ptr(int y = 0) { return (T*)data + y * step; }
	template <typename T> const T* ptr(int y = 0) const { return (T*)data + y * step; }

	void* data;
	int rows, cols, step;
	ImageType type;

private:

	DeviceAllocator allocator_;
};

} // namespace sgm

#endif // !__DEVICE_IMAGE_H__
