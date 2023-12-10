#ifndef __DEVICE_ALLOCATOR_H__
#define __DEVICE_ALLOCATOR_H__

#include <cstddef>

namespace sgm
{

class DeviceAllocator
{
public:

	DeviceAllocator();
	DeviceAllocator(const DeviceAllocator& other);
	DeviceAllocator(DeviceAllocator&& right);
	~DeviceAllocator();
	void* allocate(size_t size);
	void assign(void* data, size_t size);
	void release();

	DeviceAllocator& operator=(const DeviceAllocator& other);
	DeviceAllocator& operator=(DeviceAllocator&& right);

private:

	void copy_construct_from(const DeviceAllocator& other);
	void move_construct_from(DeviceAllocator&& right);

	void* data_;
	int* ref_count_;
	size_t capacity_;
};

} // namespace sgm

#endif // !__DEVICE_ALLOCATOR_H__
