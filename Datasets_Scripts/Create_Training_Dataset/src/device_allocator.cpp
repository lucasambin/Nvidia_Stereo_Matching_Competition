#include "device_allocator.h"

#include <cuda_runtime.h>

#include "host_utility.h"

namespace sgm
{

DeviceAllocator::DeviceAllocator() : data_(nullptr), ref_count_(nullptr), capacity_(0)
{
}

DeviceAllocator::DeviceAllocator(const DeviceAllocator& other)
{
	copy_construct_from(other);
}

DeviceAllocator::DeviceAllocator(DeviceAllocator&& right)
{
	move_construct_from(std::move(right));
}

DeviceAllocator::~DeviceAllocator()
{
	release();
}

void* DeviceAllocator::allocate(size_t size)
{
	if (size > capacity_)
	{
		release();
		CUDA_CHECK(cudaMalloc(&data_, size));
		ref_count_ = new int(1);
		capacity_ = size;
	}
	return data_;
}

void DeviceAllocator::assign(void* data, size_t size)
{
	release();
	data_ = data;
	capacity_ = size;
}

void DeviceAllocator::release()
{
	if (ref_count_ && --(*ref_count_) == 0)
	{
		CUDA_CHECK(cudaFree(data_));
		delete ref_count_;
	}

	data_ = ref_count_ = nullptr;
	capacity_ = 0;
}

DeviceAllocator& DeviceAllocator::operator=(const DeviceAllocator& other)
{
	release();
	copy_construct_from(other);
	return *this;
}

DeviceAllocator& DeviceAllocator::operator=(DeviceAllocator&& right)
{
	release();
	move_construct_from(std::move(right));
	return *this;
}

void DeviceAllocator::copy_construct_from(const DeviceAllocator& other)
{
	data_ = other.data_;
	ref_count_ = other.ref_count_;
	capacity_ = other.capacity_;

	if (ref_count_)
		(*ref_count_)++;
}

void DeviceAllocator::move_construct_from(DeviceAllocator&& right)
{
	data_ = right.data_;
	ref_count_ = right.ref_count_;
	capacity_ = right.capacity_;

	right.data_ = right.ref_count_ = nullptr;
	right.capacity_ = 0;
}

} // namespace sgm
