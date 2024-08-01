/************************************************************************//**
 * File: auxgpu.cpp
 * Description: Auxiliary utilities to manage GPU usage
 * Project: Synchrotron Radiation Workshop
 * First release: 2023
 *
 * Copyright (C) Brookhaven National Laboratory
 * All Rights Reserved
 *
 * @author H.Goel
 * @version 1.0
 ***************************************************************************/

#include <cstdio>
#include <cstdlib>
#include <new>
#include <cstring> //HG26072024

#ifdef _OFFLOAD_GPU
#include <cuda_runtime.h>
#endif

#include "auxgpu.h"

static bool isGPUAvailable = false;
static bool isGPUEnabled = false;
static bool GPUAvailabilityTested = false;
static bool deviceOffloadInitialized = false;
static int deviceCount = 0;

#ifdef _OFFLOAD_GPU
typedef struct
{
	void *devicePtr;
	void *hostPtr;
	size_t size;
	bool HostToDevUpdated;
	bool DevToHostUpdated;
	cudaEvent_t h2d_event;
	cudaEvent_t d2h_event;
} memAllocInfo_t;
static std::map<void*, memAllocInfo_t> gpuMap;
static cudaStream_t memcpy_stream;
static bool memcpy_stream_initialized = false;
static int current_device = -1;
#endif

static void CheckGPUAvailability() 
{
#ifdef _OFFLOAD_GPU
	if (!GPUAvailabilityTested)
	{
		isGPUAvailable = false;
		GPUAvailabilityTested = true;
		int deviceCount = 0;
		if (cudaGetDeviceCount(&deviceCount) != cudaSuccess)
			return;

		if (deviceCount < 1)
			return;

		isGPUAvailable = true;
	}
#else
	isGPUAvailable = false;
	isGPUEnabled = false;
	GPUAvailabilityTested = true;
#endif
}

bool CAuxGPU::GPUAvailable()
{
	CheckGPUAvailability();
	return isGPUAvailable;
}

bool CAuxGPU::GPUEnabled(TGPUUsageArg *arg) 
{
#ifdef _OFFLOAD_GPU
	if (arg == NULL)
		return false;
	if (arg->deviceIndex > 0) {
		if (arg->deviceIndex <= deviceCount)
		{
			if (memcpy_stream_initialized && current_device != arg->deviceIndex)
			{
				cudaStreamDestroy(memcpy_stream);
				memcpy_stream_initialized = false;
			}
			cudaSetDevice(arg->deviceIndex - 1);
			if (!memcpy_stream_initialized)
				cudaStreamCreateWithFlags(&memcpy_stream, cudaStreamNonBlocking);
			current_device = arg->deviceIndex;
			memcpy_stream_initialized = true;
		}
		//TODO: Add warning that GPU isn't available
		return GPUAvailable();
	}
#endif
	return false;
}

void CAuxGPU::SetGPUStatus(bool enabled)
{
	isGPUEnabled = enabled && GPUAvailable();
}

int CAuxGPU::GetDevice(TGPUUsageArg* arg)
{
#ifdef _OFFLOAD_GPU
	if (arg == NULL)
		return cudaCpuDeviceId;

	int curDevice = 0;
	cudaGetDevice(&curDevice);
	return curDevice;
#else
	return 0;
#endif
}

//void* CAuxGPU::ToDevice(TGPUUsageArg* arg, void* hostPtr, size_t size, bool dontCopy)
void* CAuxGPU::ToDevice(TGPUUsageArg* arg, void* hostPtr, size_t size, bool dontCopy, bool pinOnHost, int zeroMode) //HG26072024
{
#ifdef _OFFLOAD_GPU
	if (arg == NULL)
		return hostPtr;
	if (arg->deviceIndex == 0)
		return hostPtr;
	if (hostPtr == NULL)
		return hostPtr;
	if (size == 0)
		return hostPtr;
	if (!GPUEnabled(arg))
		return hostPtr;
	if (gpuMap.find(hostPtr) != gpuMap.end()){
		memAllocInfo_t info = gpuMap[hostPtr];
		void* devPtr = info.devicePtr;
		hostPtr = info.hostPtr;
		if (gpuMap[devPtr].HostToDevUpdated && !dontCopy){
			cudaMemcpyAsync(devPtr, hostPtr, size, cudaMemcpyHostToDevice, memcpy_stream);
			cudaEventRecord(gpuMap[devPtr].h2d_event, memcpy_stream);
		}
//#if _DEBUG
//		printf("ToDevice: %p -> %p, %d, D2H: %d, H2D: %d\n", hostPtr, devPtr, size, gpuMap[devPtr].DevToHostUpdated, gpuMap[devPtr].HostToDevUpdated); //HG28072023
//#endif
		gpuMap[devPtr].HostToDevUpdated = false;
		return devPtr;
	}

	size_t free_mem, total_mem;  //HG26072024 If the memory request is very large, it may be more optimal to pin to host memory
	cudaMemGetInfo(&free_mem, &total_mem);
	if(size >= free_mem * 0.9)
	{
		pinOnHost = true;
	}

	void *devicePtr = NULL;
	//cudaError_t err = cudaMalloc(&devicePtr, size);
	cudaError_t err = cudaSuccess; //HG26072024 Switch to asynchronous malloc
	if (!pinOnHost)
	{
		err = cudaMallocAsync(&devicePtr, size, memcpy_stream); //Try asynchronous allocation
		if (err != cudaSuccess) // Try again after freeing up some memory HG24072023
		{
			cudaDeviceSynchronize();
			err = cudaMalloc(&devicePtr, size);
			if (err != cudaSuccess) pinOnHost = true; //HG26072024 If allocation still fails, try pinning on host
		}
	}
	if (pinOnHost) //HG26072024 Fallback to pinning host memory directly
	{
		err = cudaHostRegister(hostPtr, size, cudaHostRegisterDefault);
		devicePtr = hostPtr;
	}
	if (err != cudaSuccess)
		return NULL;
//#if _DEBUG
//	printf("ToDevice: %p -> %p, %d\n", hostPtr, devicePtr, size); //HG28072023
//#endif
	memAllocInfo_t info;
	info.devicePtr = devicePtr;
	info.hostPtr = hostPtr;
	info.DevToHostUpdated = false;
	info.HostToDevUpdated = false;
	cudaEventCreateWithFlags(&info.h2d_event, cudaEventDisableTiming);
	cudaEventCreateWithFlags(&info.d2h_event, cudaEventDisableTiming);
	//if (!dontCopy){
	//	cudaMemcpyAsync(devicePtr, hostPtr, size, cudaMemcpyHostToDevice, memcpy_stream);
	//	cudaEventRecord(info.h2d_event, memcpy_stream);
	//}
	if (devicePtr != hostPtr) //HG26072024
	{
		info.HostToDevUpdated = true;
		if (!dontCopy)
		{
			cudaMemcpyAsync(devicePtr, hostPtr, size, cudaMemcpyHostToDevice, memcpy_stream);
		}
		else 
		{
			switch(zeroMode) //HG27072024 Add memset options
			{
				case 0:
					cudaMemsetAsync(devicePtr, 0, size, memcpy_stream);
					break;
				case 1:
					Memset_GPU((float*)devicePtr, 0.0f, size/sizeof(float), (long long)memcpy_stream);
					break;
				case 2:
					Memset_GPU((double*)devicePtr, 0.0, size/sizeof(double), (long long)memcpy_stream);
					break;
			}
		}
		cudaEventRecord(info.h2d_event, memcpy_stream);
	}
	info.size = size;
	gpuMap[hostPtr] = info;
	gpuMap[devicePtr] = info;
	return devicePtr;
#else
	return hostPtr;
#endif
}

void CAuxGPU::EnsureDeviceMemoryReady(TGPUUsageArg* arg, void* hostPtr)
{
#ifdef _OFFLOAD_GPU
	if (arg == NULL)
		return;
	if (arg->deviceIndex == 0)
		return;
	if (hostPtr == NULL)
		return;
	if (!GPUEnabled(arg))
		return;
	if (gpuMap.find(hostPtr) != gpuMap.end()){
		void* devPtr = gpuMap[hostPtr].devicePtr;
//#if _DEBUG
//		printf("EnsureDeviceMemoryReady: %p -> %p, %d, D2H: %d, H2D: %d\n", hostPtr, devPtr, gpuMap[devPtr].size, gpuMap[devPtr].DevToHostUpdated, gpuMap[devPtr].HostToDevUpdated); //HG28072023
//#endif
		if (gpuMap[devPtr].HostToDevUpdated){
			cudaStreamWaitEvent(0, gpuMap[devPtr].h2d_event);
			gpuMap[devPtr].HostToDevUpdated = false; //HG26072024 After this event the latest data is known to be on device
			gpuMap[gpuMap[devPtr].hostPtr].HostToDevUpdated = false; //HG26072024
		}
	}
#endif
}

void* CAuxGPU::GetHostPtr(TGPUUsageArg* arg, void* devicePtr)
{
#ifdef _OFFLOAD_GPU
	if (arg == NULL)
		return devicePtr;
	if (arg->deviceIndex == 0)
		return devicePtr;
	if (devicePtr == NULL)
		return devicePtr;
	if (!GPUEnabled(arg))
		return devicePtr;
	memAllocInfo_t info;
	if (gpuMap.find(devicePtr) == gpuMap.end())
		return devicePtr;
	info = gpuMap[devicePtr];
//#if _DEBUG
//	printf("GetHostPtr: %p -> %p\n", devicePtr, info.hostPtr); //HG28072023
//#endif
	return info.hostPtr;
#else
	return devicePtr;
#endif
}

void* CAuxGPU::ToHostAndFree(TGPUUsageArg* arg, void* devicePtr, size_t size, bool dontCopy)
{
#ifdef _OFFLOAD_GPU
	if (arg == NULL)
		return devicePtr;
	if (arg->deviceIndex == 0)
		return devicePtr;
	if (devicePtr == NULL)
		return devicePtr;
	if (size == 0)
		return devicePtr;
	if (!GPUEnabled(arg))
		return devicePtr;
	memAllocInfo_t info;
	if (gpuMap.find(devicePtr) == gpuMap.end())
		return devicePtr;
	info = gpuMap[devicePtr];
	devicePtr = info.devicePtr;
	void *hostPtr = info.hostPtr;
	if (!dontCopy && info.DevToHostUpdated)
	{
		cudaStreamWaitEvent(memcpy_stream, info.d2h_event, 0);
		//cudaMemcpyAsync(hostPtr, devicePtr, size, cudaMemcpyDeviceToHost, memcpy_stream);
		//if(hostPtr != devicePtr) cudaMemcpyAsync(hostPtr, devicePtr, size, cudaMemcpyDeviceToHost, memcpy_stream); //HG26072024 only copy if not using pinned memory
		//cudaFreeAsync(devicePtr, memcpy_stream); //HG26072024 Doing the async free here is  slightly more efficient and eliminates a potential use-after-free
		//cudaEventSynchronize(info.d2h_event); // we can't treat host memory as valid until the copy is complete
		if(hostPtr != devicePtr) //HG30072024 Properly handle pinned memory
		{
			cudaMemcpyAsync(hostPtr, devicePtr, size, cudaMemcpyDeviceToHost, memcpy_stream); //HG26072024 only copy if not using pinned memory
			cudaFreeAsync(devicePtr, memcpy_stream); //HG26072024 Doing the async free here is  slightly more efficient and eliminates a potential use-after-free
			cudaEventSynchronize(info.d2h_event); // we can't treat host memory as valid until the copy is complete
		}
		else
		{
			cudaEventSynchronize(info.d2h_event); // we can't treat host memory as valid until the copy is complete
			cudaHostUnregister(hostPtr);
		}
	}
	else //HG26072024
	{
		//cudaStreamWaitEvent(0, info.h2d_event);
		if(hostPtr != devicePtr) cudaStreamWaitEvent(0, info.h2d_event); //HG26072024 H2D events are meaningless when the memory is on host
		cudaStreamWaitEvent(0, info.d2h_event);
		cudaFreeAsync(devicePtr, 0);
	}
//#if _DEBUG
//	printf("ToHostAndFree: %p -> %p, %d, D2H:%d, dontCopy: %d\n", devicePtr, hostPtr, size, info.DevToHostUpdated, dontCopy); //HG28072023
//#endif
	cudaEventDestroy(info.h2d_event);
	cudaEventDestroy(info.d2h_event);
	gpuMap.erase(devicePtr);
	gpuMap.erase(hostPtr);
	return hostPtr;
#else
	return devicePtr;
#endif
}

//void CAuxGPU::FreeHost(void* ptr) //HG26072024 Commented out function
//{
//#ifdef _OFFLOAD_GPU
//	if (ptr == NULL)
//		return;
//	if (gpuMap.find(ptr) == gpuMap.end())
//		return;
//	memAllocInfo_t info = gpuMap[ptr];
//	void *hostPtr = info.hostPtr;
//	void *devicePtr = info.devicePtr;
////#if _DEBUG
////	printf("FreeHost: %p, %p\n", devicePtr, hostPtr);
////#endif
//    //cudaStreamWaitEvent(0, info.h2d_event);
//	//cudaStreamWaitEvent(0, info.d2h_event);
//	cudaFreeAsync(devicePtr, 0);
//	//cudaEventDestroy(info.h2d_event);
//	//cudaEventDestroy(info.d2h_event);
//	std::free(hostPtr); //OC02082023
//	//CAuxGPU::free(hostPtr);
//	gpuMap.erase(devicePtr);
//	gpuMap.erase(hostPtr);
//#endif
//	return;
//}

int CAuxGPU::SetHostPtr(TGPUUsageArg* arg, void* origPtr, void* newPtr, size_t size) //HG26072024 Add function
{
#ifdef _OFFLOAD_GPU
	if (arg == NULL)
		return -1;
	if (arg->deviceIndex == 0)
		return -1;
	if (origPtr == NULL)
		return -1;
	if (newPtr == NULL)
		return -1;
	if (!GPUEnabled(arg))
		return -1;
//#if _DEBUG
//	printf("SetHostPtr: %p -> %p\n", origPtr, newPtr);
//#endif
	if (gpuMap.find(origPtr) == gpuMap.end())
	{
		memcpy(newPtr, origPtr, size);
		return 0;
	}
	memAllocInfo_t info = gpuMap[origPtr];
	if (gpuMap.find(newPtr) != gpuMap.end())
		return -1;	//The new pointer should not already be known to the GPU memory map, else we will have a memory leak

	if (info.DevToHostUpdated) 
	{
		gpuMap.erase(origPtr);
		info.hostPtr = newPtr;
		gpuMap[info.hostPtr] = info;
		gpuMap[info.devicePtr] = info;
	}
	else 
	{
		memcpy(newPtr, origPtr, size);
	}
#endif
	return 0;
}

void CAuxGPU::MarkUpdated(TGPUUsageArg* arg, void* ptr, bool devToHost, bool hostToDev)
{
#ifdef _OFFLOAD_GPU
	if (arg == NULL)
		return;
	if (arg->deviceIndex == 0)
		return;
	if (ptr == NULL)
		return;
	if (!GPUEnabled(arg))
		return;
	if (gpuMap.find(ptr) == gpuMap.end())
		return;
	void* devPtr = gpuMap[ptr].devicePtr;
	void* hostPtr = gpuMap[ptr].hostPtr;
	if ((devToHost | gpuMap[devPtr].DevToHostUpdated) && (hostToDev | gpuMap[devPtr].HostToDevUpdated)) //HG26072024 Trying to perform both a copy to host and then a copy back to device doesn't make sense
		return;
	gpuMap[devPtr].DevToHostUpdated = devToHost;
	gpuMap[devPtr].HostToDevUpdated = hostToDev;
	gpuMap[hostPtr].DevToHostUpdated = devToHost;
	gpuMap[hostPtr].HostToDevUpdated = hostToDev;
	if (devToHost)
		cudaEventRecord(gpuMap[devPtr].d2h_event, 0);
	if (hostToDev) //HG26072024 If host data has been updated, copy it over
	{
		cudaMemcpyAsync(devPtr, hostPtr, gpuMap[devPtr].size, cudaMemcpyHostToDevice, memcpy_stream);
		cudaEventRecord(gpuMap[devPtr].h2d_event, memcpy_stream);
	}
//#if _DEBUG
//	printf("MarkUpdated: %p -> %p, D2H: %d, H2D: %d\n", ptr, devPtr, devToHost, hostToDev);
//#endif
#endif
}

void CAuxGPU::Init() {
	deviceOffloadInitialized = true;
#ifdef _OFFLOAD_GPU
	cudaGetDeviceCount(&deviceCount);
	cudaDeviceSynchronize();
#endif
}

void CAuxGPU::Fini() {
#ifdef _OFFLOAD_GPU
	SetGPUStatus(false); //HG30112023 Disable GPU

	// Copy back all updated data
	bool updated = false;
	bool freed = false;
	for (std::map<void*, memAllocInfo_t>::const_iterator it = gpuMap.cbegin(); it != gpuMap.cend(); it++)
	{
		if (it->second.DevToHostUpdated){
			cudaStreamWaitEvent(memcpy_stream, it->second.d2h_event, 0);
			cudaMemcpyAsync(it->second.hostPtr, it->second.devicePtr, it->second.size, cudaMemcpyDeviceToHost, memcpy_stream);
//#if _DEBUG
//			printf("Fini: %p -> %p, %d\n", it->second.devicePtr, it->second.hostPtr, it->second.size);
//#endif
			updated = true;
			gpuMap[it->second.hostPtr].DevToHostUpdated = false;
			gpuMap[it->second.devicePtr].DevToHostUpdated = false;
		}
	}
	for (std::map<void*, memAllocInfo_t>::const_iterator it = gpuMap.cbegin(); it != gpuMap.cend(); it++)
	{
		if (it->first == it->second.devicePtr)
		{
			cudaStreamWaitEvent(0, it->second.h2d_event);
			cudaStreamWaitEvent(0, it->second.d2h_event);
			cudaFreeAsync(it->second.devicePtr, 0);
			freed = true;
			cudaEventDestroy(it->second.h2d_event);
			cudaEventDestroy(it->second.d2h_event);
		}
	}
	if (updated | freed)
		cudaStreamSynchronize(0);
	gpuMap.clear();
//#if _DEBUG
//	printf("Fini: %d\n", gpuMap.size());
//#endif
#endif
}