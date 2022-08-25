/************************************************************************//**
 * File: utigpu.cpp
 * Description: Auxiliary utilities to support GPU management
 *
 * @author H.Goel
 ***************************************************************************/

#include <cstdio>
#include <cstdlib>
#include <new>

#ifdef _OFFLOAD_GPU
#include <cuda_runtime.h>
#endif

#include "utidev.h"

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
} memAllocInfo_t;
static std::map<void*, memAllocInfo_t> gpuMap;
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

bool UtiDev::GPUAvailable()
{
	CheckGPUAvailability();
	return isGPUAvailable;
}

bool UtiDev::GPUEnabled(gpuUsageArg_t *arg) 
{
#ifdef _OFFLOAD_GPU
	if (arg == NULL)
		return false;
	if (arg->deviceIndex > 0) {
		if (arg->deviceIndex <= deviceCount)
			cudaSetDevice(arg->deviceIndex - 1);
		return GPUAvailable();
	}
#endif
	return false;
}

void UtiDev::SetGPUStatus(bool enabled)
{
	isGPUEnabled = enabled && GPUAvailable();
}

int UtiDev::GetDevice(gpuUsageArg_t* arg)
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

void* UtiDev::ToDevice(gpuUsageArg_t* arg, void* hostPtr, size_t size, bool dontCopy)
{
#ifdef _OFFLOAD_GPU
	if (arg == NULL)
		return hostPtr;
	if (arg->deviceIndex == 0)
		return hostPtr;
	if (hostPtr == NULL)
		return NULL;
	if (size == 0)
		return NULL;
	if (!GPUEnabled(arg))
		return hostPtr;
	if (gpuMap.find(hostPtr) != gpuMap.end()){
		void* devPtr = gpuMap[hostPtr].devicePtr;
		if (gpuMap[devPtr].HostToDevUpdated && !dontCopy)
			cudaMemcpy(devPtr, hostPtr, size, cudaMemcpyHostToDevice);
		gpuMap[devPtr].HostToDevUpdated = false;
		return devPtr;
	}

	void *devicePtr = NULL;
	cudaError_t err = cudaMalloc(&devicePtr, size);
	if (err != cudaSuccess)
		return NULL;
	if (!dontCopy)
		cudaMemcpy(devicePtr, hostPtr, size, cudaMemcpyHostToDevice);
	memAllocInfo_t info;
	info.devicePtr = devicePtr;
	info.hostPtr = hostPtr;
	info.size = size;
	gpuMap[hostPtr] = info;
	gpuMap[devicePtr] = info;
	return devicePtr;
#else
	return hostPtr;
#endif
}

void* UtiDev::GetHostPtr(gpuUsageArg_t* arg, void* devicePtr)
{
#ifdef _OFFLOAD_GPU
	if (arg == NULL)
		return devicePtr;
	if (arg->deviceIndex == 0)
		return devicePtr;
	if (devicePtr == NULL)
		return NULL;
	if (!GPUEnabled(arg))
		return devicePtr;
	memAllocInfo_t info;
	if (gpuMap.find(devicePtr) == gpuMap.end())
		return NULL;
	info = gpuMap[devicePtr];
	return info.hostPtr;
#else
	return devicePtr;
#endif
}

void* UtiDev::ToHostAndFree(gpuUsageArg_t* arg, void* devicePtr, size_t size, bool dontCopy)
{
#ifdef _OFFLOAD_GPU
	if (arg == NULL)
		return devicePtr;
	if (arg->deviceIndex == 0)
		return devicePtr;
	if (devicePtr == NULL)
		return NULL;
	if (size == 0)
		return NULL;
	if (!GPUEnabled(arg))
		return devicePtr;
	memAllocInfo_t info;
	if (gpuMap.find(devicePtr) == gpuMap.end())
		return NULL;
	info = gpuMap[devicePtr];
	void *hostPtr = info.hostPtr;
	if (!dontCopy)
		cudaMemcpy(hostPtr, devicePtr, size, cudaMemcpyDeviceToHost);
	cudaFreeAsync(devicePtr, 0);
	gpuMap.erase(devicePtr);
	gpuMap.erase(hostPtr);
	return hostPtr;
#else
	return devicePtr;
#endif
}

void UtiDev::MarkUpdated(gpuUsageArg_t* arg, void* ptr, bool devToHost, bool hostToDev)
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
	gpuMap[devPtr].DevToHostUpdated = devToHost;
	gpuMap[devPtr].HostToDevUpdated = hostToDev;
#endif
}

void UtiDev::Init() {
	deviceOffloadInitialized = true;
#ifdef _OFFLOAD_GPU
	cudaGetDeviceCount(&deviceCount);
	cudaDeviceSynchronize();
#endif
}

void UtiDev::Fini() {
#ifdef _OFFLOAD_GPU
	// Copy back all updated data
	bool updated = false;
	for (std::map<void*, memAllocInfo_t>::iterator it = gpuMap.begin(); it != gpuMap.end(); it++)
	{
		if (it->second.DevToHostUpdated){
			cudaMemcpy(it->second.hostPtr, it->second.devicePtr, it->second.size, cudaMemcpyDeviceToHost);
			updated = true;
			it->second.DevToHostUpdated = false;
		}
	}
	if (updated)
		cudaDeviceSynchronize();
#endif
}