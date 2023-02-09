/************************************************************************//**
 * File: auxgpu.h
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

#ifndef __UTIGPU_H
#define __UTIGPU_H

#include <cstdlib>
#include <stdio.h>

#ifdef _OFFLOAD_GPU
#include <cuda_runtime.h>
#include <map>
//#if CUDART_VERSION < 11020
//#error CUDA version too low, need at least 11.2
//#endif
#endif



typedef struct
{
	int deviceIndex; // -1 means no device, TODO
} gpuUsageArg; 

//#define ALLOC_ARRAY(type, size) (type *)AuxGpu::malloc(sizeof(type)*(size))
//#define FREE_ARRAY(x) AuxGpu::free(x); x=NULL

#ifdef _OFFLOAD_GPU
#define GPU_ENABLED(arg) AuxGpu::GPUEnabled(arg)
#define GPU_COND(arg, code) if (GPU_ENABLED(arg)) { code }
#define GPU_PORTABLE __device__ __host__
#else
#define GPU_COND(arg, code) if(0) { }
#define GPU_ENABLED(arg) 0
#define GPU_PORTABLE 
#endif

 //*************************************************************************
class AuxGpu
{
private:
public:
	static void Init();
	static void Fini();
	static bool GPUAvailable(); //CheckGPUAvailable etc
	static bool GPUEnabled(gpuUsageArg *arg);
	static void SetGPUStatus(bool enabled);
	static int GetDevice(gpuUsageArg* arg);
	static void* ToDevice(gpuUsageArg* arg, void* hostPtr, size_t size, bool dontCopy = false);
	static void* GetHostPtr(gpuUsageArg* arg, void* devicePtr);
	static void* ToHostAndFree(gpuUsageArg* arg, void* devicePtr, size_t size, bool dontCopy = false);
	static void EnsureDeviceMemoryReady(gpuUsageArg* arg, void* devicePtr);
	static void FreeHost(void* ptr);
	static void MarkUpdated(gpuUsageArg* arg, void* ptr, bool devToHost, bool hostToDev);
	static inline void* malloc(size_t sz) {
/*#ifdef _OFFLOAD_GPU
			void *ptr;
			auto err = cudaMallocManaged(&ptr, sz);
			if (err != cudaSuccess)
				printf("Allocation Failure\r\n");
			return ptr;
#else*/
			return std::malloc(sz);
//#endif
	}

	static inline void free(void* ptr) {
//#ifdef _OFFLOAD_GPU
//		FreeHost(ptr);
//#else
		std::free(ptr);
//#endif
	}
};

//*************************************************************************
#endif