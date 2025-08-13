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

#include <cstdarg>
#include <cstdlib>
#include <stdio.h>
#include <typeinfo>

#ifdef _OFFLOAD_GPU
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <map>
#include <initializer_list>
//#if CUDART_VERSION < 11020
//#error CUDA version too low, need at least 11.2
//#endif
#endif

//typedef struct
struct TGPUUsageArg //OC18022024
{
	int deviceIndex; // -1 means no device, TODO

	TGPUUsageArg(void* pvGPU=0) //OC18022024
	{
		deviceIndex = -1;
		if(pvGPU == 0) return;
		double *arParGPU = (double*)pvGPU;
		int nPar = (int)arParGPU[0];
		if(nPar > 0) deviceIndex = (int)arParGPU[1];
		//continue here for future params
	}
}; 
//} TGPUUsageArg; //OC18022024 (commented-out)

#ifdef _OFFLOAD_GPU
#define GPU_COND(arg, code) if (arg && CAuxGPU::GPUEnabled((TGPUUsageArg*)arg)) { code }
//#define GPU_COND(arg, code) if (arg && CAuxGPU::GPUEnabled(arg)) { code }
#define GPU_PORTABLE __device__ __host__
#else
#define GPU_COND(arg, code) if(0) { }
#define GPU_PORTABLE 
#endif

#ifdef __CUDACC__
	const int PerThread = 16;
	template<typename T> __global__ void Memset_Kernel(T* p, T val, long long n)
	{
		long long offset = blockIdx.x * blockDim.x + threadIdx.x;
		offset *= PerThread;
		long long dst = min(offset + PerThread, n);
		for (; offset < dst; offset++)
			p[offset] = val;
	}
#endif

 //*************************************************************************
class CAuxGPU
{
private:

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
		bool pinned; //HG26072024
	} memAllocInfo_t;
	static std::map<void*, memAllocInfo_t> gpuMap;
	//static bool memcpy_stream_initialized = false; //HG02082024 (commented-out)
	static std::map<int, cudaStream_t*> streams; //HG02082024
	static cudaStream_t memcpy_stream;
#endif

	
	//static void* ToDevice(TGPUUsageArg* arg, void* hostPtr, size_t size, bool dontCopy = false); //HG26072024
	static void* _ToDevice(TGPUUsageArg* arg, void* hostPtr, size_t size, int flags=0); //HG26072024 Make private

	static void* _ToHostAndFree(TGPUUsageArg* arg, void* devicePtr, int flags=0, size_t size=0); //HG26072024
	//static void* ToHostAndFree(TGPUUsageArg* arg, void* devicePtr, size_t size, bool dontCopy = false);
public:
	/**
	 * Flags used by this class
	 */
	static constexpr int DONT_COPY = (1 << 0);
	static constexpr int PIN_ON_HOST = (1 << 1);
	static constexpr int HOST = (1 << 2);
	static constexpr int DEVICE = (1 << 3);

	/**
	* Initialize GPU/device functionality
	*/
	//static void Init();
	static void Init(TGPUUsageArg *arg); //HG02082024

	/**
	* Call when returning to the client layer to ensure all memory is accessible on CPU/host again
	*/
	//static void Fini();
	static void Fini(TGPUUsageArg *arg); //HG02082024

	static bool GPUAvailable(); //CheckGPUAvailable etc
	static bool GPUEnabled(TGPUUsageArg *arg);
	static void SetGPUStatus(bool enabled);

	/**
	*  Get the GPU/device number associated with arg
	*  @param [in] arg pointer to a GPU usage argument structure
	*  @return integer number of the GPU/device, -1 if CPU/host
	*/
	static int GetDevice(TGPUUsageArg* arg);

	/**
	*  Associate the specified region of host memory with memory on the device, copies the memory to device by default
	* @param [in] arg pointer to a GPU usage argument structure
	* @param [in] hostPtr pointer to the region of host memory
	* @param [in] size size in bytes of the memory region
	* @param [in] flags flags to control the memory transfer (DONT_COPY, PIN_ON_HOST)
	* @return pointer to device memory, NULL on error
	*/
	template <typename T>
	static T* ToDevice(TGPUUsageArg* arg, T* hostPtr, size_t elemCount, int flags=0) //HG30042025
	{
#ifdef _OFFLOAD_GPU
		const int typeSize = sizeof(T);
		return (T*)_ToDevice(arg, (void*)hostPtr, elemCount * typeSize, flags);
#endif
		return hostPtr;
	}

	template <typename T>
	static void Memset(TGPUUsageArg* arg, T* devicePtr, T value, size_t elemCount) //HG30042025
	{
#if defined(_OFFLOAD_GPU) && defined(__CUDACC__)
		if (arg == NULL)
			return;
		if (arg->deviceIndex == 0)
			return;
		if (!GPUEnabled(arg))
			return;
		if (devicePtr == NULL)
			return;
		if (elemCount == 0)
			return;
		if (gpuMap.find(devicePtr) != gpuMap.end()){
			void* devPtr = devicePtr;
			if (gpuMap[devPtr].DevToHostUpdated){
				cudaStreamWaitEvent(memcpy_stream, gpuMap[devPtr].d2h_event);
				gpuMap[devPtr].DevToHostUpdated = false;
				if (gpuMap[devPtr].hostPtr != NULL) gpuMap[gpuMap[devPtr].hostPtr].DevToHostUpdated = false;
			}

			int minGridSize = 0;
			int bs = 256;
			size_t elemCount_orig = elemCount;
			cudaOccupancyMaxPotentialBlockSize(&minGridSize, &bs, Memset_Kernel<T>, 0, (elemCount + PerThread - 1) / PerThread);
			elemCount = ((elemCount + PerThread - 1) / PerThread + bs - 1) / bs;
			Memset_Kernel<T> <<<elemCount, bs, 0, memcpy_stream>>> ((T*)devPtr, value, elemCount_orig);
		}
#endif
	}

	/**
	* Retrieve the host memory address for a given device or host pointer
	* @param [in] arg pointer to a GPU usage argument structure
	* @param [in] devicePtr pointer for which the host pointer is desired
	* @return the corresponding host pointer, NULL on errror
	*/
	static void* GetHostPtr(TGPUUsageArg* arg, void* devicePtr);

	/**
	* Transfer memory back to the host if necessary and free the associated device memory. Does not return until the latest copy of the data is on the host.
	* @param [in] arg pointer to a GPU usage argument structure
	* @param [in] devicePtr device pointer to the memory to be freed, if a host pointer is provided, the corresponding device pointer is freed
	* @param [in] flags flags to control the memory transfer (DONT_COPY)
	* @param [in] size size of the block to be freed
	* @return The corresponding host pointer, NULL on error
	*/
	template <typename T>
	static T* ToHostAndFree(TGPUUsageArg* arg, T* devicePtr, int flags=0, size_t elemCount=0) //HG30042025
	{
#ifdef _OFFLOAD_GPU
		const int typeSize = (typeid(T) == typeid(void)) ? 1 : sizeof(T);
		return (T*)_ToHostAndFree(arg, (void*)devicePtr, flags, elemCount * typeSize);
#endif
		return devicePtr;
	}

	/**
	* Ensure that the device memory has the latest data, used prior to kernel launches
	* @param [in] arg pointer to a GPU usage argument structure
	* @param [in] devicePtr device pointer to the memory block to be operated on
	*/
	static void EnsureDeviceMemoryReady(TGPUUsageArg* arg, void* devicePtr=0);

	// Makes it possible to pass multiple pointers to EnsureDeviceMemoryReady in a single call
	template <typename First, typename... T> 
	static void EnsureDeviceMemoryReady(TGPUUsageArg* arg, First* devicePtr, T*... ptrs) //HG30042025	
	{
#ifdef _OFFLOAD_GPU
		if (arg == NULL)
			return;
		if (arg->deviceIndex == 0)
			return;
		if (!GPUEnabled(arg))
			return;
		EnsureDeviceMemoryReady(arg, (void*)devicePtr);
		EnsureDeviceMemoryReady(arg, ptrs...);
#endif
	}

	//static void FreeHost(void* ptr); //HG26072024 (Commented out) Unused and potentially breaks this memory management model

	/**
	* If origPtr is a host pointer that has a corresponding device memory block, reassign that block to correspond to newPtr instead, otherwise copy the data from origPtr to newPtr on host
	* @param [in] arg pointer to a GPU usage argument structure
	* @param [in] origPtr original host pointer
	* @param [in] newPtr host pointer to replace it with
	* @param [in] size size of this memory region
	* @return 0 on success, -1 on error
	*/
	static int SetHostPtr(TGPUUsageArg* arg, void* origPtr, void* newPtr, size_t size); //HG26072024

	/**
	* Mark the region as having been updated.
	* @param [in] arg pointer to a GPU usage argument structure
	* @param [in] ptr pointer to the memory region, can be a host or device pointer
	* @param [in] flags flags to control the memory transfer (HOST: host to device, DEVICE: device to host)
	*/
	static void MarkUpdated(TGPUUsageArg* arg, void* ptr, int flags=0); //HG26072024
	//static void MarkUpdated(TGPUUsageArg* arg, void* ptr, bool devToHost, bool hostToDev);

	/**
	* Retrieve a compute stream index to run kernels simultaneously on one GPU.
	* @param [in] arg pointer to a GPU usage argument structure
	* @param [in] idx the 0-based index of the desired compute stream
	* @return The cudaStream ID associated with the requested compute stream index
	*/
	static long long GetComputeStream(TGPUUsageArg* arg, int idx); //HG26072024

	/**
	* Ensure that the specified compute stream is synchronized with the target compute stream
	* @param [in] arg pointer to a GPU usage argument structure
	* @param [in] targetStreamIdx the 0-based index of the target compute stream
	* @param [in] streamIdx the 0-based index of the compute stream to be synchronized
	*/
	static void SyncComputeStream(TGPUUsageArg* arg, long long targetStreamIdx, long long streamIdx); //HG24042025

#ifdef __CUDACC__
	/**
	* Determine a good distribution of threads within a block
	* @param [in] func The kernel to calculate for.
	* @param [in] grid The grid size.
	* @param [out] blocks The resulting block count.
	* @param [out] threads The resulting thread count.
	* @param [in] max_x_bs The maximum block size in the X dimension.
	* @param [in] max_bs_total The maximum block size the kernel is designed to work with.
	*/
	template<typename T>
	static void CalcLaunchDims(T* kern, dim3 grid, dim3& blocks, dim3& threads, int max_x_bs = 0, int max_bs_total = 0)
	{
		int minGridSize = 0; //HG05082024
    	int bs = 256;
		cudaOccupancyMaxPotentialBlockSize(&minGridSize, &bs, (void*)kern, 0, max_bs_total);
		if ((max_x_bs == 0) || (grid.x < bs && grid.x < max_x_bs)) max_x_bs = grid.x;

		threads.x = (max_x_bs < bs) ? max_x_bs : bs;
		threads.y = 1;
		threads.z = 1;

		blocks.x = grid.x / threads.x + !!(grid.x & (threads.x - 1)); //round up the division result
		blocks.y = grid.y;
		blocks.z = grid.z;

		int y_v = bs / max_x_bs;
		if (y_v > 1 && grid.y > 1)
		{
			//Calculate y grid
			threads.y = (y_v > grid.y) ? grid.y : y_v;
			blocks.y = grid.y / threads.y + !!(grid.y & (threads.y - 1)); //round up the division result

			int z_v = y_v / threads.y;
			if (z_v > 1 && grid.z > 1)
			{
				threads.z = (z_v > grid.z) ? grid.z : z_v;
				blocks.z = grid.z / threads.z + !!(grid.z & (threads.z - 1)); //round up the division result
			}
		}
	}
#endif
};

//*************************************************************************
#endif