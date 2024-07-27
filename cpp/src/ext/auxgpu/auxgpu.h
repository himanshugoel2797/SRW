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

 //*************************************************************************
class CAuxGPU
{
private:
public:
	/**
	* Initialize GPU/device functionality
	*/
	static void Init();

	/**
	* Call when returning to the client layer to ensure all memory is accessible on CPU/host again
	*/
	static void Fini();
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
	* @param [in] dontCopy do not copy the host memory to device if true
	* @param [in] pinOnHost attempt to allocate this memory by pinning/page locking the hostPtr
	* @param [in] zeroMode Initialization to use for device memory: =-1 -none, =0 set all bytes to 0
	* @return pointer to device memory, NULL on error
	*/
	static void* ToDevice(TGPUUsageArg* arg, void* hostPtr, size_t size, bool dontCopy = false, bool pinOnHost = false, int zeroMode = -1);
	//static void* ToDevice(TGPUUsageArg* arg, void* hostPtr, size_t size, bool dontCopy = false); //HG26072024

	/**
	* Retrieve the host memory address for a given device or host pointer
	* @param [in] arg pointer to a GPU usage argument structure
	* @param [in] devicePtr pointer for which the host pointer is desired
	* @return the corresponding host pointer, NULL on errror
	*/
	static void* GetHostPtr(TGPUUsageArg* arg, void* devicePtr);

	/**
	* Transfer memory back to the host if necessary and free the associated device memory. Ensures that the latest copy of the data is on the host by the end.
	* @param [in] arg pointer to a GPU usage argument structure
	* @param [in] devicePtr device pointer to the memory to be freed, if a host pointer is provided, the corresponding device pointer is freed
	* @param [in] size size of the block to be freed
	* @param [in] dontCopy do not copy the device memory to host if true
	* @return The corresponding host pointer, NULL on error
	*/
	static void* ToHostAndFree(TGPUUsageArg* arg, void* devicePtr, size_t size, bool dontCopy = false);

	/**
	* Ensure that the device memory has the latest data, used prior to kernel launches
	* @param [in] arg pointer to a GPU usage argument structure
	* @param [in] devicePtr device pointer to the memory block to be operated on
	*/
	static void EnsureDeviceMemoryReady(TGPUUsageArg* arg, void* devicePtr);

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
	* @param [in] devToHost true if device memory has the latest version of the data. Cannot be true if hostToDev is true
	* @param [in] hostToDev true if host memory has the latest version of the data. Cannot be true if devToHost is true
	*/
	static void MarkUpdated(TGPUUsageArg* arg, void* ptr, bool devToHost, bool hostToDev);
};

//*************************************************************************
#endif