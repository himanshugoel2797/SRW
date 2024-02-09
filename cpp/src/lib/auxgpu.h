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
#include "srwlib.h" //HG06022024

#ifdef _OFFLOAD_GPU
#include <cuda_runtime.h>
#include <map>
//#if CUDART_VERSION < 11020
//#error CUDA version too low, need at least 11.2
//#endif
#endif

//HG07022024 Replace struct with double array
//typedef struct
//{
//	int deviceIndex; // -1 means no device, TODO
//} TGPUUsageArg; 

#ifdef _OFFLOAD_GPU
//#define GPU_COND(arg, code) if (arg && CAuxGPU::GPUEnabled((double*)arg)) { code } //HG07022024
//#define GPU_COND(arg, code) if (arg && CAuxGPU::GPUEnabled((TGPUUsageArg*)arg)) { code }
//#define GPU_COND(arg, code) if (arg && CAuxGPU::GPUEnabled(arg)) { code }
#define GPU_PORTABLE __device__ __host__
#else
//#define GPU_COND(arg, code) if(0) { }
#define GPU_PORTABLE 
#endif

 //*************************************************************************
class CAuxGPU
{
private:
public:
	static void Init();
	
	static void Fini();
	
	static bool GPUAvailable(); //CheckGPUAvailable etc
	
	//static bool GPUEnabled(TGPUUsageArg *arg);
	static bool GPUEnabled(double *arg); //HG07022024

	static void SetGPUStatus(bool enabled);
	
	//static int GetDevice(TGPUUsageArg* arg);
	static int GetDevice(double* arg); //HG07022024

	//static void* ToDevice(TGPUUsageArg* arg, void* hostPtr, size_t size, bool dontCopy = false);
	static void* ToDevice(double* arg, void* hostPtr, size_t size, bool dontCopy = false);  //HG07022024
	
	//static void* GetHostPtr(TGPUUsageArg* arg, void* devicePtr);
	static void* GetHostPtr(double* arg, void* devicePtr);  //HG07022024
	
	//static void* ToHostAndFree(TGPUUsageArg* arg, void* devicePtr, size_t size, bool dontCopy = false);
	static void* ToHostAndFree(double* arg, void* devicePtr, size_t size, bool dontCopy = false);  //HG07022024
	
	//static void EnsureDeviceMemoryReady(TGPUUsageArg* arg, void* devicePtr);
	static void EnsureDeviceMemoryReady(double* arg, void* devicePtr);  //HG07022024
	
	static void FreeHost(void* ptr);
	
	//static void MarkUpdated(TGPUUsageArg* arg, void* ptr, bool devToHost, bool hostToDev);
	static void MarkUpdated(double* arg, void* ptr, bool devToHost, bool hostToDev);  //HG07022024
};

//*************************************************************************
#endif