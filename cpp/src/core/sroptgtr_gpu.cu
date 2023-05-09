/************************************************************************//**
 * File: sroptgtr_gpu.cu
 * Description: Optical element: Transmission (CUDA implementation)
 * Project: Synchrotron Radiation Workshop
 * First release: 2023
 *
 * Copyright (C) Brookhaven National Laboratory
 * All Rights Reserved
 *
 * @author H.Goel
 * @version 1.0
 ***************************************************************************/

#include "sroptgtr.h"
#ifdef _OFFLOAD_GPU
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math_constants.h"

#include <stdio.h>
#include <iostream>
#include <chrono>

int srTGenTransmission::RadPointModifierParallel(srTSRWRadStructAccessData* pRadAccessData, void* pBufVars, long pBufVarsSz, gpuUsageArg *pGpuUsage)
{
    GenTransNumData.pData = (char*)AuxGpu::ToDevice(pGpuUsage, GenTransNumData.pData, GenTransNumData.DimSizes[0] * (int)GenTransNumData.DimSizes[1] * (int)GenTransNumData.DimSizes[2] * sizeof(double) * 2);
	AuxGpu::EnsureDeviceMemoryReady(pGpuUsage, GenTransNumData.pData);
    int retCode = RadPointModifierParallelImpl<srTGenTransmission>(pRadAccessData, pBufVars, pBufVarsSz, this, pGpuUsage); 
	GenTransNumData.pData = (char*)AuxGpu::ToHostAndFree(pGpuUsage, GenTransNumData.pData, GenTransNumData.DimSizes[0] * (int)GenTransNumData.DimSizes[1] * (int)GenTransNumData.DimSizes[2] * sizeof(double) * 2, true);
    return retCode;
} //HG03092022
#endif