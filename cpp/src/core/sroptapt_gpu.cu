/************************************************************************//**
 * File: sroptdrf_gpu.cu
 * Description: Optical element: Aperture (CUDA implementation)
 * Project: Synchrotron Radiation Workshop
 * First release: 2024
 *
 * Copyright (C) Brookhaven National Laboratory
 * All Rights Reserved
 *
 * @author H.Goel
 * @version 1.0
 ***************************************************************************/

#ifdef _OFFLOAD_GPU
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math_constants.h"

#include <stdio.h>
#include <iostream>
#include <chrono>
#include "sroptapt.h"

//Implementation of the RadPointModifier's GPU function for the srTRectAperture class
int srTRectAperture::RadPointModifierParallel(srTSRWRadStructAccessData* pRadAccessData, void* pBufVars, long pBufVarsSz, TGPUUsageArg *pGPU) 
{ 
    return RadPointModifierParallelImpl<srTRectAperture>(pRadAccessData, pBufVars, pBufVarsSz, this, pGPU); 
}

//Implementation of the RadPointModifier's GPU function for the srTRectAperture class
int srTRectObstacle::RadPointModifierParallel(srTSRWRadStructAccessData* pRadAccessData, void* pBufVars, long pBufVarsSz, TGPUUsageArg* pGPU)
{
    return RadPointModifierParallelImpl<srTRectObstacle>(pRadAccessData, pBufVars, pBufVarsSz, this, pGPU);
}

//Implementation of the RadPointModifier's GPU function for the srTRectAperture class
int srTCircAperture::RadPointModifierParallel(srTSRWRadStructAccessData* pRadAccessData, void* pBufVars, long pBufVarsSz, TGPUUsageArg* pGPU)
{
    return RadPointModifierParallelImpl<srTCircAperture>(pRadAccessData, pBufVars, pBufVarsSz, this, pGPU);
}

//Implementation of the RadPointModifier's GPU function for the srTRectAperture class
int srTCircObstacle::RadPointModifierParallel(srTSRWRadStructAccessData* pRadAccessData, void* pBufVars, long pBufVarsSz, TGPUUsageArg* pGPU)
{
    return RadPointModifierParallelImpl<srTCircObstacle>(pRadAccessData, pBufVars, pBufVarsSz, this, pGPU);
}
#endif