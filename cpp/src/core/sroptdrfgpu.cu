#ifdef _OFFLOAD_GPU
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math_constants.h"

#include <stdio.h>
#include <iostream>
#include <chrono>
#include "sroptdrf.h"

int srTDriftSpace::RadPointModifierParallel(srTSRWRadStructAccessData* pRadAccessData, void* pBufVars, long pBufVarsSz, gpuUsageArg_t *pGpuUsage) 
{ 
    return RadPointModifierParallelImpl<srTDriftSpace>(pRadAccessData, pBufVars, pBufVarsSz, this, pGpuUsage); 
} //HG03092022
#endif