/************************************************************************//**
 * File: sroptang_gpu.cu
 * Description: Optical element: Angle (CUDA implementation)
 * Project: Synchrotron Radiation Workshop
 * First release: 2026
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
#include "sroptang.h"

//Implementation of the RadPointModifier's GPU function for the srTOptAngle class
int srTOptAngle::TraverseRadZXEParallel(srTSRWRadStructAccessData* pRadAccessData, void* pBufVars, long pBufVarsSz, TGPUUsageArg *pGPU) //HG20072026
{
	return TraverseRadZXEParallelImpl<srTOptAngle>(pRadAccessData, pBufVars, pBufVarsSz, this, pGPU);
}
#endif
