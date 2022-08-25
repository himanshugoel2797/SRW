#ifdef _OFFLOAD_GPU
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math_constants.h"

#include <stdio.h>
#include <iostream>
#include <chrono>
#include "sroptdrf.h"

//int srTDriftSpace::TraverseRadZXE_GPU(srTSRWRadStructAccessData* pRadAccessData, void* pBufVars, long pBufVarsSz) { return RadPointModifierParallelImpl<srTDriftSpace>(pRadAccessData, pBufVars, pBufVarsSz, this); } //HG03092022
#endif