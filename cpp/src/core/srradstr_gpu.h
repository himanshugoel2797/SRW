/************************************************************************//**
 * File: srradstr_gpu.h
 * Description: Auxiliary structures for various SR calculation methods (CUDA header)
 * Project: Synchrotron Radiation Workshop
 * First release: 2023
 *
 * Copyright (C) Brookhaven National Laboratory
 * All Rights Reserved
 *
 * @author H.Goel
 * @version 1.0
 ***************************************************************************/

#ifndef __SRRADSTRGPU_H
#define __SRRADSTRGPU_H
#include "auxgpu.h"

void MultiplyElFieldByPhaseLin_GPU(double xMult, double zMult, float* pBaseRadX, float* pBaseRadZ, int nWfr, int nz, int nx, int ne, float zStart, float zStep, float xStart, float xStep, gpuUsageArg* pGpuUsage);
void MirrorFieldData_GPU(long ne, long nx, long nz, long nwfr, float* pEX0, float* pEZ0, int mode, gpuUsageArg* pGpuUsage);

#endif