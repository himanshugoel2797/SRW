#ifndef __SRRADSTRGPU_H
#define __SRRADSTRGPU_H
#include "utidev.h"

void MultiplyElFieldByPhaseLin_GPU(double xMult, double zMult, float* pBaseRadX, float* pBaseRadZ, int nWfr, int nz, int nx, int ne, float zStart, float zStep, float xStart, float xStep, gpuUsageArg_t* pGpuUsage);

#endif