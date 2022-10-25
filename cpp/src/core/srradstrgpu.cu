#ifdef _OFFLOAD_GPU
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math_constants.h"
#include <stdio.h>
#include <iostream>
#include <chrono>
#include "srradstrgpu.h"


__global__ void MultiplyElFieldByPhaseLin_Kernel(double xMult, double zMult, float* pBaseRadX, float* pBaseRadZ, int nWfr, int nz, int nx, int ne, float zStart, float zStep, float xStart, float xStep) {
    int ix = (blockIdx.x * blockDim.x + threadIdx.x); //nx range
    int iz = (blockIdx.y * blockDim.y + threadIdx.y); //nz range
    int iwfr = (blockIdx.z * blockDim.z + threadIdx.z); //nWfr range

    if (ix < nx && iz < nz && iwfr < nWfr) 
    {
		bool RadXisDefined = (pBaseRadX != 0);
		bool RadZisDefined = (pBaseRadZ != 0);

		double z = zStart + iz * zStep;
		double x = xStart + ix * xStep;
		double dPhZ = zMult * z;
		double dPh = dPhZ + xMult * x;
		double cosPh, sinPh;
		sincos(dPh, &sinPh, &cosPh);

		long long offset = iwfr * nz * nx * ne * 2 + iz * nx * ne * 2 + ix * ne * 2;
		float* tEx = pBaseRadX + offset;
		float* tEz = pBaseRadZ + offset;
		for (int ie = 0; ie < ne; ie++)
		{
			if (RadXisDefined)
			{
				//*(tEx++) *= a; *(tEx++) *= a;
				double newReEx = (*tEx) * cosPh - (*(tEx + 1)) * sinPh;
				double newImEx = (*tEx) * sinPh + (*(tEx + 1)) * cosPh;
				*(tEx++) = (float)newReEx; *(tEx++) = (float)newImEx;
			}
			if (RadZisDefined)
			{
				//*(tEz++) *= a; *(tEz++) *= a;
				double newReEz = (*tEz) * cosPh - (*(tEz + 1)) * sinPh;
				double newImEz = (*tEz) * sinPh + (*(tEz + 1)) * cosPh;
				*(tEz++) = (float)newReEz; *(tEz++) = (float)newImEz;
			}
		}
    }
}

void MultiplyElFieldByPhaseLin_GPU(double xMult, double zMult, float* pBaseRadX, float* pBaseRadZ, int nWfr, int nz, int nx, int ne, float zStart, float zStep, float xStart, float xStep, gpuUsageArg_t* pGpuUsage)
{
	if (pBaseRadX != NULL)
		pBaseRadX = (float*)UtiDev::ToDevice(pGpuUsage, pBaseRadX, nWfr * nz * nx * ne * 2 * sizeof(float));
	if (pBaseRadZ != NULL)
		pBaseRadZ = (float*)UtiDev::ToDevice(pGpuUsage, pBaseRadZ, nWfr * nz * nx * ne * 2 * sizeof(float));

    const int bs = 256;
    dim3 blocks(nx / bs + ((nx & (bs - 1)) != 0), nz, nWfr);
    dim3 threads(bs, 1);
    MultiplyElFieldByPhaseLin_Kernel<< <blocks, threads >> > (xMult, zMult, pBaseRadX, pBaseRadZ, nWfr, nz, nx, ne, zStart, zStep, xStart, xStep);

	if (pBaseRadX != NULL)
		UtiDev::MarkUpdated(pGpuUsage, pBaseRadX, true, false);
	if (pBaseRadZ != NULL)
		UtiDev::MarkUpdated(pGpuUsage, pBaseRadZ, true, false);

	    

#ifdef _DEBUG
	if (pBaseRadX != NULL)
		pBaseRadX = (float*)UtiDev::ToHostAndFree(pGpuUsage, pBaseRadX, nWfr * nz * nx * ne * 2 * sizeof(float));
	if (pBaseRadZ != NULL)
		pBaseRadZ = (float*)UtiDev::ToHostAndFree(pGpuUsage, pBaseRadZ, nWfr * nz * nx * ne * 2 * sizeof(float));
	cudaStreamSynchronize(0);
    auto err = cudaGetLastError();
    printf("%s\r\n", cudaGetErrorString(err));
#endif
}

#endif