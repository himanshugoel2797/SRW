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
	{
		pBaseRadX = (float*)UtiDev::ToDevice(pGpuUsage, pBaseRadX, nWfr * nz * nx * ne * 2 * sizeof(float));
		UtiDev::EnsureDeviceMemoryReady(pGpuUsage, pBaseRadX);
	}
	if (pBaseRadZ != NULL)
	{
		pBaseRadZ = (float*)UtiDev::ToDevice(pGpuUsage, pBaseRadZ, nWfr * nz * nx * ne * 2 * sizeof(float));
		UtiDev::EnsureDeviceMemoryReady(pGpuUsage, pBaseRadZ);
	}

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

template<int mode> __global__ void MirrorFieldData_Kernel(long ne, long nx, long nz, long nwfr, float* pEX0, float* pEZ0) {
	int ix = (blockIdx.x * blockDim.x + threadIdx.x); //nx range
	int iz = (blockIdx.y * blockDim.y + threadIdx.y); //nz range
	int iwfr = (blockIdx.z * blockDim.z + threadIdx.z); //nWfr range

	if (ix < nx && iz < nz && iwfr < nwfr)
	{
		long long PerX = ne << 1;
		long long PerZ = PerX * nx;
		float buf;

		if (pEX0 != 0) pEX0 += ne * nx * nz * 2 * iwfr;
		if (pEZ0 != 0) pEZ0 += ne * nx * nz * 2 * iwfr;

		if (mode == 0)
		{
			if (ix >= (nx >> 1))
				return;

			long long nx_mi_1 = nx - 1; //OC26042019
			for (long long ie = 0; ie < ne; ie++)
			{
				//long Two_ie = ie << 1;
				long long Two_ie = ie << 1; //OC26042019
				
				//long izPerZ = iz*PerZ;
				long long izPerZ = iz * PerZ;
				float* pEX_StartForX = pEX0 + izPerZ;
				float* pEZ_StartForX = pEZ0 + izPerZ;

				//long ixPerX_p_Two_ie = ix*PerX + Two_ie;
				long long ixPerX_p_Two_ie = ix * PerX + Two_ie;
				float* pEX = pEX_StartForX + ixPerX_p_Two_ie;
				float* pEZ = pEZ_StartForX + ixPerX_p_Two_ie;

				//long rev_ixPerX_p_Two_ie = (nx_mi_1 - ix)*PerX + Two_ie;
				long long rev_ixPerX_p_Two_ie = (nx_mi_1 - ix) * PerX + Two_ie;
				float* rev_pEX = pEX_StartForX + rev_ixPerX_p_Two_ie;
				float* rev_pEZ = pEZ_StartForX + rev_ixPerX_p_Two_ie;

				if (pEX0 != 0)
				{
					buf = *rev_pEX; *(rev_pEX++) = *pEX; *(pEX++) = buf;
					buf = *rev_pEX; *rev_pEX = *pEX; *pEX = buf;
				}
				if (pEZ0 != 0)
				{
					buf = *rev_pEZ; *(rev_pEZ++) = *pEZ; *(pEZ++) = buf;
					buf = *rev_pEZ; *rev_pEZ = *pEZ; *pEZ = buf;
				}
			}
		}
		else if (mode == 1)
		{
			if (iz >= (nz >> 1))
				return;

			long long nz_mi_1 = nz - 1; //OC26042019
			for (long long ie = 0; ie < ne; ie++)
			{
				//long Two_ie = ie << 1;
				long long Two_ie = ie << 1;
				
				//long izPerZ = iz*PerZ;
				long long izPerZ = iz * PerZ;
				float* pEX_StartForX = pEX0 + izPerZ;
				float* pEZ_StartForX = pEZ0 + izPerZ;

				//long rev_izPerZ = (nz_mi_1 - iz)*PerZ;
				long long rev_izPerZ = (nz_mi_1 - iz) * PerZ;
				float* rev_pEX_StartForX = pEX0 + rev_izPerZ;
				float* rev_pEZ_StartForX = pEZ0 + rev_izPerZ;

				//long ixPerX_p_Two_ie = ix*PerX + Two_ie;
				long long ixPerX_p_Two_ie = ix * PerX + Two_ie;
				float* pEX = pEX_StartForX + ixPerX_p_Two_ie;
				float* pEZ = pEZ_StartForX + ixPerX_p_Two_ie;

				float* rev_pEX = rev_pEX_StartForX + ixPerX_p_Two_ie;
				float* rev_pEZ = rev_pEZ_StartForX + ixPerX_p_Two_ie;

				if (pEX0 != 0)
				{
					buf = *rev_pEX; *(rev_pEX++) = *pEX; *(pEX++) = buf;
					buf = *rev_pEX; *rev_pEX = *pEX; *pEX = buf;
				}
				if (pEZ0 != 0)
				{
					buf = *rev_pEZ; *(rev_pEZ++) = *pEZ; *(pEZ++) = buf;
					buf = *rev_pEZ; *rev_pEZ = *pEZ; *pEZ = buf;
				}
			}
		}
		else if (mode == 2)
		{
			if (iz >= (nz >> 1))
				return;

			long long nx_mi_1 = nx - 1; //OC26042019
			long long nz_mi_1 = nz - 1;
			for (long long ie = 0; ie < ne; ie++) //OC26042019
				//for(long ie=0; ie<ne; ie++)
			{
				//long Two_ie = ie << 1;
				//for(long iz=0; iz<(nz >> 1); iz++)
				long long Two_ie = ie << 1; //OC26042019
				
				//long izPerZ = iz*PerZ;
				long long izPerZ = iz * PerZ;
				float* pEX_StartForX = pEX0 + izPerZ;
				float* pEZ_StartForX = pEZ0 + izPerZ;

				//long rev_izPerZ = (nz_mi_1 - iz)*PerZ;
				long long rev_izPerZ = (nz_mi_1 - iz) * PerZ;
				float* rev_pEX_StartForX = pEX0 + rev_izPerZ;
				float* rev_pEZ_StartForX = pEZ0 + rev_izPerZ;

				//long ixPerX_p_Two_ie = ix*PerX + Two_ie;
				long long ixPerX_p_Two_ie = ix * PerX + Two_ie;
				float* pEX = pEX_StartForX + ixPerX_p_Two_ie;
				float* pEZ = pEZ_StartForX + ixPerX_p_Two_ie;

				//long rev_ixPerX_p_Two_ie = (nx_mi_1 - ix)*PerX + Two_ie;
				long long rev_ixPerX_p_Two_ie = (nx_mi_1 - ix) * PerX + Two_ie;
				float* rev_pEX = rev_pEX_StartForX + rev_ixPerX_p_Two_ie;
				float* rev_pEZ = rev_pEZ_StartForX + rev_ixPerX_p_Two_ie;

				if (pEX0 != 0)
				{
					buf = *rev_pEX; *(rev_pEX++) = *pEX; *(pEX++) = buf;
					buf = *rev_pEX; *rev_pEX = *pEX; *pEX = buf;
				}
				if (pEZ0 != 0)
				{
					buf = *rev_pEZ; *(rev_pEZ++) = *pEZ; *(pEZ++) = buf;
					buf = *rev_pEZ; *rev_pEZ = *pEZ; *pEZ = buf;
				}

				if (((nz >> 1) << 1) != nz)
				{
					//long izPerZ = ((nz >> 1) + 1)*PerZ;
					long long izPerZ = ((nz >> 1) + 1) * PerZ;
					float* pEX_StartForX = pEX0 + izPerZ;
					float* pEZ_StartForX = pEZ0 + izPerZ;

					//long ixPerX_p_Two_ie = ix*PerX + Two_ie;
					long long ixPerX_p_Two_ie = ix * PerX + Two_ie;
					float* pEX = pEX_StartForX + ixPerX_p_Two_ie;
					float* pEZ = pEZ_StartForX + ixPerX_p_Two_ie;

					//long rev_ixPerX_p_Two_ie = (nx_mi_1 - ix)*PerX + Two_ie;
					long long rev_ixPerX_p_Two_ie = (nx_mi_1 - ix) * PerX + Two_ie;
					float* rev_pEX = pEX_StartForX + rev_ixPerX_p_Two_ie;
					float* rev_pEZ = pEZ_StartForX + rev_ixPerX_p_Two_ie;

					if (pEX0 != 0)
					{
						buf = *rev_pEX; *(rev_pEX++) = *pEX; *(pEX++) = buf;
						buf = *rev_pEX; *rev_pEX = *pEX; *pEX = buf;
					}
					if (pEZ0 != 0)
					{
						buf = *rev_pEZ; *(rev_pEZ++) = *pEZ; *(pEZ++) = buf;
						buf = *rev_pEZ; *rev_pEZ = *pEZ; *pEZ = buf;
					}
				}
			}
		}
	}
}

void MirrorFieldData_GPU(long ne, long nx, long nz, long nwfr, float* pEX0, float* pEZ0, int mode, gpuUsageArg_t* pGpuUsage)
{
	if (pEX0 != NULL)
	{
		pEX0 = (float*)UtiDev::ToDevice(pGpuUsage, pEX0, nwfr * nz * nx * ne * 2 * sizeof(float));
		UtiDev::EnsureDeviceMemoryReady(pGpuUsage, pEX0);
	}
	if (pEZ0 != NULL)
	{
		pEZ0 = (float*)UtiDev::ToDevice(pGpuUsage, pEZ0, nwfr * nz * nx * ne * 2 * sizeof(float));
		UtiDev::EnsureDeviceMemoryReady(pGpuUsage, pEZ0);
	}

	const int bs = 256;
	dim3 blocks(nx / bs + ((nx & (bs - 1)) != 0), nz, nwfr);
	dim3 threads(bs, 1);
	switch (mode)
	{
	case 0:
		MirrorFieldData_Kernel<0> <<<blocks, threads>>>(ne, nx, nz, nwfr, pEX0, pEZ0);
		break;
	case 1:
		MirrorFieldData_Kernel<1> <<<blocks, threads >>> (ne, nx, nz, nwfr, pEX0, pEZ0);
		break;
	case 2:
		MirrorFieldData_Kernel<2> <<<blocks, threads >>> (ne, nx, nz, nwfr, pEX0, pEZ0);
		break;
	}


	if (pEX0 != NULL)
		UtiDev::MarkUpdated(pGpuUsage, pEX0, true, false);
	if (pEZ0 != NULL)
		UtiDev::MarkUpdated(pGpuUsage, pEZ0, true, false);

#ifdef _DEBUG
	if (pEX0 != NULL)
		pEX0 = (float*)UtiDev::ToHostAndFree(pGpuUsage, pEX0, nwfr * nz * nx * ne * 2 * sizeof(float));
	if (pEZ0 != NULL)
		pEZ0 = (float*)UtiDev::ToHostAndFree(pGpuUsage, pEZ0, nwfr * nz * nx * ne * 2 * sizeof(float));
	cudaStreamSynchronize(0);
	auto err = cudaGetLastError();
	printf("%s\r\n", cudaGetErrorString(err));
#endif
}

#endif