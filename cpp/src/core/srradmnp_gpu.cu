/************************************************************************//**
 * File: srradmnp_gpu.cu
 * Description: Various "manipulations" with Radiation data (e.g. "extraction" of Intensity from Electric Field, etc.) (CUDA implementation)
 * Project: Synchrotron Radiation Workshop
 * First release: 2023
 *
 * Copyright (C) Brookhaven National Laboratory
 * All Rights Reserved
 *
 * @author H.Goel
 * @version 1.0
 ***************************************************************************/

#ifdef _OFFLOAD_GPU
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <assert.h>
#include <math.h>
#include "srradmnp.h"
#include "gmmeth.h"

template <bool allStokesReq, bool intOverEnIsRequired, int PolCom>
__global__ void ExtractSingleElecIntensity2DvsXZ_Kernel(srTRadExtract RadExtract, srTSRWRadStructAccessData RadAccessData, srTRadGenManip *obj, double* arAuxInt, long long ie0, long long ie1, double InvStepRelArg, int Int_or_ReE)
{
	int ix = (blockIdx.x * blockDim.x + threadIdx.x); //nx range
    int iz = (blockIdx.y * blockDim.y + threadIdx.y); //nz range
    int iwfr = (blockIdx.z * blockDim.z + threadIdx.z); //nwfr range
    
	if (ix < RadAccessData.nx && iz < RadAccessData.nz && iwfr < RadAccessData.nwfr) 
    {
		//int PolCom = RadExtract.PolarizCompon;
			
		//bool allStokesReq = (PolCom == -5); //OC18042020

		float* pI = 0, * pI1 = 0, * pI2 = 0, * pI3 = 0; //OC17042020
		double* pId = 0, * pI1d = 0, * pI2d = 0, * pI3d = 0;
		long ne = RadAccessData.ne, nx = RadAccessData.nx, nz = RadAccessData.nz, nwfr = RadAccessData.nwfr;
		//float *pI = 0;
		//DOUBLE *pId = 0;
		//double *pId = 0; //OC26112019 (related to SRW port to IGOR XOP8 on Mac)
		long long nxnz = ((long long)nx) * ((long long)nz);
		if (Int_or_ReE != 2)
		{
			pI = RadExtract.pExtractedData;
			if (allStokesReq) //OC17042020
			{
				pI1 = pI + nxnz; pI2 = pI1 + nxnz; pI3 = pI2 + nxnz;
			}
		}
		else
		{
			pId = RadExtract.pExtractedDataD;
			if (allStokesReq) //OC17042020
			{
				pI1d = pId + nxnz; pI2d = pI1d + nxnz; pI3d = pI2d + nxnz;
			}
		}

		float* pEx0 = RadAccessData.pBaseRadX;
		float* pEz0 = RadAccessData.pBaseRadZ;

		//long PerX = RadAccessData.ne << 1;
		//long PerZ = PerX*RadAccessData.nx;
		//long long PerX = RadAccessData.ne << 1;
		//long long PerZ = PerX*RadAccessData.nx;
		long long PerX = ((long long)ne) << 1; //OC18042020
		long long PerZ = PerX * nx;
		long long PerWfr = PerZ * nz;

		//bool intOverEnIsRequired = (RadExtract.Int_or_Phase == 7) && (ne > 1); //OC18042020
		double resInt, resInt1, resInt2, resInt3;
		double ConstPhotEnInteg = 1.;
		long long Two_ie0 = ie0 << 1, Two_ie1 = ie1 << 1; //OC26042019
		long ie;

		long offset = iwfr * PerWfr + iz * PerZ + ix * PerX;
		long offsetDiv2 = offset >> 1;

		float* pEx_StartForX = pEx0 + offset;
		float* pEz_StartForX = pEz0 + offset;
		if (pI != 0)
		{
			pI += offsetDiv2;
			if (allStokesReq)
			{
				pI1 += offsetDiv2;
				pI2 += offsetDiv2;
				pI3 += offsetDiv2;
			}
		} 

		if (pId != 0)
		{
			pId += offsetDiv2;
			if (allStokesReq)
			{
				pI1d += offsetDiv2;
				pI2d += offsetDiv2;
				pI3d += offsetDiv2;
			}
		} 
		
		//long ixPerX = 0;

		float* pEx_St = pEx_StartForX + Two_ie0;
		float* pEz_St = pEz_StartForX + Two_ie0;
		float* pEx_Fi = pEx_StartForX + Two_ie1;
		float* pEz_Fi = pEz_StartForX + Two_ie1;

		if (intOverEnIsRequired) //OC140813
		{//integrate over photon energy / time
			double* tInt = arAuxInt;
			float* pEx_StAux = pEx_St;
			float* pEz_StAux = pEz_St;

			if (!allStokesReq) //OC17042020
			{
				for (ie = 0; ie < ne; ie++) //OC18042020
				//for(int ie=0; ie<RadAccessData.ne; ie++)
				{
					*(tInt++) = obj->IntensityComponent(pEx_StAux, pEz_StAux, PolCom, Int_or_ReE);
					pEx_StAux += 2;
					pEz_StAux += 2;
				}
				resInt = ConstPhotEnInteg * CGenMathMeth::Integ1D_FuncDefByArray(arAuxInt, ne, RadAccessData.eStep); //OC18042020
				//resInt = ConstPhotEnInteg*CGenMathMeth::Integ1D_FuncDefByArray(arAuxInt, RadAccessData.ne, RadAccessData.eStep);
			}
			else
			{
				for (ie = 0; ie < ne; ie++)
				{
					*(tInt++) = obj->IntensityComponent(pEx_StAux, pEz_StAux, -1, Int_or_ReE);
					pEx_StAux += 2; pEz_StAux += 2;
				}
				resInt = ConstPhotEnInteg * CGenMathMeth::Integ1D_FuncDefByArray(arAuxInt, ne, RadAccessData.eStep);

				tInt = arAuxInt; pEx_StAux = pEx_St; pEz_StAux = pEz_St;
				for (ie = 0; ie < ne; ie++)
				{
					*(tInt++) = obj->IntensityComponent(pEx_StAux, pEz_StAux, -2, Int_or_ReE);
					pEx_StAux += 2; pEz_StAux += 2;
				}
				resInt1 = ConstPhotEnInteg * CGenMathMeth::Integ1D_FuncDefByArray(arAuxInt, ne, RadAccessData.eStep);

				tInt = arAuxInt; pEx_StAux = pEx_St; pEz_StAux = pEz_St;
				for (ie = 0; ie < ne; ie++)
				{
					*(tInt++) = obj->IntensityComponent(pEx_StAux, pEz_StAux, -3, Int_or_ReE);
					pEx_StAux += 2; pEz_StAux += 2;
				}
				resInt2 = ConstPhotEnInteg * CGenMathMeth::Integ1D_FuncDefByArray(arAuxInt, ne, RadAccessData.eStep);

				tInt = arAuxInt; pEx_StAux = pEx_St; pEz_StAux = pEz_St;
				for (ie = 0; ie < ne; ie++)
				{
					*(tInt++) = obj->IntensityComponent(pEx_StAux, pEz_StAux, -4, Int_or_ReE);
					pEx_StAux += 2; pEz_StAux += 2;
				}
				resInt3 = ConstPhotEnInteg * CGenMathMeth::Integ1D_FuncDefByArray(arAuxInt, ne, RadAccessData.eStep);
			}
		}
		else
		{
			if (!allStokesReq) //OC18042020
			{
				resInt = obj->IntensityComponentSimpleInterpol(pEx_St, pEx_Fi, pEz_St, pEz_Fi, InvStepRelArg, PolCom, Int_or_ReE);
			}
			else //OC18042020
			{
				resInt = obj->IntensityComponentSimpleInterpol(pEx_St, pEx_Fi, pEz_St, pEz_Fi, InvStepRelArg, -1, Int_or_ReE);
				resInt1 = obj->IntensityComponentSimpleInterpol(pEx_St, pEx_Fi, pEz_St, pEz_Fi, InvStepRelArg, -2, Int_or_ReE);
				resInt2 = obj->IntensityComponentSimpleInterpol(pEx_St, pEx_Fi, pEz_St, pEz_Fi, InvStepRelArg, -3, Int_or_ReE);
				resInt3 = obj->IntensityComponentSimpleInterpol(pEx_St, pEx_Fi, pEz_St, pEz_Fi, InvStepRelArg, -4, Int_or_ReE);
			}
		}
		//OC140813
		if (pI != 0) *pI = (float)resInt;
		if (pId != 0) *pId = resInt; //OC18042020
		//if(pId != 0) *(pId++) = (double)resInt;
		if (allStokesReq) //OC18042020
		{
			if (RadExtract.pExtractedData != 0)
			{
				*pI1 = (float)resInt1; *pI2 = (float)resInt2; *pI3 = (float)resInt3;
			}
			else
			{
				*pI1d = resInt1; *pI2d = resInt2; *pI3d = resInt3;
			}
		}
	}
}

template <bool allStokesReq, bool intOverEnIsRequired>
static inline void KernelLauncher(dim3 &blocks, dim3 &threads, srTRadExtract RadExtract, srTSRWRadStructAccessData RadAccessData, srTRadGenManip *local_copy, double* arAuxInt, long long ie0, long long ie1, double InvStepRelArg, int Int_or_ReE)
{
	switch(RadExtract.PolarizCompon)
	{
		case 5: ExtractSingleElecIntensity2DvsXZ_Kernel<allStokesReq, intOverEnIsRequired, 5><<<blocks, threads>>>(RadExtract, RadAccessData, local_copy, arAuxInt, ie0, ie1, InvStepRelArg, Int_or_ReE); break;
		case 4: ExtractSingleElecIntensity2DvsXZ_Kernel<allStokesReq, intOverEnIsRequired, 4><<<blocks, threads>>>(RadExtract, RadAccessData, local_copy, arAuxInt, ie0, ie1, InvStepRelArg, Int_or_ReE); break;
		case 3: ExtractSingleElecIntensity2DvsXZ_Kernel<allStokesReq, intOverEnIsRequired, 3><<<blocks, threads>>>(RadExtract, RadAccessData, local_copy, arAuxInt, ie0, ie1, InvStepRelArg, Int_or_ReE); break;
		case 2: ExtractSingleElecIntensity2DvsXZ_Kernel<allStokesReq, intOverEnIsRequired, 2><<<blocks, threads>>>(RadExtract, RadAccessData, local_copy, arAuxInt, ie0, ie1, InvStepRelArg, Int_or_ReE); break;
		case 1: ExtractSingleElecIntensity2DvsXZ_Kernel<allStokesReq, intOverEnIsRequired, 1><<<blocks, threads>>>(RadExtract, RadAccessData, local_copy, arAuxInt, ie0, ie1, InvStepRelArg, Int_or_ReE); break;
		case 0: ExtractSingleElecIntensity2DvsXZ_Kernel<allStokesReq, intOverEnIsRequired, 0><<<blocks, threads>>>(RadExtract, RadAccessData, local_copy, arAuxInt, ie0, ie1, InvStepRelArg, Int_or_ReE); break;
		case -1: ExtractSingleElecIntensity2DvsXZ_Kernel<allStokesReq, intOverEnIsRequired, -1><<<blocks, threads>>>(RadExtract, RadAccessData, local_copy, arAuxInt, ie0, ie1, InvStepRelArg, Int_or_ReE); break;
		case -2: ExtractSingleElecIntensity2DvsXZ_Kernel<allStokesReq, intOverEnIsRequired, -2><<<blocks, threads>>>(RadExtract, RadAccessData, local_copy, arAuxInt, ie0, ie1, InvStepRelArg, Int_or_ReE); break;
		case -3: ExtractSingleElecIntensity2DvsXZ_Kernel<allStokesReq, intOverEnIsRequired, -3><<<blocks, threads>>>(RadExtract, RadAccessData, local_copy, arAuxInt, ie0, ie1, InvStepRelArg, Int_or_ReE); break;
		case -4: ExtractSingleElecIntensity2DvsXZ_Kernel<allStokesReq, intOverEnIsRequired, -4><<<blocks, threads>>>(RadExtract, RadAccessData, local_copy, arAuxInt, ie0, ie1, InvStepRelArg, Int_or_ReE); break;
		default: ExtractSingleElecIntensity2DvsXZ_Kernel<allStokesReq, intOverEnIsRequired, -5><<<blocks, threads>>>(RadExtract, RadAccessData, local_copy, arAuxInt, ie0, ie1, InvStepRelArg, Int_or_ReE); break;
	}
}

int srTRadGenManip::ExtractSingleElecIntensity2DvsXZ_GPU(srTRadExtract& RadExtract, double* arAuxInt, long long ie0, long long ie1, double InvStepRelArg, gpuUsageArg *pGpuUsage)
{
	srTSRWRadStructAccessData& RadAccessData = *((srTSRWRadStructAccessData*)(hRadAccessData.ptr()));

    const int bs = 256;
    dim3 blocks(RadAccessData.nx / bs + ((RadAccessData.nx & (bs - 1)) != 0), RadAccessData.nz, RadAccessData.nwfr);
    dim3 threads(bs, 1);

    if (RadAccessData.pBaseRadX != NULL)
	{
		RadAccessData.pBaseRadX = (float*)AuxGpu::ToDevice(pGpuUsage, RadAccessData.pBaseRadX, 2*RadAccessData.ne*RadAccessData.nx*RadAccessData.nz*RadAccessData.nwfr*sizeof(float));
		AuxGpu::EnsureDeviceMemoryReady(pGpuUsage, RadAccessData.pBaseRadX);
	}
	if (RadAccessData.pBaseRadZ != NULL)
	{
		RadAccessData.pBaseRadZ = (float*)AuxGpu::ToDevice(pGpuUsage, RadAccessData.pBaseRadZ, 2*RadAccessData.ne*RadAccessData.nx*RadAccessData.nz*RadAccessData.nwfr*sizeof(float));
		AuxGpu::EnsureDeviceMemoryReady(pGpuUsage, RadAccessData.pBaseRadZ);
	}

	srTRadGenManip *local_copy = (srTRadGenManip*)AuxGpu::ToDevice(pGpuUsage, this, sizeof(srTRadGenManip));
	AuxGpu::EnsureDeviceMemoryReady(pGpuUsage, local_copy);

    arAuxInt = (double*)AuxGpu::ToDevice(pGpuUsage, arAuxInt, RadAccessData.ne*sizeof(double));
    AuxGpu::EnsureDeviceMemoryReady(pGpuUsage, arAuxInt);

	bool allStokesReq = (RadExtract.PolarizCompon == -5);
	bool intOverEnIsRequired = (RadExtract.Int_or_Phase == 7) && (RadAccessData.ne > 1);

	int Int_or_ReE = RadExtract.Int_or_Phase;
	if (Int_or_ReE == 7) Int_or_ReE = 0; //OC150813: time/phot. energy integrated single-e intensity requires "normal" intensity here

	if (allStokesReq)
		if (intOverEnIsRequired)
    		KernelLauncher<true, true> (blocks, threads, RadExtract, RadAccessData, local_copy, arAuxInt, ie0, ie1, InvStepRelArg, Int_or_ReE);
		else
    		KernelLauncher<true, false> (blocks, threads, RadExtract, RadAccessData, local_copy, arAuxInt, ie0, ie1, InvStepRelArg, Int_or_ReE);
	else
		if (intOverEnIsRequired)
    		KernelLauncher<false, true> (blocks, threads, RadExtract, RadAccessData, local_copy, arAuxInt, ie0, ie1, InvStepRelArg, Int_or_ReE);
		else
    		KernelLauncher<false, false> (blocks, threads, RadExtract, RadAccessData, local_copy, arAuxInt, ie0, ie1, InvStepRelArg, Int_or_ReE);
	
    AuxGpu::ToHostAndFree(pGpuUsage, local_copy, sizeof(srTRadGenManip), true);
    AuxGpu::ToHostAndFree(pGpuUsage, arAuxInt, RadAccessData.ne*sizeof(double), true);
	AuxGpu::MarkUpdated(pGpuUsage, RadAccessData.pBaseRadX, true, false);
	AuxGpu::MarkUpdated(pGpuUsage, RadAccessData.pBaseRadZ, true, false);

#ifndef _DEBUG
	if (RadAccessData.pBaseRadX != NULL)
		RadAccessData.pBaseRadX = (float*)AuxGpu::GetHostPtr(pGpuUsage, RadAccessData.pBaseRadX);
	if (RadAccessData.pBaseRadZ != NULL)
		RadAccessData.pBaseRadZ = (float*)AuxGpu::GetHostPtr(pGpuUsage, RadAccessData.pBaseRadZ);
#endif

#ifdef _DEBUG
	if (RadAccessData.pBaseRadX != NULL)
		RadAccessData.pBaseRadX = (float*)AuxGpu::ToHostAndFree(pGpuUsage, RadAccessData.pBaseRadX, 2 * RadAccessData.ne * RadAccessData.nx * RadAccessData.nz * RadAccessData.nwfr * sizeof(float));
	if (RadAccessData.pBaseRadZ != NULL)
		RadAccessData.pBaseRadZ = (float*)AuxGpu::ToHostAndFree(pGpuUsage, RadAccessData.pBaseRadZ, 2 * RadAccessData.ne * RadAccessData.nx * RadAccessData.nz * RadAccessData.nwfr * sizeof(float));
	cudaStreamSynchronize(0);
	auto err = cudaGetLastError();
	printf("%s\r\n", cudaGetErrorString(err));
#endif
	return 0;
}

#endif