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
#include <array>
#include <string>
#include <assert.h>
#include <math.h>
#include <cublas_v2.h> //HG20072026 for the rank-K CSD update (cublasCherk)
#include "srradmnp.h"
#include "gmmeth.h"

template<int PolCom, bool NpIsEven>
__device__ double Integ_Intensity(srTRadGenManip *obj, float* pEx, float* pEz, int Int_or_ReE, int ne, double eStep) //HG31072024
{
	double s0 = 0., s1 = 0., s2 = 0., s3 = 0.;

	if (ne == 2)
	{
		s0 = obj->IntensityComponent(pEx, pEz, PolCom, Int_or_ReE);
		s3 = obj->IntensityComponent(pEx + 2, pEz + 2, PolCom, Int_or_ReE);
		return (s0 + s3) * 0.5 * eStep;
	}

	long long NpSim = ne;
	if (NpIsEven) NpSim--;

	float* tpEx = pEx;
	float* tpEz = pEz;

	s0 = obj->IntensityComponent(tpEx, tpEz, PolCom, Int_or_ReE);
	tpEx += 2; tpEz += 2;
	for (long long i = 1; i < ((NpSim - 3) >> 1); i++)
	{
		s1 += obj->IntensityComponent(tpEx, tpEz, PolCom, Int_or_ReE);
		tpEx += 2; tpEz += 2;
		s2 += obj->IntensityComponent(tpEx, tpEz, PolCom, Int_or_ReE);
		tpEx += 2; tpEz += 2;
	}
	s1 += obj->IntensityComponent(tpEx, tpEz, PolCom, Int_or_ReE);
	tpEx += 2; tpEz += 2;

	s3 = obj->IntensityComponent(tpEx, tpEz, PolCom, Int_or_ReE);
	tpEx += 2; tpEz += 2;

	double res = (eStep/3.)*(s0 + 4.*s1 + 2.*s2 + s3);

	if (!NpIsEven)
	{
		double s4 = obj->IntensityComponent(tpEx, tpEz, PolCom, Int_or_ReE);
		
		res += (double)(0.5 * eStep * (s3 + s4));
	}
	return res;
}

//__global__ void ExtractSingleElecIntensity2DvsXZ_Kernel(srTRadExtract RadExtract, srTSRWRadStructAccessData RadAccessData, srTRadGenManip *obj, double* arAuxInt, long long ie0, long long ie1, double InvStepRelArg, int Int_or_ReE)
template <bool allStokesReq, bool intOverEnIsRequired, int PolCom, bool NpIsEven>
__global__ void ExtractSingleElecIntensity2DvsXZ_Kernel(srTRadExtract RadExtract, srTSRWRadStructAccessData* pRadAccessData, srTRadGenManip *obj, long long ie0, long long ie1, double InvStepRelArg, int Int_or_ReE) //HG31072024 Redesigned to handle ne > 1, still needs debugging
{
	int ix = (blockIdx.x * blockDim.x + threadIdx.x); //nx range
    int iz = (blockIdx.y * blockDim.y + threadIdx.y); //nz range
    
	if (ix < pRadAccessData->nx && iz < pRadAccessData->nz) 
    {
		//int PolCom = RadExtract.PolarizCompon;
			
		//bool allStokesReq = (PolCom == -5); //OC18042020

		float* pI = 0, * pI1 = 0, * pI2 = 0, * pI3 = 0; //OC17042020
		double* pId = 0, * pI1d = 0, * pI2d = 0, * pI3d = 0;
		long ne = pRadAccessData->ne, nx = pRadAccessData->nx, nz = pRadAccessData->nz;
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

		float* pEx0 = pRadAccessData->pBaseRadX;
		float* pEz0 = pRadAccessData->pBaseRadZ;

		//long PerX = pRadAccessData->ne << 1;
		//long PerZ = PerX*pRadAccessData->nx;
		//long long PerX = pRadAccessData->ne << 1;
		//long long PerZ = PerX*pRadAccessData->nx;
		long long PerX = ((long long)ne) << 1; //OC18042020
		long long PerZ = PerX * nx;

		//bool intOverEnIsRequired = (RadExtract.Int_or_Phase == 7) && (ne > 1); //OC18042020
		double resInt, resInt1, resInt2, resInt3;
		double ConstPhotEnInteg = 1.;
		long long Two_ie0 = ie0 << 1, Two_ie1 = ie1 << 1; //OC26042019
		
		long offset = iz * PerZ + ix * PerX;
		long offsetExIntens = offset / PerX;

		float* pEx_StartForX = pEx0 + offset;
		float* pEz_StartForX = pEz0 + offset;
		if (pI != 0)
		{
			pI += offsetExIntens;
			if (allStokesReq)
			{
				pI1 += offsetExIntens;
				pI2 += offsetExIntens;
				pI3 += offsetExIntens;
			}
		} 

		if (pId != 0)
		{
			pId += offsetExIntens;
			if (allStokesReq)
			{
				pI1d += offsetExIntens;
				pI2d += offsetExIntens;
				pI3d += offsetExIntens;
			}
		} 
		
		//long ixPerX = 0;

		float* pEx_St = pEx_StartForX + Two_ie0;
		float* pEz_St = pEz_StartForX + Two_ie0;
		float* pEx_Fi = pEx_StartForX + Two_ie1;
		float* pEz_Fi = pEz_StartForX + Two_ie1;

		if (intOverEnIsRequired) //OC140813
		{//integrate over photon energy / time
			//float* pEx_StAux = pEx_St;
			//float* pEz_StAux = pEz_St;

			if (!allStokesReq) //OC17042020
			{
				
				//for (ie = 0; ie < ne; ie++) //OC18042020
				//for(int ie=0; ie<RadAccessData.ne; ie++)
				//{
				//	*(tInt++) = obj->IntensityComponent(pEx_StAux, pEz_StAux, PolCom, Int_or_ReE);
				//	pEx_StAux += 2;
				//	pEz_StAux += 2;
				//}
				//resInt = ConstPhotEnInteg * CGenMathMeth::Integ1D_FuncDefByArray(arAuxInt, ne, RadAccessData.eStep); //OC18042020
				//resInt = ConstPhotEnInteg*CGenMathMeth::Integ1D_FuncDefByArray(arAuxInt, RadAccessData.ne, RadAccessData.eStep);
				resInt = ConstPhotEnInteg * Integ_Intensity<PolCom, NpIsEven>(obj, pEx_St, pEz_St, Int_or_ReE, ne, pRadAccessData->eStep); //HG31072024
			}
			else
			{
				//for (ie = 0; ie < ne; ie++)
				//{
				//	*(tInt++) = obj->IntensityComponent(pEx_StAux, pEz_StAux, -1, Int_or_ReE);
				//	pEx_StAux += 2; pEz_StAux += 2;
				//}
				//resInt = ConstPhotEnInteg * CGenMathMeth::Integ1D_FuncDefByArray(arAuxInt, ne, RadAccessData.eStep);

				//tInt = arAuxInt; pEx_StAux = pEx_St; pEz_StAux = pEz_St;
				//for (ie = 0; ie < ne; ie++)
				//{
				//	*(tInt++) = obj->IntensityComponent(pEx_StAux, pEz_StAux, -2, Int_or_ReE);
				//	pEx_StAux += 2; pEz_StAux += 2;
				//}
				//resInt1 = ConstPhotEnInteg * CGenMathMeth::Integ1D_FuncDefByArray(arAuxInt, ne, RadAccessData.eStep);

				//tInt = arAuxInt; pEx_StAux = pEx_St; pEz_StAux = pEz_St;
				//for (ie = 0; ie < ne; ie++)
				//{
				//	*(tInt++) = obj->IntensityComponent(pEx_StAux, pEz_StAux, -3, Int_or_ReE);
				//	pEx_StAux += 2; pEz_StAux += 2;
				//}
				//resInt2 = ConstPhotEnInteg * CGenMathMeth::Integ1D_FuncDefByArray(arAuxInt, ne, RadAccessData.eStep);

				//tInt = arAuxInt; pEx_StAux = pEx_St; pEz_StAux = pEz_St;
				//for (ie = 0; ie < ne; ie++)
				//{
				//	*(tInt++) = obj->IntensityComponent(pEx_StAux, pEz_StAux, -4, Int_or_ReE);
				//	pEx_StAux += 2; pEz_StAux += 2;
				//}
				//resInt3 = ConstPhotEnInteg * CGenMathMeth::Integ1D_FuncDefByArray(arAuxInt, ne, RadAccessData.eStep);
				resInt = ConstPhotEnInteg * Integ_Intensity<-1, NpIsEven>(obj, pEx_St, pEz_St, Int_or_ReE, ne, pRadAccessData->eStep); //HG31072024
				resInt1 = ConstPhotEnInteg * Integ_Intensity<-2, NpIsEven>(obj, pEx_St, pEz_St, Int_or_ReE, ne, pRadAccessData->eStep);
				resInt2 = ConstPhotEnInteg * Integ_Intensity<-3, NpIsEven>(obj, pEx_St, pEz_St, Int_or_ReE, ne, pRadAccessData->eStep);
				resInt3 = ConstPhotEnInteg * Integ_Intensity<-4, NpIsEven>(obj, pEx_St, pEz_St, Int_or_ReE, ne, pRadAccessData->eStep);
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

//int srTRadGenManip::ExtractSingleElecIntensity2DvsXZ_GPU(srTRadExtract& RadExtract, double* arAuxInt, long long ie0, long long ie1, double InvStepRelArg, TGPUUsageArg* pGPU)
int srTRadGenManip::ExtractSingleElecIntensity2DvsXZ_GPU(srTRadExtract& RadExtract, long long ie0, long long ie1, double InvStepRelArg, TGPUUsageArg* pGPU) //HG31072024
{
#define GEN_MEMBERS(i) \
	ExtractSingleElecIntensity2DvsXZ_Kernel<false, false, i, false>, \
	ExtractSingleElecIntensity2DvsXZ_Kernel<false, false, i, true>, \
	ExtractSingleElecIntensity2DvsXZ_Kernel<false, true,  i, false>, \
	ExtractSingleElecIntensity2DvsXZ_Kernel<false, true,  i, true>, \
	ExtractSingleElecIntensity2DvsXZ_Kernel<true,  false, i, false>, \
	ExtractSingleElecIntensity2DvsXZ_Kernel<true,  false, i, true>, \
	ExtractSingleElecIntensity2DvsXZ_Kernel<true,  true,  i, false>, \
	ExtractSingleElecIntensity2DvsXZ_Kernel<true,  true,  i, true>,

	decltype(ExtractSingleElecIntensity2DvsXZ_Kernel<false, false, 0, false>) *ExtractSingleElecIntensity2DvsXZ_tbl[] = {
		GEN_MEMBERS(-4)
		GEN_MEMBERS(-3)
		GEN_MEMBERS(-2)
		GEN_MEMBERS(-1)
		GEN_MEMBERS(0)
		GEN_MEMBERS(1)
		GEN_MEMBERS(2)
		GEN_MEMBERS(3)
		GEN_MEMBERS(4)
		GEN_MEMBERS(5)
		GEN_MEMBERS(-5)
	};
#undef GEN_MEMBERS

	srTSRWRadStructAccessData& RadAccessData = *((srTSRWRadStructAccessData*)(hRadAccessData.ptr()));

    dim3 blocks(RadAccessData.nx, RadAccessData.nz);
    dim3 threads(1);

    if (RadAccessData.pBaseRadX != NULL)
	{
		RadAccessData.pBaseRadX = CAuxGPU::ToDevice(pGPU, RadAccessData.pBaseRadX, 2*RadAccessData.ne*RadAccessData.nx*RadAccessData.nz, CAuxGPU::DISCARD_HOST);
		CAuxGPU::EnsureDeviceMemoryReady(pGPU, RadAccessData.pBaseRadX);
	}
	if (RadAccessData.pBaseRadZ != NULL)
	{
		RadAccessData.pBaseRadZ = CAuxGPU::ToDevice(pGPU, RadAccessData.pBaseRadZ, 2*RadAccessData.ne*RadAccessData.nx*RadAccessData.nz, CAuxGPU::DISCARD_HOST);
		CAuxGPU::EnsureDeviceMemoryReady(pGPU, RadAccessData.pBaseRadZ);
	}

	srTRadGenManip *local_copy = CAuxGPU::ToDevice(pGPU, this, 1, CAuxGPU::DISCARD_HOST);
	CAuxGPU::EnsureDeviceMemoryReady(pGPU, local_copy);

	srTSRWRadStructAccessData* pRadAccessData_dev = CAuxGPU::ToDevice(pGPU, &RadAccessData, 1, CAuxGPU::DISCARD_HOST);
	CAuxGPU::EnsureDeviceMemoryReady(pGPU, pRadAccessData_dev);

	bool allStokesReq = (RadExtract.PolarizCompon == -5);
	bool intOverEnIsRequired = (RadExtract.Int_or_Phase == 7) && (RadAccessData.ne > 1);

	int Int_or_ReE = RadExtract.Int_or_Phase;
	if (Int_or_ReE == 7) Int_or_ReE = 0; //OC150813: time/phot. energy integrated single-e intensity requires "normal" intensity here
	
	if (Int_or_ReE != 2) //HG13012024 Fixed bug: Output array was not allocated properly
	{
		if (allStokesReq)
		{
			RadExtract.pExtractedData = CAuxGPU::ToDevice(pGPU, RadExtract.pExtractedData, 4*RadAccessData.nx*RadAccessData.nz, CAuxGPU::DONT_COPY);
			CAuxGPU::Memset(pGPU, RadExtract.pExtractedData, 0.0f, 4*RadAccessData.nx*RadAccessData.nz);
		}
		else
		{
			RadExtract.pExtractedData = CAuxGPU::ToDevice(pGPU, RadExtract.pExtractedData, RadAccessData.nx*RadAccessData.nz, CAuxGPU::DONT_COPY);
			CAuxGPU::Memset(pGPU, RadExtract.pExtractedData, 0.0f, RadAccessData.nx*RadAccessData.nz);
		}
		CAuxGPU::EnsureDeviceMemoryReady(pGPU, RadExtract.pExtractedData);
	}
	else
	{
		if (allStokesReq)
		{
			RadExtract.pExtractedDataD = CAuxGPU::ToDevice(pGPU, RadExtract.pExtractedDataD, 4*RadAccessData.nx*RadAccessData.nz, CAuxGPU::DONT_COPY);
			CAuxGPU::Memset(pGPU, RadExtract.pExtractedDataD, 0.0, 4*RadAccessData.nx*RadAccessData.nz);
		}
		else
		{
			RadExtract.pExtractedDataD = CAuxGPU::ToDevice(pGPU, RadExtract.pExtractedDataD, RadAccessData.nx*RadAccessData.nz, CAuxGPU::DONT_COPY);
			CAuxGPU::Memset(pGPU, RadExtract.pExtractedDataD, 0.0, RadAccessData.nx*RadAccessData.nz);
		}
		CAuxGPU::EnsureDeviceMemoryReady(pGPU, RadExtract.pExtractedDataD);
	}

	bool NpIsEven = ((RadAccessData.ne % 2) == 0);
	int idx = RadExtract.PolarizCompon;
	idx = (((idx < -4 || idx > 5) ? 10 : (idx + 4)) << 3) | ((allStokesReq & 1) << 2) | ((intOverEnIsRequired & 1) << 1) | (NpIsEven & 1);
	
	CAuxGPU::CalcLaunchDims(ExtractSingleElecIntensity2DvsXZ_tbl[idx], blocks, blocks, threads);
	ExtractSingleElecIntensity2DvsXZ_tbl[idx]<<<blocks, threads>>>(RadExtract, pRadAccessData_dev, local_copy, ie0, ie1, InvStepRelArg, Int_or_ReE);
	
	if(Int_or_ReE != 2) //HG13012024 Fixed bug: Output array was not allocated properly
	{
		if(RadExtract.pExtractedData != NULL)
		{
			CAuxGPU::MarkUpdated(pGPU, RadExtract.pExtractedData, CAuxGPU::DEVICE);
			RadExtract.pExtractedData = CAuxGPU::GetHostPtr(pGPU, RadExtract.pExtractedData);
		}
	}
	else
	{
		if(RadExtract.pExtractedDataD != NULL)
		{
			CAuxGPU::MarkUpdated(pGPU, RadExtract.pExtractedDataD, CAuxGPU::DEVICE);
			RadExtract.pExtractedDataD = CAuxGPU::GetHostPtr(pGPU, RadExtract.pExtractedDataD);
		}
	}

	CAuxGPU::MarkUpdatedBatch(pGPU, CAuxGPU::DEVICE, RadAccessData.pBaseRadX, RadAccessData.pBaseRadZ, pRadAccessData_dev, local_copy);
	CAuxGPU::ToHostAndFree(pGPU, pRadAccessData_dev); //HG27072024
//HG26022024 (commented out)
//#ifdef _DEBUG
//	if(Int_or_ReE != 2)
//	{
//		if (RadExtract.pExtractedData != NULL)
//			RadExtract.pExtractedData = (float*)CAuxGPU::ToHostAndFree(pGPU, RadExtract.pExtractedData, 2*RadAccessData.ne*RadAccessData.nx*RadAccessData.nz*sizeof(float));
//	}
//	else
//	{
//		if (RadExtract.pExtractedDataD != NULL)
//			RadExtract.pExtractedDataD = (double*)CAuxGPU::ToHostAndFree(pGPU, RadExtract.pExtractedDataD, 2*RadAccessData.ne*RadAccessData.nx*RadAccessData.nz*sizeof(double));
//	}
//#endif

    CAuxGPU::ToHostAndFree(pGPU, local_copy);
	//CAuxGPU::ToHostAndFree(pGPU, arAuxInt, RadAccessData.ne*sizeof(double), true); //HG31072024
    //CAuxGPU::MarkUpdated(pGPU, RadAccessData.pBaseRadX, true, false);
	//CAuxGPU::MarkUpdated(pGPU, RadAccessData.pBaseRadZ, true, false);

	//if (RadAccessData.pBaseRadX != NULL)
	//	RadAccessData.pBaseRadX = (float*)CAuxGPU::ToHostAndFree(pGPU, RadAccessData.pBaseRadX, 2 * RadAccessData.ne * RadAccessData.nx * RadAccessData.nz * sizeof(float), true); //HG13012024 Original wavefront data does not need to be copied back to CPU
	//if (RadAccessData.pBaseRadZ != NULL)
	//	RadAccessData.pBaseRadZ = (float*)CAuxGPU::ToHostAndFree(pGPU, RadAccessData.pBaseRadZ, 2 * RadAccessData.ne * RadAccessData.nx * RadAccessData.nz * sizeof(float), true); //HG13012024 Original wavefront data does not need to be copied back to CPU

	if (RadAccessData.pBaseRadX != NULL)
		RadAccessData.pBaseRadX = CAuxGPU::GetHostPtr(pGPU, RadAccessData.pBaseRadX); //HG13012024 Original wavefront data does not need to be copied back to CPU
	if (RadAccessData.pBaseRadZ != NULL)
		RadAccessData.pBaseRadZ = CAuxGPU::GetHostPtr(pGPU, RadAccessData.pBaseRadZ); //HG13012024 Original wavefront data does not need to be copied back to CPU

//HG26022024 (commented out)
//#ifdef _DEBUG
//	cudaStreamSynchronize(0);
//	auto err = cudaGetLastError();
//	printf("%s\r\n", cudaGetErrorString(err));
//#endif
	return 0;
}

template <int PolCom, bool EhOK, bool EvOK, int gt1_iter, int itPerBlk>
__global__ void ExtractSingleElecMutualIntensityVsXZ_Kernel(const float* __restrict__ pEx0, const float* __restrict__ pEz0, float* __restrict__ pMI0, long nxnz, long itStart, long itEnd, long PerX, long iter0)
{
	//Calculate coordinates as the typical triangular matrix
	int i0 = (blockIdx.x * blockDim.x + threadIdx.x); //<=nxnz range
	int it0_0 = (blockIdx.y * blockDim.y + threadIdx.y); //nxnz/(2*itPerBlk) range
	long iter = iter0;

	if (i0 > nxnz) return;
	//HG20072026 was `it0_0 > nxnz / 2`, which for even nxnz admits it0 == nxnz/2 -- one
	//value too many. The fold below maps rows [nxnz/2, nxnz-1] onto [0, nxnz/2-1], so
	//it0 == nxnz/2 re-covers rows nxnz/2-1 and nxnz/2 that it0 == nxnz/2-1 already
	//covered. Because the update is a read-modify-write running average, those two rows
	//had the electron applied TWICE and came out wrong -- not merely duplicated writes of
	//the same value. Correct bound is (nxnz-1)/2, which also handles odd nxnz.
	if (it0_0 > (nxnz - 1) / 2) return;

	for (int it0 = it0_0 * itPerBlk; it0 < it0_0 * itPerBlk + itPerBlk; it0++)
	{
		long it = it0;
		long i = i0;
		if (i0 > it0) //If the coordinates are past the triangular bounds, switch to the lower half of the triangle
		{
			it = nxnz - it0 - 1;
			i = i0 - (it0 + 1);
			//HG20072026 For ODD nxnz the middle row is its own mirror, so the folded branch
			//would re-cover the row the unfolded branch of this same it0 already did.
			if (it <= it0) return;
		}

		//HG20072026 was `it >= itEnd`, i.e. EXCLUSIVE, while the CPU loop this mirrors is
		//INCLUSIVE: `for(long long it=itStart; it<=itEnd; it++)` (srradmnp.cpp:2161) with
		//itEnd defaulting to nxnz-1 (srradmnp.cpp:1643). The GPU therefore never computed
		//it == nxnz-1 and silently dropped the last row (and, via the Hermitian mirror, the
		//last column) of the CSD -- 2*nxnz-1 elements left at whatever the host buffer held.
		//Invisible on a centred Gaussian test beam, where the field at the mesh edge is ~0;
		//on a real undulator source at 16x16 it was 86% of peak.
		if (it > itEnd) {
			return;
		}

		//float* pMI = pMI0 + it0 * (nxnz << 1) + (i0 << 1); //Compact representation coordinates
		float* pMI = pMI0 + (it - itStart) * (nxnz << 1) + (i << 1); //Full representation coordinates
		const float* pEx = pEx0 + i * PerX;
		const float* pEz = pEz0 + i * PerX;
		const float* pExT = pEx0 + (it - itStart) * PerX;
		const float* pEzT = pEz0 + (it - itStart) * PerX;

		float ExRe = 0., ExIm = 0., EzRe = 0., EzIm = 0.;
		float ExReT = 0., ExImT = 0., EzReT = 0., EzImT = 0.;

		{
			if (EhOK)
			{
				ExRe = *pEx; ExIm = *(pEx + 1);
				if (i != (it - itStart)) {
					ExReT = *pExT; ExImT = *(pExT + 1);
				}
				else {
					ExReT = ExRe;
					ExImT = ExIm;
				}
			}
			if (EvOK) {
				EzRe = *pEz; EzIm = *(pEz + 1);
				if (i != (it - itStart)) {
					EzReT = *pEzT; EzImT = *(pEzT + 1);
				}
				else {
					EzReT = EzRe;
					EzImT = EzIm;
				}
			}
		}
		float ReMI = 0., ImMI = 0.;

		switch (PolCom)
		{
		case 0: // Lin. Hor.
		{
			ReMI = ExRe * ExReT + ExIm * ExImT;
			ImMI = ExIm * ExReT - ExRe * ExImT;
			break;
		}
		case 1: // Lin. Vert.
		{
			ReMI = EzRe * EzReT + EzIm * EzImT;
			ImMI = EzIm * EzReT - EzRe * EzImT;
			break;
		}
		case 2: // Linear 45 deg.
		{
			float ExRe_p_EzRe = ExRe + EzRe, ExIm_p_EzIm = ExIm + EzIm;
			float ExRe_p_EzReT = ExReT + EzReT, ExIm_p_EzImT = ExImT + EzImT;
			ReMI = 0.5f * (ExRe_p_EzRe * ExRe_p_EzReT + ExIm_p_EzIm * ExIm_p_EzImT);
			ImMI = 0.5f * (ExIm_p_EzIm * ExRe_p_EzReT - ExRe_p_EzRe * ExIm_p_EzImT);
			break;
		}
		case 3: // Linear 135 deg.
		{
			float ExRe_mi_EzRe = ExRe - EzRe, ExIm_mi_EzIm = ExIm - EzIm;
			float ExRe_mi_EzReT = ExReT - EzReT, ExIm_mi_EzImT = ExImT - EzImT;
			ReMI = 0.5f * (ExRe_mi_EzRe * ExRe_mi_EzReT + ExIm_mi_EzIm * ExIm_mi_EzImT);
			ImMI = 0.5f * (ExIm_mi_EzIm * ExRe_mi_EzReT - ExRe_mi_EzRe * ExIm_mi_EzImT);
			break;
		}
		case 5: // Circ. Left //OC08092019: corrected to be in compliance with definitions for right-hand frame (x,z,s) and with corresponding definition and calculation of Stokes params
			//case 4: // Circ. Right
		{
			float ExRe_mi_EzIm = ExRe - EzIm, ExIm_p_EzRe = ExIm + EzRe;
			float ExRe_mi_EzImT = ExReT - EzImT, ExIm_p_EzReT = ExImT + EzReT;
			ReMI = 0.5f * (ExRe_mi_EzIm * ExRe_mi_EzImT + ExIm_p_EzRe * ExIm_p_EzReT);
			ImMI = 0.5f * (ExIm_p_EzRe * ExRe_mi_EzImT - ExRe_mi_EzIm * ExIm_p_EzReT);
			break;
		}
		case 4: // Circ. Right //OC08092019: corrected to be in compliance with definitions for right-hand frame (x,z,s) and with corresponding definition and calculation of Stokes params
			//case 5: // Circ. Left
		{
			float ExRe_p_EzIm = ExRe + EzIm, ExIm_mi_EzRe = ExIm - EzRe;
			float ExRe_p_EzImT = ExReT + EzImT, ExIm_mi_EzReT = ExImT - EzReT;
			ReMI = 0.5f * (ExRe_p_EzIm * ExRe_p_EzImT + ExIm_mi_EzRe * ExIm_mi_EzReT);
			ImMI = 0.5f * (ExIm_mi_EzRe * ExRe_p_EzImT - ExRe_p_EzIm * ExIm_mi_EzReT);
			break;
		}
		case -1: // s0
		{
			ReMI = ExRe * ExReT + ExIm * ExImT + EzRe * EzReT + EzIm * EzImT;
			ImMI = ExIm * ExReT - ExRe * ExImT + EzIm * EzReT - EzRe * EzImT;
			break;
		}
		case -2: // s1
		{
			ReMI = ExRe * ExReT + ExIm * ExImT - (EzRe * EzReT + EzIm * EzImT);
			ImMI = ExIm * ExReT - ExRe * ExImT - (EzIm * EzReT - EzRe * EzImT);
			break;
		}
		case -3: // s2
		{
			ReMI = ExImT * EzIm + ExIm * EzImT + ExReT * EzRe + ExRe * EzReT;
			ImMI = ExReT * EzIm - ExRe * EzImT - ExImT * EzRe + ExIm * EzReT;
			break;
		}
		case -4: // s3
		{
			ReMI = ExReT * EzIm + ExRe * EzImT - ExImT * EzRe - ExIm * EzReT;
			ImMI = ExIm * EzImT - ExImT * EzIm - ExReT * EzRe + ExRe * EzReT;
			break;
		}
		default: // total mutual intensity, same as s0
		{
			ReMI = ExRe * ExReT + ExIm * ExImT + EzRe * EzReT + EzIm * EzImT;
			ImMI = ExIm * ExReT - ExRe * ExImT + EzIm * EzReT - EzRe * EzImT;
			break;
			//return CAN_NOT_EXTRACT_MUT_INT;
		}
		}

		if (gt1_iter > 0)
		{
			pMI[0] = (pMI[0] * iter + (float)ReMI) / (float)(iter + 1.);
			pMI[1] = (pMI[1] * iter + (float)ImMI) / (float)(iter + 1.);
		}
		else if (gt1_iter == 0)
		{
			pMI[0] = (float)ReMI;
			pMI[1] = (float)ImMI;
		}
		else
		{
			pMI[0] += (float)ReMI;
			pMI[1] += (float)ImMI;
		}
	}
}

//HG20072026 ---------------------------------------------------------------------------
// Rank-K batched CSD accumulation.
//
// The rank-1 kernel below is a running-average read-modify-write over the whole
// (nxnz x nxnz) CSD once per macro-electron: ~1 FLOP/byte, measured at 69% of the
// A100's memory bandwidth, i.e. at the hardware limit. Buffering K electrons as the
// columns of an N x K matrix A and applying
//
//     MI <- (MI*iter0 + sum_g w_g A_g A_g^H) / (iter0 + K)
//
// with one cuBLAS cherk per weight group touches the CSD ONCE per K electrons instead
// of K times. The batching is exact, not an approximation: a sequential running mean
// over K electrons equals the batched form identically (see tools/csd_rank_k_proto.py).
//
// Every PolCom is a sum of at most two rank-1 forms +-w * a a^H with a a fixed complex
// combination of (Ex, Ez) -- e.g. s0/total is Ex Ex^H + Ez Ez^H (two positive columns),
// s1 is Ex Ex^H - Ez Ez^H (one positive, one negative). The decomposition for all 12
// PolCom cases is verified against the CPU switch formulas in
// .agents/rankk/scratch/check_colspec.py (max deviation 1.8e-15 in double).
//
// Storage convention: the CPU/GPU rank-1 code writes MI[it*2N + 2i] for i <= it with
// value E_i * conj(E_it). Viewed as a column-major complex N x N matrix that is element
// (row=i, col=it) of the UPPER triangle with value (A A^H)_{i,it} -- exactly what
// cublasCherk(CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N) computes, over exactly the same
// elements (the lower triangle and the diagonal imaginary parts are treated the same
// way by both: untouched resp. zero).
//
// The batch state is file-static: the buffered columns must outlive the individual
// srTRadGenManip instances (one is constructed per CalcIntFromElecField call), living
// in the same persistent GPU session as the CSD itself. The state is only reachable
// when a session is open and the caller requested batching (TGPUUsageArg::csdBatchK
// > 1, wired from Python as srwl.CalcIntFromElecField(..., [dev, K])); any pending
// partial batch MUST be applied before the session closes, exposed to Python as
// srwl.UtiGPUProc(4, dev). Failure to flush loses at most the last K-1 electrons on
// an abnormal exit -- on the normal path srwl_wfr_emit_prop_multi_e flushes before
// every session close (both the periodic-save close and the final one).
//---------------------------------------------------------------------------------------

struct TCSDRankKState
{
	bool active;       //column buffers allocated, spec fields valid
	bool disabled;     //unrecoverable failure -> rank-1 path for the rest of the process
	long N;            //nxnz
	int K;             //electrons per batch
	int nPend;         //electrons currently buffered
	int polCom;
	bool ehOK, evOK;
	bool additive;     //iter < 0 mode: MI += sum (no running average)
	long iter0;        //electrons already folded into the device CSD (averaging mode)
	int nPosPerElec, nNegPerElec; //columns appended per electron, per weight group
	float wPos, wNeg;  //weight magnitudes of the two groups
	float2 *dApos, *dAneg; //device column buffers, column-major N x (K*nPerElec)
	float *hostMI;     //host pointer of the CSD buffer (the key into the CAuxGPU map)
	cublasHandle_t blas;
	bool blasInit;
};
static TCSDRankKState gCSDRankK = {};

//Per-electron column specification: column a = cEx*Ex + cEz*Ez, contribution
//sign*w * a a^H. Returns false for a PolCom this path does not handle.
static bool CSDRankKColumnSpec(int PolCom, float2* cEx, float2* cEz, int* grp, int& nPos, int& nNeg, float& wPos, float& wNeg)
{
	auto C = [](float re, float im) { float2 c; c.x = re; c.y = im; return c; };
	nPos = 1; nNeg = 0; wPos = 1.f; wNeg = 1.f;
	grp[0] = 0; grp[1] = 0;
	switch(PolCom)
	{
		case 0: cEx[0] = C(1, 0); cEz[0] = C(0, 0); break; //Lin. Hor.
		case 1: cEx[0] = C(0, 0); cEz[0] = C(1, 0); break; //Lin. Vert.
		case 2: cEx[0] = C(1, 0); cEz[0] = C(1, 0); wPos = 0.5f; break; //Lin. 45
		case 3: cEx[0] = C(1, 0); cEz[0] = C(-1, 0); wPos = 0.5f; break; //Lin. 135
		case 5: cEx[0] = C(1, 0); cEz[0] = C(0, 1); wPos = 0.5f; break; //Circ. Left: Ex + i*Ez
		case 4: cEx[0] = C(1, 0); cEz[0] = C(0, -1); wPos = 0.5f; break; //Circ. Right: Ex - i*Ez
		case -2: //s1 = Ex Ex^H - Ez Ez^H
			cEx[0] = C(1, 0); cEz[0] = C(0, 0);
			cEx[1] = C(0, 0); cEz[1] = C(1, 0); grp[1] = 1; nNeg = 1; break;
		case -3: //s2 = 0.5[(Ex+Ez)(...)^H - (Ex-Ez)(...)^H]
			cEx[0] = C(1, 0); cEz[0] = C(1, 0); wPos = 0.5f;
			cEx[1] = C(1, 0); cEz[1] = C(-1, 0); grp[1] = 1; nNeg = 1; wNeg = 0.5f; break;
		case -4: //s3 = 0.5[(Ex-iEz)(...)^H - (Ex+iEz)(...)^H]
			cEx[0] = C(1, 0); cEz[0] = C(0, -1); wPos = 0.5f;
			cEx[1] = C(1, 0); cEz[1] = C(0, 1); grp[1] = 1; nNeg = 1; wNeg = 0.5f; break;
		case -1: case -5: case 6: default: //s0 / total (the kernel's default is also "total")
			cEx[0] = C(1, 0); cEz[0] = C(0, 0);
			cEx[1] = C(0, 0); cEz[1] = C(1, 0); nPos = 2; break;
	}
	return true;
}

//Gathers one electron's field into up to two contiguous complex columns.
//The E arrays are strided by PerX floats per transverse point (PerX = 2*ne).
__global__ void CSDRankKAppend_Kernel(const float* __restrict__ pEx, const float* __restrict__ pEz,
	long N, long PerX, bool ehOK, bool evOK,
	float2 cEx0, float2 cEz0, float2* __restrict__ dst0,
	float2 cEx1, float2 cEz1, float2* __restrict__ dst1)
{
	long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
	if(i >= N) return;
	float exRe = 0.f, exIm = 0.f, ezRe = 0.f, ezIm = 0.f;
	if(ehOK) { exRe = pEx[i*PerX]; exIm = pEx[i*PerX + 1]; }
	if(evOK) { ezRe = pEz[i*PerX]; ezIm = pEz[i*PerX + 1]; }
	float2 v;
	v.x = cEx0.x*exRe - cEx0.y*exIm + cEz0.x*ezRe - cEz0.y*ezIm;
	v.y = cEx0.x*exIm + cEx0.y*exRe + cEz0.x*ezIm + cEz0.y*ezRe;
	dst0[i] = v;
	if(dst1 != 0)
	{
		v.x = cEx1.x*exRe - cEx1.y*exIm + cEz1.x*ezRe - cEz1.y*ezIm;
		v.y = cEx1.x*exIm + cEx1.y*exRe + cEz1.x*ezIm + cEz1.y*ezRe;
		dst1[i] = v;
	}
}

int srTRadGenManip::FlushMutualIntensityRankK_GPU(TGPUUsageArg* pGPU, bool teardown)
{
	TCSDRankKState& S = gCSDRankK;
	int res = 0;
	if(S.active && (S.nPend > 0))
	{
		if(!S.blasInit)
		{
			if(cublasCreate(&S.blas) != CUBLAS_STATUS_SUCCESS)
			{
				printf("SRW rank-K CSD: cublasCreate FAILED; %d buffered electron(s) LOST\n", S.nPend);
				S.disabled = true; res = -1;
			}
			else
			{
				S.blasInit = true;
				cublasSetMathMode(S.blas, CUBLAS_DEFAULT_MATH); //no TF32: match fp32 rank-1 rounding class
			}
		}
		if(res == 0)
		{
			float* devMI = CAuxGPU::ToDevice(pGPU, S.hostMI, ((size_t)S.N)*((size_t)S.N)*2);
			if(devMI == 0)
			{
				printf("SRW rank-K CSD: CSD buffer could not be mapped to the device; %d buffered electron(s) LOST\n", S.nPend);
				res = -1;
			}
			else
			{
				CAuxGPU::EnsureDeviceMemoryReady(pGPU, devMI);
				float alphaP, alphaN, beta;
				if(S.additive) { alphaP = S.wPos; alphaN = -S.wNeg; beta = 1.f; }
				else
				{	//Running average: MI <- (MI*iter0 + sum w a a^H)/(iter0 + nPend).
					//iter0 == 0 gives beta == 0, i.e. plain overwrite -- same semantics as
					//the rank-1 kernel's iter == 0 case (cherk does not read C when beta==0).
					double denom = (double)S.iter0 + (double)S.nPend;
					alphaP = (float)(S.wPos/denom);
					alphaN = (float)(-S.wNeg/denom);
					beta = (float)(((double)S.iter0)/denom);
				}
				int kPos = S.nPend*S.nPosPerElec, kNeg = S.nPend*S.nNegPerElec;
				cublasStatus_t st = CUBLAS_STATUS_SUCCESS;
				if(kPos > 0) st = cublasCherk(S.blas, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, (int)S.N, kPos, &alphaP, (const cuComplex*)S.dApos, (int)S.N, &beta, (cuComplex*)devMI, (int)S.N);
				if((st == CUBLAS_STATUS_SUCCESS) && (kNeg > 0))
				{
					float one = 1.f; //the positive-group call above already applied beta
					st = cublasCherk(S.blas, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, (int)S.N, kNeg, &alphaN, (const cuComplex*)S.dAneg, (int)S.N, &one, (cuComplex*)devMI, (int)S.N);
				}
				if(st != CUBLAS_STATUS_SUCCESS)
				{
					printf("SRW rank-K CSD: cublasCherk FAILED (status %d); %d buffered electron(s) LOST\n", (int)st, S.nPend);
					S.disabled = true; res = -1;
				}
				else
				{
					CAuxGPU::MarkUpdated(pGPU, devMI, CAuxGPU::DEVICE);
					if(!S.additive) S.iter0 += S.nPend;
					S.nPend = 0;
				}
			}
		}
	}
	if(teardown && S.active)
	{
		if(S.dApos != 0) cudaFree(S.dApos);
		if(S.dAneg != 0) cudaFree(S.dAneg);
		S.dApos = 0; S.dAneg = 0;
		S.nPend = 0;
		S.active = false;
		//iter0 is deliberately kept: the electrons already applied to the (still
		//session-resident) CSD remain applied; a later re-init picks iter0 up from
		//the incoming iter index again.
	}
	return res;
}

int srTRadGenManip::AppendMutualIntensityRankK_GPU(float* pEx, float* pEz, float* pMI0, long nxnz, long PerX, long iter, int PolCom, bool EhOK, bool EvOK, int batchK, TGPUUsageArg* pGPU)
{
	TCSDRankKState& S = gCSDRankK;
	if(S.disabled) return -1; //caller falls back to the rank-1 path
	if(batchK < 2) return -1;
	bool additive = (iter < 0);

	float2 cEx[2], cEz[2]; int grp[2]; int nPos = 0, nNeg = 0; float wPos = 1.f, wNeg = 1.f;
	if(!CSDRankKColumnSpec(PolCom, cEx, cEz, grp, nPos, nNeg, wPos, wNeg)) return -1;

	//Any change of geometry/mode/output buffer: apply what is buffered, drop the buffers,
	//re-initialize below. (In production none of this changes within a session.)
	if(S.active && ((S.N != nxnz) || (S.polCom != PolCom) || (S.ehOK != EhOK) || (S.evOK != EvOK)
		|| (S.additive != additive) || (S.hostMI != pMI0) || (S.K != batchK)))
	{
		FlushMutualIntensityRankK_GPU(pGPU, true);
		if(S.disabled) return -1;
	}
	if(!S.active)
	{
		size_t colBytes = ((size_t)nxnz)*sizeof(float2);
		S.dApos = 0; S.dAneg = 0;
		if(cudaMalloc((void**)&S.dApos, colBytes*batchK*nPos) != cudaSuccess) S.dApos = 0;
		if((S.dApos != 0) && (nNeg > 0))
		{
			if(cudaMalloc((void**)&S.dAneg, colBytes*batchK*nNeg) != cudaSuccess)
			{
				cudaFree(S.dApos); S.dApos = 0;
			}
		}
		if(S.dApos == 0)
		{	//Not enough device memory for the column buffers: degrade gracefully to the
			//rank-1 path (which allocates nothing beyond what is already resident) for
			//the rest of the process, rather than failing.
			printf("SRW rank-K CSD: column buffer allocation failed (N=%ld K=%d); using the rank-1 path\n", nxnz, batchK);
			S.disabled = true;
			return -1;
		}
		S.N = nxnz; S.K = batchK; S.nPend = 0;
		S.polCom = PolCom; S.ehOK = EhOK; S.evOK = EvOK; S.additive = additive;
		S.iter0 = additive ? 0 : iter;
		S.nPosPerElec = nPos; S.nNegPerElec = nNeg; S.wPos = wPos; S.wNeg = wNeg;
		S.hostMI = pMI0;
		S.active = true;
	}
	//Averaging mode keeps a strict electron count: the device CSD holds the mean over
	//iter0 electrons and the buffers hold nPend more, so the incoming index must be
	//iter0 + nPend. Anything else (a restarted accumulation, a resumed run) applies
	//what is buffered and restarts the count from the caller's index -- for a restart
	//at iter == 0 the next flush then has beta == 0 and overwrites, exactly like the
	//rank-1 kernel's iter == 0 case.
	if(!additive && (iter != S.iter0 + S.nPend))
	{
		if(FlushMutualIntensityRankK_GPU(pGPU, false)) return -1;
		S.iter0 = iter;
	}

	float* devEx = 0; float* devEz = 0;
	if(EhOK && (pEx != 0)) devEx = CAuxGPU::ToDevice(pGPU, pEx, nxnz*2, CAuxGPU::DISCARD_HOST);
	if(EvOK && (pEz != 0)) devEz = CAuxGPU::ToDevice(pGPU, pEz, nxnz*2, CAuxGPU::DISCARD_HOST);
	CAuxGPU::EnsureDeviceMemoryReady(pGPU, devEx, devEz);
	bool ehOKdev = (devEx != 0), evOKdev = (devEz != 0);

	//Column destinations for this electron, in appending order (pos group first).
	float2* dst[2] = { 0, 0 };
	int iPos = S.nPend*S.nPosPerElec, iNeg = S.nPend*S.nNegPerElec;
	int nCols = nPos + nNeg;
	for(int c = 0; c < nCols; c++)
	{
		if(grp[c] == 0) { dst[c] = S.dApos + ((size_t)(iPos++))*S.N; }
		else { dst[c] = S.dAneg + ((size_t)(iNeg++))*S.N; }
	}

	dim3 threads(256), blocks((unsigned)((nxnz + 255)/256));
	CSDRankKAppend_Kernel<<<blocks, threads>>>(devEx, devEz, nxnz, PerX, ehOKdev, evOKdev,
		cEx[0], cEz[0], dst[0], cEx[1], cEz[1], (nCols > 1) ? dst[1] : 0);

	CAuxGPU::MarkUpdatedBatch(pGPU, CAuxGPU::DEVICE, devEx, devEz);
	if(devEx != 0) CAuxGPU::ToHostAndFree(pGPU, devEx);
	if(devEz != 0) CAuxGPU::ToHostAndFree(pGPU, devEz);

	S.nPend++;
	if(S.nPend >= S.K)
	{
		if(FlushMutualIntensityRankK_GPU(pGPU, false)) return -1;
	}
	return 0;
}
//HG20072026 --------------------------------------------------------------------------- (end rank-K)

//template <int PolCom, int gt1_iter>
//int ExtractSingleElecMutualIntensityVsXZ_GPUSub(float* pEx, float* pEz, float* pMI0, long nx, long nz, long ne, long itStart, long itEnd, long PerX, long iter, int PolCom, bool EhOK, bool EvOK, TGPUUsageArg* pGPU)
int srTRadGenManip::ExtractSingleElecMutualIntensityVsXZ_GPU(float* pEx, float* pEz, float* pMI0, long nx, long nz, long ne, long itStart, long itEnd, long PerX, long iter, int PolCom, bool EhOK, bool EvOK, TGPUUsageArg* pGPU)
{
#define GEN_MEMBERS0(i, a, b) \
	ExtractSingleElecMutualIntensityVsXZ_Kernel<i, a, b, 0, 1>, \
	ExtractSingleElecMutualIntensityVsXZ_Kernel<i, a, b, 1, 1>, \
	ExtractSingleElecMutualIntensityVsXZ_Kernel<i, a, b, -1, 1>, \
	NULL,

#define GEN_MEMBERS(i) \
	GEN_MEMBERS0(i, false, false) \
	GEN_MEMBERS0(i, false, true) \
	GEN_MEMBERS0(i, true, false) \
	GEN_MEMBERS0(i, true, true)

	decltype(ExtractSingleElecMutualIntensityVsXZ_Kernel<0, false, false, 0, 1>) *ExtractSingleElecMutualIntensityVsXZ_tbl[] = {
		GEN_MEMBERS(-5)
		GEN_MEMBERS(-4)
		GEN_MEMBERS(-3)
		GEN_MEMBERS(-2)
		GEN_MEMBERS(-1)
		GEN_MEMBERS(0)
		GEN_MEMBERS(1)
		GEN_MEMBERS(2)
		GEN_MEMBERS(3)
		GEN_MEMBERS(4)
		GEN_MEMBERS(5)
		GEN_MEMBERS(6) //HG20072026 PolCom=6 ("Total") was MISSING, and it is the most commonly used setting.
		               //The index below is ((PolCom+5)<<4)|..., so PolCom=6 produced idx>=176 on a
		               //176-entry table: the GPU path jumped through an out-of-bounds function
		               //pointer and segfaulted, while the CPU path handled 6 fine via its `default`
		               //case. The kernel itself was always correct for 6 (its switch `default:` is
		               //"total mutual intensity, same as s0") -- only the instantiation was absent.
	};
#undef GEN_MEMBERS0
#undef GEN_MEMBERS

	//HG20072026 The table is indexed by ((PolCom+5)<<4)|..., so an out-of-range PolCom
	//reads past the end and launches a garbage function pointer. Returning non-zero makes
	//the caller fall back to the CPU loop, which handles any PolCom via its `default` case.
	//This is a guard, not the fix -- PolCom=6 is now instantiated above.
	const int nPolTblEntries = (int)(sizeof(ExtractSingleElecMutualIntensityVsXZ_tbl)
	                                 / sizeof(ExtractSingleElecMutualIntensityVsXZ_tbl[0]));
	if((PolCom < -5) || ((((PolCom + 5) << 4) | 0xF) >= nPolTblEntries)) return -1;

	long long nxnz = ((long long)nx) * ((long long)nz); //HG26022024 NOTE: GPU implementation is only called for nxnz < UINT_MAX to avoid integer overflows

	//HG20072026 Rank-K batched path (see the block comment above TCSDRankKState). Only for
	//full-range updates: a partial [itStart, itEnd] window (used by the _n_mpi > 1 CSD
	//splitting) keeps the proven rank-1 kernel. On any refusal (unsupported shape, failed
	//allocation) fall through to the rank-1 code below -- the refusal is sticky per process,
	//so a session never mixes deferred-batch and immediate updates inconsistently.
	if((pGPU != 0) && (pGPU->csdBatchK > 1) && (itStart == 0) && (itEnd == nxnz - 1) && (nxnz > 1))
	{
		if(AppendMutualIntensityRankK_GPU(pEx, pEz, pMI0, (long)nxnz, PerX, iter, PolCom, EhOK, EvOK, pGPU->csdBatchK, pGPU) == 0) return 0;
	}

	const int itPerBlk = 1;
	dim3 threads = dim3(48, 16, 1);
	dim3 grid = dim3((unsigned int)((nxnz + 1) / threads.x + (threads.x > 1)), (unsigned int)((nxnz / 2) / (threads.y * itPerBlk) + (threads.y > 1)), 1); //OC19022024 (cast to remove warning)
	//dim3 grid = dim3((nxnz + 1) / threads.x + (threads.x > 1), (nxnz / 2) / (threads.y * itPerBlk) + (threads.y > 1), 1);

	pEx = CAuxGPU::ToDevice(pGPU, pEx, nxnz*2, CAuxGPU::DISCARD_HOST);
	pEz = CAuxGPU::ToDevice(pGPU, pEz, nxnz*2, CAuxGPU::DISCARD_HOST);
	//HG20072026 (itEnd - itStart + 1), not (itEnd - itStart): itEnd is an INCLUSIVE bound
	//(srradmnp.cpp:1643 sets it to nxnz-1 and the CPU loop runs it<=itEnd), so the region
	//spans itEnd-itStart+1 rows. Under-mapping by one row meant the last row of the host
	//CSD was never registered and so never received results on copy-back.
	pMI0 = CAuxGPU::ToDevice(pGPU, pMI0, (itEnd - itStart + 1)*nxnz*2);
	CAuxGPU::EnsureDeviceMemoryReady(pGPU, pEx, pEz, pMI0); //HG31072024 Bug-fix

	int idx = ((PolCom + 5) << 4) | ((EhOK & 1) << 3) | ((EvOK & 1) << 2);
	if (iter > 0) idx |= 1;
	else if (iter < 0) idx |= 2;

	ExtractSingleElecMutualIntensityVsXZ_tbl[idx]<<<grid, threads >>> (pEx, pEz, pMI0, (long)nxnz, itStart, itEnd, PerX, iter);

	CAuxGPU::MarkUpdatedBatch(pGPU, CAuxGPU::DEVICE, pEx, pEz, pMI0);
	pEx = CAuxGPU::ToHostAndFree(pGPU, pEx);
	pEz = CAuxGPU::ToHostAndFree(pGPU, pEz);

//HG26022024 (commented out)
//#ifdef _DEBUG
//	if (pMI0 != NULL)
//		pMI0 = (float*)CAuxGPU::ToHostAndFree(pGPU, pMI0, (itEnd - itStart)*ne*nx*nz*2*sizeof(float));
//
//	cudaStreamSynchronize(0);
//	auto err = cudaGetLastError();
//	printf("%s\r\n", cudaGetErrorString(err));
//#endif
	return 0;
}

#endif