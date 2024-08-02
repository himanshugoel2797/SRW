/************************************************************************//**
 * File: sroptelm_gpu.cu
 * Description: Optical element (general CUDA functions)
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
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math_constants.h"
#include "cooperative_groups.h" //HG31072024
#include "cooperative_groups/reduce.h" //HG31072024

#include <stdio.h>
#include <iostream>
#include <chrono>
#include "sroptelm.h"
#include "sroptelm_gpu.h"

namespace cg = cooperative_groups; //HG31072024

//__global__ void TreatStronglyOscillatingTerm_Kernel(srTSRWRadStructAccessData RadAccessData, bool TreatPolCompX, bool TreatPolCompZ, double ConstRx, double ConstRz, int ieStart) 
__global__ void TreatStronglyOscillatingTerm_Kernel(srTSRWRadStructAccessData* pRadAccessData, bool TreatPolCompX, bool TreatPolCompZ, double ConstRx, double ConstRz, int ieStart, int ieBefEnd) //HG27072024
{
    int ie = (blockIdx.x * blockDim.x + threadIdx.x); //ne range
	int ix = (blockIdx.y * blockDim.y + threadIdx.y); //nx range
	int iz = (blockIdx.z * blockDim.z + threadIdx.z); //nz range
	
    if (ix < pRadAccessData->nx && iz < pRadAccessData->nz && ie < ieBefEnd) 
    {
        double ePh = pRadAccessData->eStart + pRadAccessData->eStep * (ie - ieStart);
        if (pRadAccessData->PresT == 1)
        {
            ePh = pRadAccessData->avgPhotEn; //?? OC041108
        }

        double ConstRxE = ConstRx * ePh;
        double ConstRzE = ConstRz * ePh;
        if (pRadAccessData->Pres == 1)
        {
            //double Lambda_m = 1.239854e-06/ePh;
            double Lambda_m = 1.239842e-06 / ePh;
            if (pRadAccessData->PhotEnergyUnit == 1) Lambda_m *= 0.001; // if keV

            double Lambda_me2 = Lambda_m * Lambda_m;
            ConstRxE *= Lambda_me2;
            ConstRzE *= Lambda_me2;
        }

        double z = (pRadAccessData->zStart - pRadAccessData->zc) + (iz * pRadAccessData->zStep);
        double PhaseAddZ = 0;
        if (pRadAccessData->WfrQuadTermCanBeTreatedAtResizeZ) PhaseAddZ = ConstRzE * z * z;

        double x = (pRadAccessData->xStart - pRadAccessData->xc) + (ix * pRadAccessData->xStep);
        double Phase = PhaseAddZ;
        if (pRadAccessData->WfrQuadTermCanBeTreatedAtResizeX) Phase += ConstRxE * x * x;

        float SinPh, CosPh;
        sincosf(Phase, &SinPh, &CosPh);

        long long PerX = pRadAccessData->ne << 1;
        long long PerZ = PerX * pRadAccessData->nx;
        long long offset = ie * 2 + iz * PerZ + ix * PerX;
        
		if (TreatPolCompX)
		{
			float* pExRe = pRadAccessData->pBaseRadX + offset;
			float* pExIm = pExRe + 1;
			double ExReNew = (*pExRe) * CosPh - (*pExIm) * SinPh;
			double ExImNew = (*pExRe) * SinPh + (*pExIm) * CosPh;
			*pExRe = (float)ExReNew; *pExIm = (float)ExImNew;
		}
		if (TreatPolCompZ)
		{
			float* pEzRe = pRadAccessData->pBaseRadZ + offset;
			float* pEzIm = pEzRe + 1;
			double EzReNew = (*pEzRe) * CosPh - (*pEzIm) * SinPh;
			double EzImNew = (*pEzRe) * SinPh + (*pEzIm) * CosPh;
			*pEzRe = (float)EzReNew; *pEzIm = (float)EzImNew;
		}
    }
}

void srTGenOptElem::TreatStronglyOscillatingTerm_GPU(srTSRWRadStructAccessData& RadAccessData, bool TreatPolCompX, bool TreatPolCompZ, double ConstRx, double ConstRz, int ieStart, int ieBefEnd, TGPUUsageArg* pGPU)
{
	if (RadAccessData.pBaseRadX != NULL)
	{
		RadAccessData.pBaseRadX = (float*)CAuxGPU::ToDevice(pGPU, RadAccessData.pBaseRadX, 2*RadAccessData.ne*RadAccessData.nx*RadAccessData.nz*sizeof(float));
		CAuxGPU::EnsureDeviceMemoryReady(pGPU, RadAccessData.pBaseRadX);
	}
	if (RadAccessData.pBaseRadZ != NULL)
	{
		RadAccessData.pBaseRadZ = (float*)CAuxGPU::ToDevice(pGPU, RadAccessData.pBaseRadZ, 2*RadAccessData.ne*RadAccessData.nx*RadAccessData.nz*sizeof(float));
		CAuxGPU::EnsureDeviceMemoryReady(pGPU, RadAccessData.pBaseRadZ);
	}

	srTSRWRadStructAccessData* pRadAccessData_dev = (srTSRWRadStructAccessData*)CAuxGPU::ToDevice(pGPU, &RadAccessData, sizeof(srTSRWRadStructAccessData)); //HG27072024
	CAuxGPU::EnsureDeviceMemoryReady(pGPU, pRadAccessData_dev);

    const int bs = 256;
	dim3 blocks(ieBefEnd - ieStart, RadAccessData.nx, RadAccessData.nz);
	dim3 threads(1, 1, 1);
	if (blocks.x > 1)
	{
		if (blocks.x >= bs)
			threads.x = bs;
		else
			threads.x = blocks.x;


		if (blocks.x % threads.x > 0) blocks.x = blocks.x / threads.x + 1;
		else blocks.x = blocks.x / threads.x;

		if (bs / threads.x > 1)
		{
			threads.y = bs / threads.x;
			if (blocks.y % threads.y > 0) blocks.y = blocks.y / threads.y + 1;
			else blocks.y = blocks.y / threads.y;
		}
	}
	else
	{
		if (bs / blocks.y > 1)
		{
			threads.y = blocks.y;
			blocks.y = 1;
			threads.z = bs / threads.y;
			if (blocks.z % threads.z > 0) blocks.z = blocks.z / threads.z + 1;
			else blocks.z = blocks.z / threads.z;
		}
		else
		{
			threads.y = bs;
			if (blocks.y % threads.y > 0) blocks.y = blocks.y / threads.y + 1;
			else blocks.y = blocks.y / threads.y;
		}
	}

    //TreatStronglyOscillatingTerm_Kernel<< <blocks, threads >> > (RadAccessData, TreatPolCompX, TreatPolCompZ, ConstRx, ConstRz, ieStart);
    TreatStronglyOscillatingTerm_Kernel<< <blocks, threads >> > (pRadAccessData_dev, TreatPolCompX, TreatPolCompZ, ConstRx, ConstRz, ieStart, ieBefEnd); //HG27072024

	CAuxGPU::ToHostAndFree(pGPU, pRadAccessData_dev, sizeof(srTSRWRadStructAccessData), true); //HG27072024
	
	CAuxGPU::MarkUpdated(pGPU, RadAccessData.pBaseRadX, true, false);
	CAuxGPU::MarkUpdated(pGPU, RadAccessData.pBaseRadZ, true, false);

//#ifndef _DEBUG
	if (RadAccessData.pBaseRadX != NULL)
		RadAccessData.pBaseRadX = (float*)CAuxGPU::GetHostPtr(pGPU, RadAccessData.pBaseRadX);
	if (RadAccessData.pBaseRadZ != NULL)
		RadAccessData.pBaseRadZ = (float*)CAuxGPU::GetHostPtr(pGPU, RadAccessData.pBaseRadZ);
//#endif

//#ifdef _DEBUG
//	if (RadAccessData.pBaseRadX != NULL)
//		RadAccessData.pBaseRadX = (float*)CAuxGPU::ToHostAndFree(pGPU, RadAccessData.pBaseRadX, 2*RadAccessData.ne*RadAccessData.nx*RadAccessData.nz*sizeof(float));
//	if (RadAccessData.pBaseRadZ != NULL)
//		RadAccessData.pBaseRadZ = (float*)CAuxGPU::ToHostAndFree(pGPU, RadAccessData.pBaseRadZ, 2*RadAccessData.ne*RadAccessData.nx*RadAccessData.nz*sizeof(float));
//	cudaStreamSynchronize(0);
//	auto err = cudaGetLastError();
//	printf("%s\r\n", cudaGetErrorString(err));
//#endif
}

__global__ void MakeWfrEdgeCorrection_Kernel(srTSRWRadStructAccessData* pRadAccessData, float* pDataEx, float* pDataEz, srTDataPtrsForWfrEdgeCorr DataPtrs, float dxSt, float dxFi, float dzSt, float dzFi)
{
    int ix = (blockIdx.x * blockDim.x + threadIdx.x); //nx range
    int iz = (blockIdx.y * blockDim.y + threadIdx.y); //nz range

    if (ix < pRadAccessData->nx && iz < pRadAccessData->nz)
    {
		//float dxSt = (float)DataPtrs.dxSt;
		//float dxFi = (float)DataPtrs.dxFi;
		//float dzSt = (float)DataPtrs.dzSt;
		//float dzFi = (float)DataPtrs.dzFi;
		float dxSt_dzSt = dxSt * dzSt;
		float dxSt_dzFi = dxSt * dzFi;
		float dxFi_dzSt = dxFi * dzSt;
		float dxFi_dzFi = dxFi * dzFi;

		//long TwoNz = pRadAccessData->nz << 1; //OC25012024 (commented-out)
		long PerX = 2;
		long PerZ = PerX * pRadAccessData->nx;

        float fSSExRe = DataPtrs.fxStzSt[0];
        float fSSExIm = DataPtrs.fxStzSt[1];
        float fSSEzRe = DataPtrs.fxStzSt[2];
        float fSSEzIm = DataPtrs.fxStzSt[3];
        
        float fFSExRe = DataPtrs.fxFizSt[0];
        float fFSExIm = DataPtrs.fxFizSt[1];
        float fFSEzRe = DataPtrs.fxFizSt[2];
        float fFSEzIm = DataPtrs.fxFizSt[3];
        
        float fSFExRe = DataPtrs.fxStzFi[0];
        float fSFExIm = DataPtrs.fxStzFi[1];
        float fSFEzRe = DataPtrs.fxStzFi[2];
        float fSFEzIm = DataPtrs.fxStzFi[3];
        
        float fFFExRe = DataPtrs.fxFizFi[0];
        float fFFExIm = DataPtrs.fxFizFi[1];
        float fFFEzRe = DataPtrs.fxFizFi[2];
        float fFFEzIm = DataPtrs.fxFizFi[3];

		float bRe, bIm, cRe, cIm;

		long long Two_iz = iz << 1;
		long long Two_iz_p_1 = Two_iz + 1;
		long long Two_ix = ix << 1;
		long long Two_ix_p_1 = Two_ix + 1;

		float* tEx = pDataEx + iz * PerZ + ix * PerX, * tEz = pDataEz + iz * PerZ + ix * PerX;
		float ExRe = *tEx, ExIm = *(tEx + 1);
		float EzRe = *tEz, EzIm = *(tEz + 1);

		if (dxSt != 0.f)
		{
			float ExpXStRe = DataPtrs.ExpArrXSt[Two_ix], ExpXStIm = DataPtrs.ExpArrXSt[Two_ix_p_1];

			bRe = DataPtrs.FFTArrXStEx[Two_iz]; bIm = DataPtrs.FFTArrXStEx[Two_iz_p_1];
			ExRe += (float)(dxSt * (ExpXStRe * bRe - ExpXStIm * bIm));
			ExIm += (float)(dxSt * (ExpXStRe * bIm + ExpXStIm * bRe));

			bRe = DataPtrs.FFTArrXStEz[Two_iz]; bIm = DataPtrs.FFTArrXStEz[Two_iz_p_1];
			EzRe += (float)(dxSt * (ExpXStRe * bRe - ExpXStIm * bIm));
			EzIm += (float)(dxSt * (ExpXStRe * bIm + ExpXStIm * bRe));

			if (dzSt != 0.f)
			{
				bRe = DataPtrs.ExpArrZSt[Two_iz], bIm = DataPtrs.ExpArrZSt[Two_iz_p_1];
				cRe = ExpXStRe * bRe - ExpXStIm * bIm; cIm = ExpXStRe * bIm + ExpXStIm * bRe;

				ExRe += (float)(dxSt_dzSt * (fSSExRe * cRe - fSSExIm * cIm));
				ExIm += (float)(dxSt_dzSt * (fSSExRe * cIm + fSSExIm * cRe));
				EzRe += (float)(dxSt_dzSt * (fSSEzRe * cRe - fSSEzIm * cIm));
				EzIm += (float)(dxSt_dzSt * (fSSEzRe * cIm + fSSEzIm * cRe));
			}
			if (dzFi != 0.f)
			{
				bRe = DataPtrs.ExpArrZFi[Two_iz], bIm = DataPtrs.ExpArrZFi[Two_iz_p_1];
				cRe = ExpXStRe * bRe - ExpXStIm * bIm; cIm = ExpXStRe * bIm + ExpXStIm * bRe;

				ExRe -= (float)(dxSt_dzFi * (fSFExRe * cRe - fSFExIm * cIm));
				ExIm -= (float)(dxSt_dzFi * (fSFExRe * cIm + fSFExIm * cRe));
				EzRe -= (float)(dxSt_dzFi * (fSFEzRe * cRe - fSFEzIm * cIm));
				EzIm -= (float)(dxSt_dzFi * (fSFEzRe * cIm + fSFEzIm * cRe));
			}
		}
		if (dxFi != 0.f)
		{
			float ExpXFiRe = DataPtrs.ExpArrXFi[Two_ix], ExpXFiIm = DataPtrs.ExpArrXFi[Two_ix_p_1];

			bRe = DataPtrs.FFTArrXFiEx[Two_iz]; bIm = DataPtrs.FFTArrXFiEx[Two_iz_p_1];
			ExRe -= (float)(dxFi * (ExpXFiRe * bRe - ExpXFiIm * bIm));
			ExIm -= (float)(dxFi * (ExpXFiRe * bIm + ExpXFiIm * bRe));

			bRe = DataPtrs.FFTArrXFiEz[Two_iz]; bIm = DataPtrs.FFTArrXFiEz[Two_iz_p_1];
			EzRe -= (float)(dxFi * (ExpXFiRe * bRe - ExpXFiIm * bIm));
			EzIm -= (float)(dxFi * (ExpXFiRe * bIm + ExpXFiIm * bRe));

			if (dzSt != 0.f)
			{
				bRe = DataPtrs.ExpArrZSt[Two_iz], bIm = DataPtrs.ExpArrZSt[Two_iz_p_1];
				cRe = ExpXFiRe * bRe - ExpXFiIm * bIm; cIm = ExpXFiRe * bIm + ExpXFiIm * bRe;

				ExRe -= (float)(dxFi_dzSt * (fFSExRe * cRe - fFSExIm * cIm));
				ExIm -= (float)(dxFi_dzSt * (fFSExRe * cIm + fFSExIm * cRe));
				EzRe -= (float)(dxFi_dzSt * (fFSEzRe * cRe - fFSEzIm * cIm));
				EzIm -= (float)(dxFi_dzSt * (fFSEzRe * cIm + fFSEzIm * cRe));
			}
			if (dzFi != 0.f)
			{
				bRe = DataPtrs.ExpArrZFi[Two_iz], bIm = DataPtrs.ExpArrZFi[Two_iz_p_1];
				cRe = ExpXFiRe * bRe - ExpXFiIm * bIm; cIm = ExpXFiRe * bIm + ExpXFiIm * bRe;

				ExRe += (float)(dxFi_dzFi * (fFFExRe * cRe - fFFExIm * cIm));
				ExIm += (float)(dxFi_dzFi * (fFFExRe * cIm + fFFExIm * cRe));
				EzRe += (float)(dxFi_dzFi * (fFFEzRe * cRe - fFFEzIm * cIm));
				EzIm += (float)(dxFi_dzFi * (fFFEzRe * cIm + fFFEzIm * cRe));
			}
		}
		if (dzSt != 0.f)
		{
			float ExpZStRe = DataPtrs.ExpArrZSt[Two_iz], ExpZStIm = DataPtrs.ExpArrZSt[Two_iz_p_1];

			bRe = DataPtrs.FFTArrZStEx[Two_ix]; bIm = DataPtrs.FFTArrZStEx[Two_ix_p_1];
			ExRe += (float)(dzSt * (ExpZStRe * bRe - ExpZStIm * bIm));
			ExIm += (float)(dzSt * (ExpZStRe * bIm + ExpZStIm * bRe));

			bRe = DataPtrs.FFTArrZStEz[Two_ix]; bIm = DataPtrs.FFTArrZStEz[Two_ix_p_1];
			EzRe += (float)(DataPtrs.dzSt * (ExpZStRe * bRe - ExpZStIm * bIm));
			EzIm += (float)(DataPtrs.dzSt * (ExpZStRe * bIm + ExpZStIm * bRe));
		}
		if (dzFi != 0.f)
		{
			float ExpZFiRe = DataPtrs.ExpArrZFi[Two_iz], ExpZFiIm = DataPtrs.ExpArrZFi[Two_iz_p_1];

			bRe = DataPtrs.FFTArrZFiEx[Two_ix]; bIm = DataPtrs.FFTArrZFiEx[Two_ix_p_1];
			ExRe -= (float)(dzFi * (ExpZFiRe * bRe - ExpZFiIm * bIm));
			ExIm -= (float)(dzFi * (ExpZFiRe * bIm + ExpZFiIm * bRe));

			bRe = DataPtrs.FFTArrZFiEz[Two_ix]; bIm = DataPtrs.FFTArrZFiEz[Two_ix_p_1];
			EzRe -= (float)(dzFi * (ExpZFiRe * bRe - ExpZFiIm * bIm));
			EzIm -= (float)(dzFi * (ExpZFiRe * bIm + ExpZFiIm * bRe));
		}

		*tEx = ExRe; *(tEx + 1) = ExIm;
		*tEz = EzRe; *(tEz + 1) = EzIm;
    }
}

void srTGenOptElem::MakeWfrEdgeCorrection_GPU(srTSRWRadStructAccessData* RadAccessData, float* pDataEx, float* pDataEz, srTDataPtrsForWfrEdgeCorr& DataPtrs, TGPUUsageArg* pGPU)
{
	pDataEx = (float*)CAuxGPU::ToDevice(pGPU, pDataEx, 2*RadAccessData->ne*RadAccessData->nx*RadAccessData->nz*sizeof(float));
	pDataEz = (float*)CAuxGPU::ToDevice(pGPU, pDataEz, 2*RadAccessData->ne*RadAccessData->nx*RadAccessData->nz*sizeof(float));
	DataPtrs.FFTArrXStEx = (float*)CAuxGPU::ToDevice(pGPU, DataPtrs.FFTArrXStEx, 2*RadAccessData->nz*sizeof(float));
	DataPtrs.FFTArrXStEz = (float*)CAuxGPU::ToDevice(pGPU, DataPtrs.FFTArrXStEz, 2*RadAccessData->nz*sizeof(float));
	DataPtrs.FFTArrXFiEx = (float*)CAuxGPU::ToDevice(pGPU, DataPtrs.FFTArrXFiEx, 2*RadAccessData->nz*sizeof(float));
	DataPtrs.FFTArrXFiEz = (float*)CAuxGPU::ToDevice(pGPU, DataPtrs.FFTArrXFiEz, 2*RadAccessData->nz*sizeof(float));
	DataPtrs.FFTArrZStEx = (float*)CAuxGPU::ToDevice(pGPU, DataPtrs.FFTArrZStEx, 2*RadAccessData->nx*sizeof(float));
	DataPtrs.FFTArrZStEz = (float*)CAuxGPU::ToDevice(pGPU, DataPtrs.FFTArrZStEz, 2*RadAccessData->nx*sizeof(float));
	DataPtrs.FFTArrZFiEx = (float*)CAuxGPU::ToDevice(pGPU, DataPtrs.FFTArrZFiEx, 2*RadAccessData->nx*sizeof(float));
	DataPtrs.FFTArrZFiEz = (float*)CAuxGPU::ToDevice(pGPU, DataPtrs.FFTArrZFiEz, 2*RadAccessData->nx*sizeof(float));
	DataPtrs.ExpArrXSt = (float*)CAuxGPU::ToDevice(pGPU, DataPtrs.ExpArrXSt, 2*RadAccessData->nx*sizeof(float));
	DataPtrs.ExpArrXFi = (float*)CAuxGPU::ToDevice(pGPU, DataPtrs.ExpArrXFi, 2*RadAccessData->nx*sizeof(float));
	DataPtrs.ExpArrZSt = (float*)CAuxGPU::ToDevice(pGPU, DataPtrs.ExpArrZSt, 2*RadAccessData->nz*sizeof(float));
	DataPtrs.ExpArrZFi = (float*)CAuxGPU::ToDevice(pGPU, DataPtrs.ExpArrZFi, 2*RadAccessData->nz*sizeof(float));

	CAuxGPU::EnsureDeviceMemoryReady(pGPU, pDataEx);
	CAuxGPU::EnsureDeviceMemoryReady(pGPU, pDataEz);
	CAuxGPU::EnsureDeviceMemoryReady(pGPU, DataPtrs.FFTArrXStEx);
	CAuxGPU::EnsureDeviceMemoryReady(pGPU, DataPtrs.FFTArrXStEz);
	CAuxGPU::EnsureDeviceMemoryReady(pGPU, DataPtrs.FFTArrXFiEx);
	CAuxGPU::EnsureDeviceMemoryReady(pGPU, DataPtrs.FFTArrXFiEz);
	CAuxGPU::EnsureDeviceMemoryReady(pGPU, DataPtrs.FFTArrZStEx);
	CAuxGPU::EnsureDeviceMemoryReady(pGPU, DataPtrs.FFTArrZStEz);
	CAuxGPU::EnsureDeviceMemoryReady(pGPU, DataPtrs.FFTArrZFiEx);
	CAuxGPU::EnsureDeviceMemoryReady(pGPU, DataPtrs.FFTArrZFiEz);
	CAuxGPU::EnsureDeviceMemoryReady(pGPU, DataPtrs.ExpArrXSt);
	CAuxGPU::EnsureDeviceMemoryReady(pGPU, DataPtrs.ExpArrXFi);
	CAuxGPU::EnsureDeviceMemoryReady(pGPU, DataPtrs.ExpArrZSt);
	CAuxGPU::EnsureDeviceMemoryReady(pGPU, DataPtrs.ExpArrZFi);

	srTSRWRadStructAccessData* pRadAccessData_dev = (srTSRWRadStructAccessData*)CAuxGPU::ToDevice(pGPU, RadAccessData, sizeof(srTSRWRadStructAccessData)); //HG27072024
	CAuxGPU::EnsureDeviceMemoryReady(pGPU, pRadAccessData_dev);

	const int bs = 256;
	dim3 blocks(RadAccessData->nx / bs + ((RadAccessData->nx & (bs - 1)) != 0), RadAccessData->nz);
	dim3 threads(bs, 1);
	//MakeWfrEdgeCorrection_Kernel << <blocks, threads >> > (*RadAccessData, pDataEx, pDataEz, DataPtrs, (float)DataPtrs.dxSt, (float)DataPtrs.dxFi, (float)DataPtrs.dzSt, (float)DataPtrs.dzFi);
	MakeWfrEdgeCorrection_Kernel << <blocks, threads >> > (pRadAccessData_dev, pDataEx, pDataEz, DataPtrs, (float)DataPtrs.dxSt, (float)DataPtrs.dxFi, (float)DataPtrs.dzSt, (float)DataPtrs.dzFi); //HG27072024

	CAuxGPU::ToHostAndFree(pGPU, pRadAccessData_dev, sizeof(srTSRWRadStructAccessData), true); //HG27072024

	DataPtrs.FFTArrXStEx = (float*)CAuxGPU::ToHostAndFree(pGPU, DataPtrs.FFTArrXStEx, 2*RadAccessData->nz*sizeof(float), true);
	DataPtrs.FFTArrXStEz = (float*)CAuxGPU::ToHostAndFree(pGPU, DataPtrs.FFTArrXStEz, 2*RadAccessData->nz*sizeof(float), true);
	DataPtrs.FFTArrXFiEx = (float*)CAuxGPU::ToHostAndFree(pGPU, DataPtrs.FFTArrXFiEx, 2*RadAccessData->nz*sizeof(float), true);
	DataPtrs.FFTArrXFiEz = (float*)CAuxGPU::ToHostAndFree(pGPU, DataPtrs.FFTArrXFiEz, 2*RadAccessData->nz*sizeof(float), true);
	DataPtrs.FFTArrZStEx = (float*)CAuxGPU::ToHostAndFree(pGPU, DataPtrs.FFTArrZStEx, 2*RadAccessData->nx*sizeof(float), true);
	DataPtrs.FFTArrZStEz = (float*)CAuxGPU::ToHostAndFree(pGPU, DataPtrs.FFTArrZStEz, 2*RadAccessData->nx*sizeof(float), true);
	DataPtrs.FFTArrZFiEx = (float*)CAuxGPU::ToHostAndFree(pGPU, DataPtrs.FFTArrZFiEx, 2*RadAccessData->nx*sizeof(float), true);
	DataPtrs.FFTArrZFiEz = (float*)CAuxGPU::ToHostAndFree(pGPU, DataPtrs.FFTArrZFiEz, 2*RadAccessData->nx*sizeof(float), true);
	DataPtrs.ExpArrXSt = (float*)CAuxGPU::ToHostAndFree(pGPU, DataPtrs.ExpArrXSt, 2*RadAccessData->nx*sizeof(float), true);
	DataPtrs.ExpArrXFi = (float*)CAuxGPU::ToHostAndFree(pGPU, DataPtrs.ExpArrXFi, 2*RadAccessData->nx*sizeof(float), true);
	DataPtrs.ExpArrZSt = (float*)CAuxGPU::ToHostAndFree(pGPU, DataPtrs.ExpArrZSt, 2*RadAccessData->nz*sizeof(float), true);
	DataPtrs.ExpArrZFi = (float*)CAuxGPU::ToHostAndFree(pGPU, DataPtrs.ExpArrZFi, 2*RadAccessData->nz*sizeof(float), true);

	CAuxGPU::MarkUpdated(pGPU, pDataEx, true, false);
	CAuxGPU::MarkUpdated(pGPU, pDataEz, true, false);

//#ifdef _DEBUG
//	CAuxGPU::ToHostAndFree(pGPU, pDataEx, 2*RadAccessData->ne*RadAccessData->nx*RadAccessData->nz*sizeof(float));
//	CAuxGPU::ToHostAndFree(pGPU, pDataEz, 2*RadAccessData->ne*RadAccessData->nx*RadAccessData->nz*sizeof(float));
//	cudaStreamSynchronize(0);
//	auto err = cudaGetLastError();
//	printf("%s\r\n", cudaGetErrorString(err));
//#endif
}

template<bool TreatPolCompX, bool TreatPolCompZ> __global__ void RadResizeCore_Kernel(srTSRWRadStructAccessData* pOldRadAccessData, srTSRWRadStructAccessData* pNewRadAccessData)
{
	int ixStart = int(pNewRadAccessData->AuxLong1);
	int ixEnd = int(pNewRadAccessData->AuxLong2);
	int izStart = int(pNewRadAccessData->AuxLong3);
	int izEnd = int(pNewRadAccessData->AuxLong4);

    int ix = (blockIdx.x * blockDim.x + threadIdx.x) + ixStart; //nx range
    int iz = (blockIdx.y * blockDim.y + threadIdx.y) + izStart; //nz range
    int ie = (blockIdx.z * blockDim.z + threadIdx.z); //ne range

	if (ix > ixEnd) return;
	if (iz > izEnd) return;

	const double DistAbsTol = 1.E-10;
	double xStepInvOld = 1./pOldRadAccessData->xStep;
	double zStepInvOld = 1./pOldRadAccessData->zStep;
	int nx_mi_1Old = pOldRadAccessData->nx - 1;
	int nz_mi_1Old = pOldRadAccessData->nz - 1;
	int nx_mi_2Old = nx_mi_1Old - 1;
	int nz_mi_2Old = nz_mi_1Old - 1;

	//OC31102018: moved by SY at parallelizing SRW via OpenMP
	//srTInterpolAux01 InterpolAux01;
	//srTInterpolAux02 InterpolAux02[4], InterpolAux02I[2];
	//srTInterpolAuxF AuxF[4], AuxFI[2];
	//int ixStOld, izStOld, ixStOldPrev = -1000, izStOldPrev = -1000;

	//long PerX_New = pNewRadAccessData->ne << 1;
	//long PerZ_New = PerX_New*pNewRadAccessData->nx;
	long long PerX_New = pNewRadAccessData->ne << 1;
	long long PerZ_New = PerX_New*pNewRadAccessData->nx;

	//long PerX_Old = PerX_New;
	//long PerZ_Old = PerX_Old*pOldRadAccessData->nx;
	long long PerX_Old = PerX_New;
	long long PerZ_Old = PerX_Old*pOldRadAccessData->nx;

	float *pEX0_New = 0, *pEZ0_New = 0;
	pEX0_New = pNewRadAccessData->pBaseRadX;
	pEZ0_New = pNewRadAccessData->pBaseRadZ;

	float* pEX0_Old = 0, * pEZ0_Old = 0;
	pEX0_Old = pOldRadAccessData->pBaseRadX;
	pEZ0_Old = pOldRadAccessData->pBaseRadZ;

	
	int ixStOld, izStOld; //OC25012024 //ixStOldPrev = -1000, izStOldPrev = -1000;
	//int ixStOld, izStOld, ixStOldPrev = -1000, izStOldPrev = -1000;
	//SY: do we need this (always returns 0, updates some clock)
	//if(result = srYield.Check()) return result;

	double zAbs = pNewRadAccessData->zStart + iz * pNewRadAccessData->zStep;

	char FieldShouldBeZeroedDueToZ = 0;
	if (pNewRadAccessData->WfrEdgeCorrShouldBeDone)
	{
		if ((zAbs < pNewRadAccessData->zWfrMin - DistAbsTol) || (zAbs > pNewRadAccessData->zWfrMax + DistAbsTol)) FieldShouldBeZeroedDueToZ = 1;
	}

	int izcOld = int((zAbs - pOldRadAccessData->zStart) * zStepInvOld + 1.E-06);

	double zRel = zAbs - (pOldRadAccessData->zStart + izcOld * pOldRadAccessData->zStep);

	if (izcOld == nz_mi_1Old) { izStOld = izcOld - 3; zRel += 2. * pOldRadAccessData->zStep; }
	else if (izcOld == nz_mi_2Old) { izStOld = izcOld - 2; zRel += pOldRadAccessData->zStep; }
	else if (izcOld == 0) { izStOld = izcOld; zRel -= pOldRadAccessData->zStep; }
	else izStOld = izcOld - 1;

	zRel *= zStepInvOld;

	int izcOld_mi_izStOld = izcOld - izStOld;
	//long izPerZ_New = iz*PerZ_New;
	long long izPerZ_New = iz * PerZ_New;

	double xAbs = pNewRadAccessData->xStart + ix * pNewRadAccessData->xStep;

	char FieldShouldBeZeroedDueToX = 0;
	if (pNewRadAccessData->WfrEdgeCorrShouldBeDone)
	{
		if ((xAbs < pNewRadAccessData->xWfrMin - DistAbsTol) || (xAbs > pNewRadAccessData->xWfrMax + DistAbsTol)) FieldShouldBeZeroedDueToX = 1;
	}
	char FieldShouldBeZeroed = (FieldShouldBeZeroedDueToX || FieldShouldBeZeroedDueToZ);

	int ixcOld = int((xAbs - pOldRadAccessData->xStart) * xStepInvOld + 1.E-06);
	double xRel = xAbs - (pOldRadAccessData->xStart + ixcOld * pOldRadAccessData->xStep);

	if (ixcOld == nx_mi_1Old) { ixStOld = ixcOld - 3; xRel += 2. * pOldRadAccessData->xStep; }
	else if (ixcOld == nx_mi_2Old) { ixStOld = ixcOld - 2; xRel += pOldRadAccessData->xStep; }
	else if (ixcOld == 0) { ixStOld = ixcOld; xRel -= pOldRadAccessData->xStep; }
	else ixStOld = ixcOld - 1;

	xRel *= xStepInvOld;

	int ixcOld_mi_ixStOld = ixcOld - ixStOld;

	//or (int ie = 0; ie < pNewRadAccessData->ne; ie++)
	{
		//OC31102018: modified by SY at OpenMP parallelization
		//ixStOldPrev = -1000; izStOldPrev = -1000;

		//OC31102018: moved by SY at OpenMP parallelization
		srTInterpolAux01 InterpolAux01;
		srTInterpolAux02 InterpolAux02[4], InterpolAux02I[2];
		srTInterpolAuxF AuxF[4], AuxFI[2];
		//ixStOldPrev = -1000; izStOldPrev = -1000; //OC25012024 (commented-out: never used?)
		float BufF[4], BufFI[2];
		char UseLowOrderInterp_PolCompX = 0, UseLowOrderInterp_PolCompZ = 0;

		//long Two_ie = ie << 1;
		long long Two_ie = ie << 1;

		float* pEX_StartForX_New = 0, * pEZ_StartForX_New = 0;
		pEX_StartForX_New = pEX0_New + izPerZ_New;
		pEZ_StartForX_New = pEZ0_New + izPerZ_New;

		//long ixPerX_New_p_Two_ie = ix*PerX_New + Two_ie;
		long long ixPerX_New_p_Two_ie = ix * PerX_New + Two_ie;
		float* pEX_New = 0, * pEZ_New = 0;
		pEX_New = pEX_StartForX_New + ixPerX_New_p_Two_ie;
		pEZ_New = pEZ_StartForX_New + ixPerX_New_p_Two_ie;

		//long TotOffsetOld = izStOld*PerZ_Old + ixStOld*PerX_Old + Two_ie;
		long long TotOffsetOld = izStOld * PerZ_Old + ixStOld * PerX_Old + Two_ie;

		if (TreatPolCompX)
		{
			float* pExSt_Old = pEX0_Old + TotOffsetOld;
			srTGenOptElem::GetCellDataForInterpol(pExSt_Old, PerX_Old, PerZ_Old, AuxF);

			srTGenOptElem::SetupCellDataI(AuxF, AuxFI);
			UseLowOrderInterp_PolCompX = srTGenOptElem::CheckForLowOrderInterp(AuxF, AuxFI, ixcOld_mi_ixStOld, izcOld_mi_izStOld, &InterpolAux01, InterpolAux02, InterpolAux02I);

			if (!UseLowOrderInterp_PolCompX)
			{
				for (int i = 0; i < 2; i++)
				{
					srTGenOptElem::SetupInterpolAux02(AuxF + i, &InterpolAux01, InterpolAux02 + i);
				}
				srTGenOptElem::SetupInterpolAux02(AuxFI, &InterpolAux01, InterpolAux02I);
			}

			if (UseLowOrderInterp_PolCompX)
			{
				srTGenOptElem::InterpolF_LowOrder(InterpolAux02, xRel, zRel, BufF, 0);
				srTGenOptElem::InterpolFI_LowOrder(InterpolAux02I, xRel, zRel, BufFI, 0);
			}
			else
			{
				srTGenOptElem::InterpolF(InterpolAux02, xRel, zRel, BufF, 0);
				srTGenOptElem::InterpolFI(InterpolAux02I, xRel, zRel, BufFI, 0);
			}

			(*BufFI) *= AuxFI->fNorm;
			srTGenOptElem::ImproveReAndIm(BufF, BufFI);

			if (FieldShouldBeZeroed)
			{
				*BufF = 0.; *(BufF + 1) = 0.;
			}

			*pEX_New = *BufF;
			*(pEX_New + 1) = *(BufF + 1);
		}
		if (TreatPolCompZ)
		{
			float* pEzSt_Old = pEZ0_Old + TotOffsetOld;
			srTGenOptElem::GetCellDataForInterpol(pEzSt_Old, PerX_Old, PerZ_Old, AuxF + 2);

			srTGenOptElem::SetupCellDataI(AuxF + 2, AuxFI + 1);
			UseLowOrderInterp_PolCompZ = srTGenOptElem::CheckForLowOrderInterp(AuxF + 2, AuxFI + 1, ixcOld_mi_ixStOld, izcOld_mi_izStOld, &InterpolAux01, InterpolAux02 + 2, InterpolAux02I + 1);

			if (!UseLowOrderInterp_PolCompZ)
			{
				for (int i = 0; i < 2; i++)
				{
					srTGenOptElem::SetupInterpolAux02(AuxF + 2 + i, &InterpolAux01, InterpolAux02 + 2 + i);
				}
				srTGenOptElem::SetupInterpolAux02(AuxFI + 1, &InterpolAux01, InterpolAux02I + 1);
			}
			
			if (UseLowOrderInterp_PolCompZ)
			{
				srTGenOptElem::InterpolF_LowOrder(InterpolAux02, xRel, zRel, BufF, 2);
				srTGenOptElem::InterpolFI_LowOrder(InterpolAux02I, xRel, zRel, BufFI, 1);
			}
			else
			{
				srTGenOptElem::InterpolF(InterpolAux02, xRel, zRel, BufF, 2);
				srTGenOptElem::InterpolFI(InterpolAux02I, xRel, zRel, BufFI, 1);
			}

			(*(BufFI + 1)) *= (AuxFI + 1)->fNorm;
			srTGenOptElem::ImproveReAndIm(BufF + 2, BufFI + 1);

			if (FieldShouldBeZeroed)
			{
				*(BufF + 2) = 0.; *(BufF + 3) = 0.;
			}

			*pEZ_New = *(BufF + 2);
			*(pEZ_New + 1) = *(BufF + 3);
		}
	}
}

int srTGenOptElem::RadResizeCore_GPU(srTSRWRadStructAccessData& OldRadAccessData, srTSRWRadStructAccessData& NewRadAccessData, char PolComp, TGPUUsageArg* pGPU)
{
	char TreatPolCompX = ((PolComp == 0) || (PolComp == 'x'));
	char TreatPolCompZ = ((PolComp == 0) || (PolComp == 'z'));

	int nx = NewRadAccessData.AuxLong2 - NewRadAccessData.AuxLong1 + 1;
	int nz = NewRadAccessData.AuxLong4 - NewRadAccessData.AuxLong3 + 1;
	int ne = NewRadAccessData.ne;
	OldRadAccessData.pBaseRadX = (float*)CAuxGPU::ToDevice(pGPU, OldRadAccessData.pBaseRadX, 2*OldRadAccessData.ne*OldRadAccessData.nx*OldRadAccessData.nz*sizeof(float));
	OldRadAccessData.pBaseRadZ = (float*)CAuxGPU::ToDevice(pGPU, OldRadAccessData.pBaseRadZ, 2*OldRadAccessData.ne*OldRadAccessData.nx*OldRadAccessData.nz*sizeof(float));
	NewRadAccessData.pBaseRadX = (float*)CAuxGPU::ToDevice(pGPU, NewRadAccessData.pBaseRadX, 2*NewRadAccessData.ne*NewRadAccessData.nx*NewRadAccessData.nz*sizeof(float), true, false, 0);
	NewRadAccessData.pBaseRadZ = (float*)CAuxGPU::ToDevice(pGPU, NewRadAccessData.pBaseRadZ, 2*NewRadAccessData.ne*NewRadAccessData.nx*NewRadAccessData.nz*sizeof(float), true, false, 0);
	
	CAuxGPU::EnsureDeviceMemoryReady(pGPU, OldRadAccessData.pBaseRadX);
	CAuxGPU::EnsureDeviceMemoryReady(pGPU, OldRadAccessData.pBaseRadZ);
	CAuxGPU::EnsureDeviceMemoryReady(pGPU, NewRadAccessData.pBaseRadX);
	CAuxGPU::EnsureDeviceMemoryReady(pGPU, NewRadAccessData.pBaseRadZ);

	srTSRWRadStructAccessData* pOldRadAccessData_dev = (srTSRWRadStructAccessData*)CAuxGPU::ToDevice(pGPU, &OldRadAccessData, sizeof(srTSRWRadStructAccessData)); //HG27072024
	srTSRWRadStructAccessData* pNewRadAccessData_dev = (srTSRWRadStructAccessData*)CAuxGPU::ToDevice(pGPU, &NewRadAccessData, sizeof(srTSRWRadStructAccessData)); //HG27072024
	CAuxGPU::EnsureDeviceMemoryReady(pGPU, pOldRadAccessData_dev); //HG27072024
	CAuxGPU::EnsureDeviceMemoryReady(pGPU, pNewRadAccessData_dev); //HG27072024

	const int bs = 32;
	dim3 blocks(nx / bs + ((nx & (bs - 1)) != 0), nz, ne);
	dim3 threads(bs, 1);
	
	if (TreatPolCompX && TreatPolCompZ) RadResizeCore_Kernel<true, true> << <blocks, threads >> > (pOldRadAccessData_dev, pNewRadAccessData_dev);
	else if (TreatPolCompX) RadResizeCore_Kernel<true, false> << <blocks, threads >> > (pOldRadAccessData_dev, pNewRadAccessData_dev);
	else if (TreatPolCompZ) RadResizeCore_Kernel<false, true> << <blocks, threads >> > (pOldRadAccessData_dev, pNewRadAccessData_dev);

	CAuxGPU::ToHostAndFree(pGPU, pOldRadAccessData_dev, sizeof(srTSRWRadStructAccessData), true); //HG27072024
	CAuxGPU::ToHostAndFree(pGPU, pNewRadAccessData_dev, sizeof(srTSRWRadStructAccessData), true); //HG27072024

	OldRadAccessData.pBaseRadX = (float*)CAuxGPU::ToHostAndFree(pGPU, OldRadAccessData.pBaseRadX, 2*OldRadAccessData.ne*OldRadAccessData.nx*OldRadAccessData.nz*sizeof(float), true);
	OldRadAccessData.pBaseRadZ = (float*)CAuxGPU::ToHostAndFree(pGPU, OldRadAccessData.pBaseRadZ, 2*OldRadAccessData.ne*OldRadAccessData.nx*OldRadAccessData.nz*sizeof(float), true);
	//NewRadAccessData.pBaseRadX = CAuxGPU::ToHostAndFree(pGPU, NewRadAccessData.pBaseRadX, 2*NewRadAccessData.ne*NewRadAccessData.nx*NewRadAccessData.nz*sizeof(float));
	//NewRadAccessData.pBaseRadZ = CAuxGPU::ToHostAndFree(pGPU, NewRadAccessData.pBaseRadZ, 2*NewRadAccessData.ne*NewRadAccessData.nx*NewRadAccessData.nz*sizeof(float));
	CAuxGPU::MarkUpdated(pGPU, NewRadAccessData.pBaseRadX, true, false);
	CAuxGPU::MarkUpdated(pGPU, NewRadAccessData.pBaseRadZ, true, false);
//#ifndef _DEBUG
	NewRadAccessData.pBaseRadX = (float*)CAuxGPU::GetHostPtr(pGPU, NewRadAccessData.pBaseRadX);
	NewRadAccessData.pBaseRadZ = (float*)CAuxGPU::GetHostPtr(pGPU, NewRadAccessData.pBaseRadZ);
//#endif

//#ifdef _DEBUG
	//NewRadAccessData.pBaseRadX = (float*)CAuxGPU::ToHostAndFree(pGPU, NewRadAccessData.pBaseRadX, 2*NewRadAccessData.ne*NewRadAccessData.nx*NewRadAccessData.nz*sizeof(float), false);
	//NewRadAccessData.pBaseRadZ = (float*)CAuxGPU::ToHostAndFree(pGPU, NewRadAccessData.pBaseRadZ, 2*NewRadAccessData.ne*NewRadAccessData.nx*NewRadAccessData.nz*sizeof(float), false);
	//cudaStreamSynchronize(0);
	//auto err = cudaGetLastError();
	//printf("%s\r\n", cudaGetErrorString(err));

//#endif

	return 0;
}

template<bool TreatPolCompX, bool TreatPolCompZ> __global__ void RadResizeCore_OnlyLargerRange_Kernel(srTSRWRadStructAccessData* pOldRadAccessData, srTSRWRadStructAccessData* pNewRadAccessData)
{

	int ixStart = int(pNewRadAccessData->AuxLong1);
	int ixEnd = int(pNewRadAccessData->AuxLong2);
	int izStart = int(pNewRadAccessData->AuxLong3);
	int izEnd = int(pNewRadAccessData->AuxLong4);


	int ix = (blockIdx.x * blockDim.x + threadIdx.x) + ixStart; //nx range
	int iz = (blockIdx.y * blockDim.y + threadIdx.y) + izStart; //nz range
	int ie = (blockIdx.z * blockDim.z + threadIdx.z); //ne range

	if (ix > ixEnd) return;
	if (iz > izEnd) return;

	float* pEX0_New = pNewRadAccessData->pBaseRadX;
	float* pEZ0_New = pNewRadAccessData->pBaseRadZ;

	float* pEX0_Old = pOldRadAccessData->pBaseRadX;
	float* pEZ0_Old = pOldRadAccessData->pBaseRadZ;

	//long PerX_New = pNewRadAccessData->ne << 1;
	//long PerZ_New = PerX_New*pNewRadAccessData->nx;
	long long PerX_New = pNewRadAccessData->ne << 1;
	long long PerZ_New = PerX_New*pNewRadAccessData->nx;

	//long PerX_Old = PerX_New;
	//long PerZ_Old = PerX_Old*pOldRadAccessData->nx;
	long long PerX_Old = PerX_New;
	long long PerZ_Old = PerX_Old*pOldRadAccessData->nx;

	double xStepInvOld = 1./pOldRadAccessData->xStep;
	double zStepInvOld = 1./pOldRadAccessData->zStep;
	
	//long Two_ie = ie << 1;
	long long Two_ie = ie << 1;
	
	//long izPerZ_New = iz*PerZ_New;
	long long izPerZ_New = iz*PerZ_New;
	float* pEX_StartForX_New = pEX0_New + izPerZ_New;
	float* pEZ_StartForX_New = pEZ0_New + izPerZ_New;

	//long izPerZ_Old = (iz - izStart)*PerZ_Old;

	double zAbs = pNewRadAccessData->zStart + iz*pNewRadAccessData->zStep;
	long izOld = long((zAbs - pOldRadAccessData->zStart)*zStepInvOld + 1.E-08);
	//long izPerZ_Old = izOld*PerZ_Old;
	long long izPerZ_Old = izOld*PerZ_Old;

	float* pEX_StartForX_Old = pEX0_Old + izPerZ_Old;
	float* pEZ_StartForX_Old = pEZ0_Old + izPerZ_Old;

	//long ixPerX_New_p_Two_ie = ix*PerX_New + Two_ie;
	long long ixPerX_New_p_Two_ie = ix*PerX_New + Two_ie;
	float* pEX_New = pEX_StartForX_New + ixPerX_New_p_Two_ie;
	float* pEZ_New = pEZ_StartForX_New + ixPerX_New_p_Two_ie;

	//long ixPerX_Old_p_Two_ie = (ix - ixStart)*PerX_Old + Two_ie;

	double xAbs = pNewRadAccessData->xStart + ix*pNewRadAccessData->xStep;
	long ixOld = long((xAbs - pOldRadAccessData->xStart)*xStepInvOld + 1.E-08);
	//long ixPerX_Old_p_Two_ie = ixOld*PerX_Old + Two_ie;
	long long ixPerX_Old_p_Two_ie = ixOld*PerX_Old + Two_ie;

	float* pEX_Old = pEX_StartForX_Old + ixPerX_Old_p_Two_ie;
	float* pEZ_Old = pEZ_StartForX_Old + ixPerX_Old_p_Two_ie;

	if (TreatPolCompX) { *pEX_New = *pEX_Old; *(pEX_New + 1) = *(pEX_Old + 1); }
	if (TreatPolCompZ) { *pEZ_New = *pEZ_Old; *(pEZ_New + 1) = *(pEZ_Old + 1); }
}


int srTGenOptElem::RadResizeCore_OnlyLargerRange_GPU(srTSRWRadStructAccessData& OldRadAccessData, srTSRWRadStructAccessData& NewRadAccessData, char PolComp, TGPUUsageArg* pGPU)
{
	char TreatPolCompX = ((PolComp == 0) || (PolComp == 'x'));
	char TreatPolCompZ = ((PolComp == 0) || (PolComp == 'z'));

	int nx = NewRadAccessData.AuxLong2 - NewRadAccessData.AuxLong1 + 1;
	int nz = NewRadAccessData.AuxLong4 - NewRadAccessData.AuxLong3 + 1;
	int ne = NewRadAccessData.ne;

	OldRadAccessData.pBaseRadX = (float*)CAuxGPU::ToDevice(pGPU, OldRadAccessData.pBaseRadX, 2 * OldRadAccessData.ne * OldRadAccessData.nx * OldRadAccessData.nz * sizeof(float));
	OldRadAccessData.pBaseRadZ = (float*)CAuxGPU::ToDevice(pGPU, OldRadAccessData.pBaseRadZ, 2 * OldRadAccessData.ne * OldRadAccessData.nx * OldRadAccessData.nz * sizeof(float));
	NewRadAccessData.pBaseRadX = (float*)CAuxGPU::ToDevice(pGPU, NewRadAccessData.pBaseRadX, 2 * NewRadAccessData.ne * NewRadAccessData.nx * NewRadAccessData.nz * sizeof(float), true, false, 1);
	NewRadAccessData.pBaseRadZ = (float*)CAuxGPU::ToDevice(pGPU, NewRadAccessData.pBaseRadZ, 2 * NewRadAccessData.ne * NewRadAccessData.nx * NewRadAccessData.nz * sizeof(float), true, false, 1);

	CAuxGPU::EnsureDeviceMemoryReady(pGPU, OldRadAccessData.pBaseRadX);
	CAuxGPU::EnsureDeviceMemoryReady(pGPU, OldRadAccessData.pBaseRadZ);
	CAuxGPU::EnsureDeviceMemoryReady(pGPU, NewRadAccessData.pBaseRadX);
	CAuxGPU::EnsureDeviceMemoryReady(pGPU, NewRadAccessData.pBaseRadZ);

	srTSRWRadStructAccessData* pOldRadAccessData_dev = (srTSRWRadStructAccessData*)CAuxGPU::ToDevice(pGPU, &OldRadAccessData, sizeof(srTSRWRadStructAccessData));
	srTSRWRadStructAccessData* pNewRadAccessData_dev = (srTSRWRadStructAccessData*)CAuxGPU::ToDevice(pGPU, &NewRadAccessData, sizeof(srTSRWRadStructAccessData));
	CAuxGPU::EnsureDeviceMemoryReady(pGPU, pOldRadAccessData_dev);
	CAuxGPU::EnsureDeviceMemoryReady(pGPU, pNewRadAccessData_dev);

	const int bs = 32;
	dim3 blocks(nx / bs + ((nx & (bs - 1)) != 0), nz, ne);
	dim3 threads(bs, 1);

	if (TreatPolCompX && TreatPolCompZ) RadResizeCore_OnlyLargerRange_Kernel<true, true> << <blocks, threads >> > (pOldRadAccessData_dev, pNewRadAccessData_dev);
	else if (TreatPolCompX) RadResizeCore_OnlyLargerRange_Kernel<true, false> << <blocks, threads >> > (pOldRadAccessData_dev, pNewRadAccessData_dev);
	else if (TreatPolCompZ) RadResizeCore_OnlyLargerRange_Kernel<false, true> << <blocks, threads >> > (pOldRadAccessData_dev, pNewRadAccessData_dev);

	CAuxGPU::ToHostAndFree(pGPU, pOldRadAccessData_dev, sizeof(srTSRWRadStructAccessData), true);
	CAuxGPU::ToHostAndFree(pGPU, pNewRadAccessData_dev, sizeof(srTSRWRadStructAccessData), true);

	OldRadAccessData.pBaseRadX = (float*)CAuxGPU::ToHostAndFree(pGPU, OldRadAccessData.pBaseRadX, 2 * OldRadAccessData.ne * OldRadAccessData.nx * OldRadAccessData.nz * sizeof(float), true);
	OldRadAccessData.pBaseRadZ = (float*)CAuxGPU::ToHostAndFree(pGPU, OldRadAccessData.pBaseRadZ, 2 * OldRadAccessData.ne * OldRadAccessData.nx * OldRadAccessData.nz * sizeof(float), true);
	
	CAuxGPU::MarkUpdated(pGPU, NewRadAccessData.pBaseRadX, true, false);
	CAuxGPU::MarkUpdated(pGPU, NewRadAccessData.pBaseRadZ, true, false);
	
	NewRadAccessData.pBaseRadX = (float*)CAuxGPU::GetHostPtr(pGPU, NewRadAccessData.pBaseRadX);
	NewRadAccessData.pBaseRadZ = (float*)CAuxGPU::GetHostPtr(pGPU, NewRadAccessData.pBaseRadZ);
	
	return 0;
}

template<bool TreatPolCompX, bool TreatPolCompZ> __global__ void RadResizeCore_OnlyLargerRangeE_Kernel(srTSRWRadStructAccessData* pOldRadAccessData, srTSRWRadStructAccessData* pNewRadAccessData)
{
	int ieStart = int(pNewRadAccessData->AuxLong1);
	
	int ix = (blockIdx.x * blockDim.x + threadIdx.x); //nx range
	int iz = (blockIdx.y * blockDim.y + threadIdx.y); //nz range
	int ie = (blockIdx.z * blockDim.z + threadIdx.z) + ieStart; //ne range

	if (ix > pNewRadAccessData->nx) return;
	if (iz > pNewRadAccessData->nz) return;
	
	float* pEX0_New = pNewRadAccessData->pBaseRadX;
	float* pEZ0_New = pNewRadAccessData->pBaseRadZ;

	float* pEX0_Old = pOldRadAccessData->pBaseRadX;
	float* pEZ0_Old = pOldRadAccessData->pBaseRadZ;

	//long PerX_New = pNewRadAccessData->ne << 1;
	//long PerZ_New = PerX_New*pNewRadAccessData->nx;
	long long PerX_New = pNewRadAccessData->ne << 1;
	long long PerZ_New = PerX_New * pNewRadAccessData->nx;

	//long PerX_Old = pOldRadAccessData->ne << 1;
	//long PerZ_Old = PerX_Old*pOldRadAccessData->nx;
	long long PerX_Old = pOldRadAccessData->ne << 1;
	long long PerZ_Old = PerX_Old * pOldRadAccessData->nx;

	double eStepInvOld = 1. / pOldRadAccessData->eStep;

	//long iz_PerZ_New = iz*PerZ_New;
	//long iz_PerZ_Old = iz*PerZ_Old;
	long long iz_PerZ_New = iz * PerZ_New;
	long long iz_PerZ_Old = iz * PerZ_Old;

	//long iz_PerZ_New_p_ix_PerX_New = iz_PerZ_New + ix*PerX_New;
	//long iz_PerZ_Old_p_ix_PerX_Old = iz_PerZ_Old + ix*PerX_Old;
	long long iz_PerZ_New_p_ix_PerX_New = iz_PerZ_New + ix * PerX_New;
	long long iz_PerZ_Old_p_ix_PerX_Old = iz_PerZ_Old + ix * PerX_Old;

	//long ofstNew = iz_PerZ_New_p_ix_PerX_New + (ie << 1);
	long long ofstNew = iz_PerZ_New_p_ix_PerX_New + (ie << 1);
	float* pEX_New = pEX0_New + ofstNew;
	float* pEZ_New = pEZ0_New + ofstNew;

	double eAbs = pNewRadAccessData->eStart + ie * pNewRadAccessData->eStep;
	long ieOld = long((eAbs - pOldRadAccessData->eStart) * eStepInvOld + 1.E-08);

	//long ofstOld = iz_PerZ_Old_p_ix_PerX_Old + (ieOld << 1);
	long long ofstOld = iz_PerZ_Old_p_ix_PerX_Old + (ieOld << 1);
	float* pEX_Old = pEX0_Old + ofstOld;
	float* pEZ_Old = pEZ0_Old + ofstOld;

	if (TreatPolCompX) { *pEX_New = *pEX_Old; *(pEX_New + 1) = *(pEX_Old + 1); }
	if (TreatPolCompZ) { *pEZ_New = *pEZ_Old; *(pEZ_New + 1) = *(pEZ_Old + 1); }
}


int srTGenOptElem::RadResizeCore_OnlyLargerRangeE_GPU(srTSRWRadStructAccessData& OldRadAccessData, srTSRWRadStructAccessData& NewRadAccessData, char PolComp, TGPUUsageArg* pGPU)
{
	char TreatPolCompX = ((PolComp == 0) || (PolComp == 'x')) && (OldRadAccessData.pBaseRadX != 0);
	char TreatPolCompZ = ((PolComp == 0) || (PolComp == 'z')) && (OldRadAccessData.pBaseRadZ != 0);

	int ne = NewRadAccessData.AuxLong2 - NewRadAccessData.AuxLong1 + 1;

	OldRadAccessData.pBaseRadX = (float*)CAuxGPU::ToDevice(pGPU, OldRadAccessData.pBaseRadX, 2 * OldRadAccessData.ne * OldRadAccessData.nx * OldRadAccessData.nz * sizeof(float));
	OldRadAccessData.pBaseRadZ = (float*)CAuxGPU::ToDevice(pGPU, OldRadAccessData.pBaseRadZ, 2 * OldRadAccessData.ne * OldRadAccessData.nx * OldRadAccessData.nz * sizeof(float));
	NewRadAccessData.pBaseRadX = (float*)CAuxGPU::ToDevice(pGPU, NewRadAccessData.pBaseRadX, 2 * NewRadAccessData.ne * NewRadAccessData.nx * NewRadAccessData.nz * sizeof(float), true);
	NewRadAccessData.pBaseRadZ = (float*)CAuxGPU::ToDevice(pGPU, NewRadAccessData.pBaseRadZ, 2 * NewRadAccessData.ne * NewRadAccessData.nx * NewRadAccessData.nz * sizeof(float), true);

	CAuxGPU::EnsureDeviceMemoryReady(pGPU, OldRadAccessData.pBaseRadX);
	CAuxGPU::EnsureDeviceMemoryReady(pGPU, OldRadAccessData.pBaseRadZ);
	CAuxGPU::EnsureDeviceMemoryReady(pGPU, NewRadAccessData.pBaseRadX);
	CAuxGPU::EnsureDeviceMemoryReady(pGPU, NewRadAccessData.pBaseRadZ);

	srTSRWRadStructAccessData* pOldRadAccessData_dev = (srTSRWRadStructAccessData*)CAuxGPU::ToDevice(pGPU, &OldRadAccessData, sizeof(srTSRWRadStructAccessData)); //HG27072024
	srTSRWRadStructAccessData* pNewRadAccessData_dev = (srTSRWRadStructAccessData*)CAuxGPU::ToDevice(pGPU, &NewRadAccessData, sizeof(srTSRWRadStructAccessData)); //HG27072024
	CAuxGPU::EnsureDeviceMemoryReady(pGPU, pOldRadAccessData_dev);
	CAuxGPU::EnsureDeviceMemoryReady(pGPU, pNewRadAccessData_dev);

	const int bs = 32;
	dim3 blocks(NewRadAccessData.nx / bs + ((NewRadAccessData.nx & (bs - 1)) != 0), NewRadAccessData.nz, ne);
	dim3 threads(bs, 1);

	if (TreatPolCompX && TreatPolCompZ) RadResizeCore_OnlyLargerRangeE_Kernel<true, true> << <blocks, threads >> > (pOldRadAccessData_dev, pNewRadAccessData_dev);
	else if (TreatPolCompX) RadResizeCore_OnlyLargerRangeE_Kernel<true, false> << <blocks, threads >> > (pOldRadAccessData_dev, pNewRadAccessData_dev);
	else if (TreatPolCompZ) RadResizeCore_OnlyLargerRangeE_Kernel<false, true> << <blocks, threads >> > (pOldRadAccessData_dev, pNewRadAccessData_dev);

	CAuxGPU::ToHostAndFree(pGPU, pOldRadAccessData_dev, sizeof(srTSRWRadStructAccessData), true); //HG27072024
	CAuxGPU::ToHostAndFree(pGPU, pNewRadAccessData_dev, sizeof(srTSRWRadStructAccessData), true); //HG27072024

	OldRadAccessData.pBaseRadX = (float*)CAuxGPU::ToHostAndFree(pGPU, OldRadAccessData.pBaseRadX, 2 * OldRadAccessData.ne * OldRadAccessData.nx * OldRadAccessData.nz * sizeof(float), true);
	OldRadAccessData.pBaseRadZ = (float*)CAuxGPU::ToHostAndFree(pGPU, OldRadAccessData.pBaseRadZ, 2 * OldRadAccessData.ne * OldRadAccessData.nx * OldRadAccessData.nz * sizeof(float), true);
	//NewRadAccessData.pBaseRadX = CAuxGPU::ToHostAndFree(pGPU, NewRadAccessData.pBaseRadX, 2*NewRadAccessData.ne*NewRadAccessData.nx*NewRadAccessData.nz*sizeof(float));
	//NewRadAccessData.pBaseRadZ = CAuxGPU::ToHostAndFree(pGPU, NewRadAccessData.pBaseRadZ, 2*NewRadAccessData.ne*NewRadAccessData.nx*NewRadAccessData.nz*sizeof(float));
	CAuxGPU::MarkUpdated(pGPU, NewRadAccessData.pBaseRadX, true, false);
	CAuxGPU::MarkUpdated(pGPU, NewRadAccessData.pBaseRadZ, true, false);
	//#ifndef _DEBUG
	NewRadAccessData.pBaseRadX = (float*)CAuxGPU::GetHostPtr(pGPU, NewRadAccessData.pBaseRadX);
	NewRadAccessData.pBaseRadZ = (float*)CAuxGPU::GetHostPtr(pGPU, NewRadAccessData.pBaseRadZ);
	//#endif

	//#ifdef _DEBUG
	//NewRadAccessData.pBaseRadX = (float*)CAuxGPU::ToHostAndFree(pGPU, NewRadAccessData.pBaseRadX, 2 * NewRadAccessData.ne * NewRadAccessData.nx * NewRadAccessData.nz * sizeof(float), false);
	//NewRadAccessData.pBaseRadZ = (float*)CAuxGPU::ToHostAndFree(pGPU, NewRadAccessData.pBaseRadZ, 2 * NewRadAccessData.ne * NewRadAccessData.nx * NewRadAccessData.nz * sizeof(float), false);
	//cudaStreamSynchronize(0);
	//auto err = cudaGetLastError();
	//printf("%s\r\n", cudaGetErrorString(err));

	//#endif

	return 0;
}

__global__ void ExtractRadSliceConstE_Kernel(srTSRWRadStructAccessData *pRadAccessData, long ie, float* pOutEx, float* pOutEz)
{
	int ix = (blockIdx.x * blockDim.x + threadIdx.x); //nx range
	int iz = (blockIdx.y * blockDim.y + threadIdx.y); //nz range
	
	if(ix >= pRadAccessData->nx) return;
	if(iz >= pRadAccessData->nz) return;

	float *pEx0 = pRadAccessData->pBaseRadX;
	float *pEz0 = pRadAccessData->pBaseRadZ;
	long long  PerX = pRadAccessData->ne << 1;
	long long PerZ = PerX*pRadAccessData->nx;
	long long izPerZ = iz * PerZ;
	long long ixPerX = ix * PerX;
	long long iePerE = ie << 1;
	long long ixPerX_p_iePerE = ixPerX + iePerE;
	
	float *tOutEx = pOutEx, *tOutEz = pOutEz;
	float *pEx = pEx0 + izPerZ + ixPerX_p_iePerE;
	float *pEz = pEz0 + izPerZ + ixPerX_p_iePerE;

	tOutEx += (iz*pRadAccessData->nx + ix) << 1;
	*tOutEx = *pEx; *(tOutEx + 1) = *(pEx + 1);

	tOutEz += (iz*pRadAccessData->nx + ix) << 1;
	*tOutEz = *pEz; *(tOutEz + 1) = *(pEz + 1);
}

int srTGenOptElem::ExtractRadSliceConstE_GPU(srTSRWRadStructAccessData* pRadAccessData, long ie, float* pOutEx, float* pOutEz, TGPUUsageArg* pGPU)
{
	//printf("ExtractRadSliceConstE_GPU\r\n Data Size: %llu\r\n", 2 * pRadAccessData->ne * pRadAccessData->nx * pRadAccessData->nz * sizeof(float));
	//printf("Dst Size: %llu\r\n", 2 * pRadAccessData->nx * pRadAccessData->nz * sizeof(float));
	pRadAccessData->pBaseRadX = (float*)CAuxGPU::ToDevice(pGPU, pRadAccessData->pBaseRadX, 2 * pRadAccessData->ne * pRadAccessData->nx * pRadAccessData->nz * sizeof(float));
	pRadAccessData->pBaseRadZ = (float*)CAuxGPU::ToDevice(pGPU, pRadAccessData->pBaseRadZ, 2 * pRadAccessData->ne * pRadAccessData->nx * pRadAccessData->nz * sizeof(float));
	pOutEx = (float*)CAuxGPU::ToDevice(pGPU, pOutEx, 2 * pRadAccessData->nx * pRadAccessData->nz * sizeof(float), true);
	pOutEz = (float*)CAuxGPU::ToDevice(pGPU, pOutEz, 2 * pRadAccessData->nx * pRadAccessData->nz * sizeof(float), true);
	
	CAuxGPU::EnsureDeviceMemoryReady(pGPU, pRadAccessData->pBaseRadX);
	CAuxGPU::EnsureDeviceMemoryReady(pGPU, pRadAccessData->pBaseRadZ);
	CAuxGPU::EnsureDeviceMemoryReady(pGPU, pOutEx);
	CAuxGPU::EnsureDeviceMemoryReady(pGPU, pOutEz);
	
	srTSRWRadStructAccessData* pRadAccessData_dev = (srTSRWRadStructAccessData*)CAuxGPU::ToDevice(pGPU, pRadAccessData, sizeof(srTSRWRadStructAccessData));
	CAuxGPU::EnsureDeviceMemoryReady(pGPU, pRadAccessData_dev);

	const int bs = 256;
	dim3 blocks(pRadAccessData->nx / bs + ((pRadAccessData->nx & (bs - 1)) != 0), pRadAccessData->nz, 1);
	dim3 threads(bs, 1);

	ExtractRadSliceConstE_Kernel <<<blocks, threads >>> (pRadAccessData_dev, ie, pOutEx, pOutEz);

	CAuxGPU::ToHostAndFree(pGPU, pRadAccessData_dev, sizeof(srTSRWRadStructAccessData), true);
	
	CAuxGPU::MarkUpdated(pGPU, pOutEx, true, false);
	CAuxGPU::MarkUpdated(pGPU, pOutEz, true, false);
	pRadAccessData->pBaseRadX = (float*)CAuxGPU::GetHostPtr(pGPU, pRadAccessData->pBaseRadX);
	pRadAccessData->pBaseRadZ = (float*)CAuxGPU::GetHostPtr(pGPU, pRadAccessData->pBaseRadZ);
	pOutEx = (float*)CAuxGPU::GetHostPtr(pGPU, pOutEx);
	pOutEz = (float*)CAuxGPU::GetHostPtr(pGPU, pOutEz);

	//pRadAccessData->pBaseRadX = (float*)CAuxGPU::ToHostAndFree(pGPU, pRadAccessData->pBaseRadX, 2 * pRadAccessData->ne * pRadAccessData->nx * pRadAccessData->nz * sizeof(float), true);
	//pRadAccessData->pBaseRadZ = (float*)CAuxGPU::ToHostAndFree(pGPU, pRadAccessData->pBaseRadZ, 2 * pRadAccessData->ne * pRadAccessData->nx * pRadAccessData->nz * sizeof(float), true);
	return 0;
}

__global__ void UpdateGenRadStructSliceConstE_Meth_0_Kernel(srTSRWRadStructAccessData* pRadDataSliceConstE, int ie, srTSRWRadStructAccessData* pRadAccessData)
{
	int ix = (blockIdx.x * blockDim.x + threadIdx.x); //nx range
	int iz = (blockIdx.y * blockDim.y + threadIdx.y); //nz range
	
	int neCom = pRadAccessData->ne;
	int nxCom = pRadAccessData->nx;
	int nzCom = pRadAccessData->nz;

	if(ix >= nxCom) return;
	if(iz >= nzCom) return;
	
	float *pEx0 = pRadAccessData->pBaseRadX;
	float *pEz0 = pRadAccessData->pBaseRadZ;

	long long PerX = neCom << 1;
	long long PerZ = PerX*nxCom;
	long long iePerE = ie << 1;

	float *tSliceEx = pRadDataSliceConstE->pBaseRadX + iz*nxCom*2 + ix*2;
	float *tSliceEz = pRadDataSliceConstE->pBaseRadZ + iz*nxCom*2 + ix*2;

	float *pEx = pEx0 + iz*PerZ + ix*PerX + iePerE;
	float *pEz = pEz0 + iz*PerZ + ix*PerX + iePerE;

	*(pEx++) = *(tSliceEx++); *pEx = *(tSliceEx);
	*(pEz++) = *(tSliceEz++); *pEz = *(tSliceEz);
}

int srTGenOptElem::UpdateGenRadStructSliceConstE_Meth_0_GPU(srTSRWRadStructAccessData* pRadDataSliceConstE, int ie, srTSRWRadStructAccessData* pRadAccessData, TGPUUsageArg* pGPU)
{
	pRadAccessData->pBaseRadX = (float*)CAuxGPU::ToDevice(pGPU, pRadAccessData->pBaseRadX, 2 * pRadAccessData->ne * pRadAccessData->nx * pRadAccessData->nz * sizeof(float));
	pRadAccessData->pBaseRadZ = (float*)CAuxGPU::ToDevice(pGPU, pRadAccessData->pBaseRadZ, 2 * pRadAccessData->ne * pRadAccessData->nx * pRadAccessData->nz * sizeof(float));
	pRadDataSliceConstE->pBaseRadX = (float*)CAuxGPU::ToDevice(pGPU, pRadDataSliceConstE->pBaseRadX, 2 * pRadAccessData->nx * pRadAccessData->nz * sizeof(float));
	pRadDataSliceConstE->pBaseRadZ = (float*)CAuxGPU::ToDevice(pGPU, pRadDataSliceConstE->pBaseRadZ, 2 * pRadAccessData->nx * pRadAccessData->nz * sizeof(float));
	
	CAuxGPU::EnsureDeviceMemoryReady(pGPU, pRadAccessData->pBaseRadX);
	CAuxGPU::EnsureDeviceMemoryReady(pGPU, pRadAccessData->pBaseRadZ);
	CAuxGPU::EnsureDeviceMemoryReady(pGPU, pRadDataSliceConstE->pBaseRadX);
	CAuxGPU::EnsureDeviceMemoryReady(pGPU, pRadDataSliceConstE->pBaseRadZ);
	
	srTSRWRadStructAccessData* pRadDataSliceConstE_dev = (srTSRWRadStructAccessData*)CAuxGPU::ToDevice(pGPU, pRadDataSliceConstE, sizeof(srTSRWRadStructAccessData));
	CAuxGPU::EnsureDeviceMemoryReady(pGPU, pRadDataSliceConstE_dev);

	srTSRWRadStructAccessData* pRadAccessData_dev = (srTSRWRadStructAccessData*)CAuxGPU::ToDevice(pGPU, pRadAccessData, sizeof(srTSRWRadStructAccessData));
	CAuxGPU::EnsureDeviceMemoryReady(pGPU, pRadAccessData_dev);

	const int bs = 256;
	dim3 blocks(pRadAccessData->nx / bs + ((pRadAccessData->nx & (bs - 1)) != 0), pRadAccessData->nz, 1);
	dim3 threads(bs, 1);

	UpdateGenRadStructSliceConstE_Meth_0_Kernel <<<blocks, threads >>> (pRadDataSliceConstE_dev, ie, pRadAccessData_dev);

	CAuxGPU::ToHostAndFree(pGPU, pRadAccessData_dev, sizeof(srTSRWRadStructAccessData), true);
	CAuxGPU::ToHostAndFree(pGPU, pRadDataSliceConstE_dev, sizeof(srTSRWRadStructAccessData), true);
	
	pRadDataSliceConstE->pBaseRadX = (float*)CAuxGPU::ToHostAndFree(pGPU, pRadDataSliceConstE->pBaseRadX, 2 * pRadAccessData->nx * pRadAccessData->nz * sizeof(float), true);
	pRadDataSliceConstE->pBaseRadZ = (float*)CAuxGPU::ToHostAndFree(pGPU, pRadDataSliceConstE->pBaseRadZ, 2 * pRadAccessData->nx * pRadAccessData->nz * sizeof(float), true);

	CAuxGPU::MarkUpdated(pGPU, pRadAccessData->pBaseRadX, true, false);
	CAuxGPU::MarkUpdated(pGPU, pRadAccessData->pBaseRadZ, true, false);
	//pRadAccessData->pBaseRadX = (float*)CAuxGPU::GetHostPtr(pGPU, pRadAccessData->pBaseRadX);
	//pRadAccessData->pBaseRadZ = (float*)CAuxGPU::GetHostPtr(pGPU, pRadAccessData->pBaseRadZ);
	
	//pRadAccessData->pBaseRadX = (float*)CAuxGPU::ToHostAndFree(pGPU, pRadAccessData->pBaseRadX, 2 * pRadAccessData->ne * pRadAccessData->nx * pRadAccessData->nz * sizeof(float));
	//pRadAccessData->pBaseRadZ = (float*)CAuxGPU::ToHostAndFree(pGPU, pRadAccessData->pBaseRadZ, 2 * pRadAccessData->ne * pRadAccessData->nx * pRadAccessData->nz * sizeof(float));
	
	return 0;
}

__global__ void ReInterpolateWfrSliceSingleE_Kernel(srTSRWRadStructAccessData *pOldRadSingleE, srTSRWRadStructAccessData *pNewRadMultiE, int ie)
{
	const double DistAbsTol = 1.E-10;
	bool TreatPolCompX=true, TreatPolCompZ=true;

	int ixStart = 0; //int(NewRadAccessData.AuxLong1);
	int ixEnd = pNewRadMultiE->nx - 1; //int(NewRadAccessData.AuxLong2);
	int izStart = 0; //int(NewRadAccessData.AuxLong3);
	int izEnd = pNewRadMultiE->nz - 1; //int(NewRadAccessData.AuxLong4);

	int ix = (blockIdx.x * blockDim.x + threadIdx.x) + ixStart; //nx range
	int iz = (blockIdx.y * blockDim.y + threadIdx.y) + izStart; //nz range

	if(ix > ixEnd) return;
	if(iz > izEnd) return;

	double xStepInvOld = 1./pOldRadSingleE->xStep;
	double zStepInvOld = 1./pOldRadSingleE->zStep;
	int nx_mi_1Old = pOldRadSingleE->nx - 1;
	int nz_mi_1Old = pOldRadSingleE->nz - 1;
	int nx_mi_2Old = nx_mi_1Old - 1;
	int nz_mi_2Old = nz_mi_1Old - 1;

	srTInterpolAux01 InterpolAux01;
	srTInterpolAux02 InterpolAux02[4], InterpolAux02I[2];
	srTInterpolAuxF AuxF[4], AuxFI[2];
	int ixStOld, izStOld, ixStOldPrev = -1000, izStOldPrev = -1000;

	float *pEX0_New = 0, *pEZ0_New = 0;
	if(TreatPolCompX) pEX0_New = pNewRadMultiE->pBaseRadX;
	if(TreatPolCompZ) pEZ0_New = pNewRadMultiE->pBaseRadZ;

	//long PerX_New = pNewRadMultiE->ne << 1;
	//long PerZ_New = PerX_New*pNewRadMultiE->nx;
	long long PerX_New = pNewRadMultiE->ne << 1;
	long long PerZ_New = PerX_New*pNewRadMultiE->nx;

	//long PerX_Old = 2; //PerX_New;
	//long PerZ_Old = PerX_Old*pOldRadSingleE->nx;
	long long PerX_Old = 2; //PerX_New;
	long long PerZ_Old = PerX_Old*pOldRadSingleE->nx;

	float BufF[4], BufFI[2];
	int UseLowOrderInterp_PolCompX, UseLowOrderInterp_PolCompZ;
	int result = 0;

	//for(int ie=0; ie<NewRadAccessData.ne; ie++)
	//{
	//ixStOldPrev = -1000; izStOldPrev = -1000;

	//long Two_ie = ie << 1;
	long long Two_ie = ie << 1;

	double zAbs = pNewRadMultiE->zStart + iz*pNewRadMultiE->zStep;
	char FieldShouldBeZeroedDueToZ = 0;
	if(pNewRadMultiE->WfrEdgeCorrShouldBeDone)
	{
		if((zAbs < pNewRadMultiE->zWfrMin - DistAbsTol) || (zAbs > pNewRadMultiE->zWfrMax + DistAbsTol)) FieldShouldBeZeroedDueToZ = 1;
	}
	int izcOld = int((zAbs - pOldRadSingleE->zStart)*zStepInvOld + 1.E-06);
	if((izcOld < 0) || (izcOld > nz_mi_1Old))
	{
		//set El. field to 0 for all ix
		FieldShouldBeZeroedDueToZ = 1;
	}

	double zRel = zAbs - (pOldRadSingleE->zStart + izcOld*pOldRadSingleE->zStep);

	if(izcOld == nz_mi_1Old) { izStOld = izcOld - 3; zRel += 2.*pOldRadSingleE->zStep;}
	else if(izcOld == nz_mi_2Old) { izStOld = izcOld - 2; zRel += pOldRadSingleE->zStep;}
	else if(izcOld == 0) { izStOld = izcOld; zRel -= pOldRadSingleE->zStep;}
	else izStOld = izcOld - 1;

	zRel *= zStepInvOld;
	int izcOld_mi_izStOld = izcOld - izStOld;
	//long izPerZ_New = iz*PerZ_New;
	long long izPerZ_New = iz*PerZ_New;

	float *pEX_StartForX_New = 0, *pEZ_StartForX_New = 0;
	if(TreatPolCompX) pEX_StartForX_New = pEX0_New + izPerZ_New;
	if(TreatPolCompZ) pEZ_StartForX_New = pEZ0_New + izPerZ_New;

	//long ixPerX_New_p_Two_ie = ix*PerX_New + Two_ie;
	long long ixPerX_New_p_Two_ie = ix*PerX_New + Two_ie;
	float *pEX_New = 0, *pEZ_New = 0;
	if(TreatPolCompX) pEX_New = pEX_StartForX_New + ixPerX_New_p_Two_ie;
	if(TreatPolCompZ) pEZ_New = pEZ_StartForX_New + ixPerX_New_p_Two_ie;

	double xAbs = pNewRadMultiE->xStart + ix*pNewRadMultiE->xStep;
	char FieldShouldBeZeroedDueToX = 0;
	if(pNewRadMultiE->WfrEdgeCorrShouldBeDone)
	{
		if((xAbs < pNewRadMultiE->xWfrMin - DistAbsTol) || (xAbs > pNewRadMultiE->xWfrMax + DistAbsTol)) FieldShouldBeZeroedDueToX = 1;
	}

	int ixcOld = int((xAbs - pOldRadSingleE->xStart)*xStepInvOld + 1.E-06);
	if((ixcOld < 0) || (ixcOld > nx_mi_1Old))
	{
		FieldShouldBeZeroedDueToX = 1;
	}
	char FieldShouldBeZeroed = (FieldShouldBeZeroedDueToX || FieldShouldBeZeroedDueToZ);

	if(FieldShouldBeZeroed)
	{
		//*BufF = 0.; *(BufF+1) = 0.;
		if(TreatPolCompX)
		{
			*pEX_New = 0.;
			*(pEX_New+1) = 0.;
		}
		if(TreatPolCompZ)
		{
			*pEZ_New = 0.;
			*(pEZ_New+1) = 0.;
		}
		return;
	}

	double xRel = xAbs - (pOldRadSingleE->xStart + ixcOld*pOldRadSingleE->xStep);

	if(ixcOld == nx_mi_1Old) { ixStOld = ixcOld - 3; xRel += 2.*pOldRadSingleE->xStep;}
	else if(ixcOld == nx_mi_2Old) { ixStOld = ixcOld - 2; xRel += pOldRadSingleE->xStep;}
	else if(ixcOld == 0) { ixStOld = ixcOld; xRel -= pOldRadSingleE->xStep;}
	else ixStOld = ixcOld - 1;

	xRel *= xStepInvOld;
	int ixcOld_mi_ixStOld = ixcOld - ixStOld;

	if((izStOld != izStOldPrev) || (ixStOld != ixStOldPrev))
	{
		UseLowOrderInterp_PolCompX = 0, UseLowOrderInterp_PolCompZ = 0;
		//long TotOffsetOld = izStOld*PerZ_Old + ixStOld*PerX_Old + Two_ie;
		//long TotOffsetOld = izStOld*PerZ_Old + ixStOld*PerX_Old; //old is single slice
		long long TotOffsetOld = izStOld*PerZ_Old + ixStOld*PerX_Old; //old is single slice

		if(TreatPolCompX)
		{
			float* pExSt_Old = pOldRadSingleE->pBaseRadX + TotOffsetOld;
			srTGenOptElem::GetCellDataForInterpol(pExSt_Old, PerX_Old, PerZ_Old, AuxF);
			srTGenOptElem::SetupCellDataI(AuxF, AuxFI);
			UseLowOrderInterp_PolCompX = srTGenOptElem::CheckForLowOrderInterp(AuxF, AuxFI, ixcOld_mi_ixStOld, izcOld_mi_izStOld, &InterpolAux01, InterpolAux02, InterpolAux02I);

			if(!UseLowOrderInterp_PolCompX)
			{
				for(int i=0; i<2; i++) 
				{
					srTGenOptElem::SetupInterpolAux02(AuxF + i, &InterpolAux01, InterpolAux02 + i);
				}
				srTGenOptElem::SetupInterpolAux02(AuxFI, &InterpolAux01, InterpolAux02I);
			}
		}
		if(TreatPolCompZ)
		{
			float* pEzSt_Old = pOldRadSingleE->pBaseRadZ + TotOffsetOld;
			srTGenOptElem::GetCellDataForInterpol(pEzSt_Old, PerX_Old, PerZ_Old, AuxF+2);
			srTGenOptElem::SetupCellDataI(AuxF+2, AuxFI+1);
			UseLowOrderInterp_PolCompZ = srTGenOptElem::CheckForLowOrderInterp(AuxF+2, AuxFI+1, ixcOld_mi_ixStOld, izcOld_mi_izStOld, &InterpolAux01, InterpolAux02+2, InterpolAux02I+1);

			if(!UseLowOrderInterp_PolCompZ)
			{
				for(int i=0; i<2; i++) 
				{
					srTGenOptElem::SetupInterpolAux02(AuxF+2+i, &InterpolAux01, InterpolAux02+2+i);
				}
				srTGenOptElem::SetupInterpolAux02(AuxFI+1, &InterpolAux01, InterpolAux02I+1);
			}
		}
		ixStOldPrev = ixStOld; izStOldPrev = izStOld;
	}

	if(TreatPolCompX)
	{
		if(UseLowOrderInterp_PolCompX) 
		{
			srTGenOptElem::InterpolF_LowOrder(InterpolAux02, xRel, zRel, BufF, 0);
			srTGenOptElem::InterpolFI_LowOrder(InterpolAux02I, xRel, zRel, BufFI, 0);
		}
		else
		{
			srTGenOptElem::InterpolF(InterpolAux02, xRel, zRel, BufF, 0);
			srTGenOptElem::InterpolFI(InterpolAux02I, xRel, zRel, BufFI, 0);
		}

		(*BufFI) *= AuxFI->fNorm;
		srTGenOptElem::ImproveReAndIm(BufF, BufFI);

		//if(FieldShouldBeZeroed)
		//{
		//	*BufF = 0.; *(BufF+1) = 0.;
		//}

		*pEX_New = *BufF;
		*(pEX_New+1) = *(BufF+1);
	}
	if(TreatPolCompZ)
	{
		if(UseLowOrderInterp_PolCompZ) 
		{
			srTGenOptElem::InterpolF_LowOrder(InterpolAux02, xRel, zRel, BufF, 2);
			srTGenOptElem::InterpolFI_LowOrder(InterpolAux02I, xRel, zRel, BufFI, 1);
		}
		else
		{
			srTGenOptElem::InterpolF(InterpolAux02, xRel, zRel, BufF, 2);
			srTGenOptElem::InterpolFI(InterpolAux02I, xRel, zRel, BufFI, 1);
		}

		(*(BufFI+1)) *= (AuxFI+1)->fNorm;
		srTGenOptElem::ImproveReAndIm(BufF+2, BufFI+1);

		//if(FieldShouldBeZeroed)
		//{
		//	*(BufF+2) = 0.; *(BufF+3) = 0.;
		//}

		*pEZ_New = *(BufF+2);
		*(pEZ_New+1) = *(BufF+3);
	}
}

int srTGenOptElem::ReInterpolateWfrSliceSingleE_GPU(srTSRWRadStructAccessData& oldRadSingleE, srTSRWRadStructAccessData& newRadMultiE, int ie, TGPUUsageArg* pGPU)
{
	oldRadSingleE.pBaseRadX = (float*)CAuxGPU::ToDevice(pGPU, oldRadSingleE.pBaseRadX, 2 * oldRadSingleE.ne * oldRadSingleE.nx * oldRadSingleE.nz * sizeof(float));
	oldRadSingleE.pBaseRadZ = (float*)CAuxGPU::ToDevice(pGPU, oldRadSingleE.pBaseRadZ, 2 * oldRadSingleE.ne * oldRadSingleE.nx * oldRadSingleE.nz * sizeof(float));
	newRadMultiE.pBaseRadX = (float*)CAuxGPU::ToDevice(pGPU, newRadMultiE.pBaseRadX, 2 * newRadMultiE.ne * newRadMultiE.nx * newRadMultiE.nz * sizeof(float));
	newRadMultiE.pBaseRadZ = (float*)CAuxGPU::ToDevice(pGPU, newRadMultiE.pBaseRadZ, 2 * newRadMultiE.ne * newRadMultiE.nx * newRadMultiE.nz * sizeof(float));

	CAuxGPU::EnsureDeviceMemoryReady(pGPU, oldRadSingleE.pBaseRadX);
	CAuxGPU::EnsureDeviceMemoryReady(pGPU, oldRadSingleE.pBaseRadZ);
	CAuxGPU::EnsureDeviceMemoryReady(pGPU, newRadMultiE.pBaseRadX);
	CAuxGPU::EnsureDeviceMemoryReady(pGPU, newRadMultiE.pBaseRadZ);

	const int bs = 32;
	dim3 blocks(newRadMultiE.nx / bs + ((newRadMultiE.nx & (bs - 1)) != 0), newRadMultiE.nz, 1);
	dim3 threads(bs, 1);

	srTSRWRadStructAccessData* pOldRadSingleE_dev = NULL; //HG27072024
	pOldRadSingleE_dev = (srTSRWRadStructAccessData*)CAuxGPU::ToDevice(pGPU, &oldRadSingleE, sizeof(srTSRWRadStructAccessData));
	CAuxGPU::EnsureDeviceMemoryReady(pGPU, pOldRadSingleE_dev);

	srTSRWRadStructAccessData* pNewRadMultiE_dev = NULL; //HG27072024
	pNewRadMultiE_dev = (srTSRWRadStructAccessData*)CAuxGPU::ToDevice(pGPU, &newRadMultiE, sizeof(srTSRWRadStructAccessData));
	CAuxGPU::EnsureDeviceMemoryReady(pGPU, pNewRadMultiE_dev);

	ReInterpolateWfrSliceSingleE_Kernel <<<blocks, threads >>> (pOldRadSingleE_dev, pNewRadMultiE_dev, ie);

	CAuxGPU::ToHostAndFree(pGPU, pNewRadMultiE_dev, sizeof(srTSRWRadStructAccessData), true);
	CAuxGPU::ToHostAndFree(pGPU, pOldRadSingleE_dev, sizeof(srTSRWRadStructAccessData), true);

	oldRadSingleE.pBaseRadX = (float*)CAuxGPU::ToHostAndFree(pGPU, oldRadSingleE.pBaseRadX, 2 * oldRadSingleE.ne * oldRadSingleE.nx * oldRadSingleE.nz * sizeof(float), true);
	oldRadSingleE.pBaseRadZ = (float*)CAuxGPU::ToHostAndFree(pGPU, oldRadSingleE.pBaseRadZ, 2 * oldRadSingleE.ne * oldRadSingleE.nx * oldRadSingleE.nz * sizeof(float), true);
	
	CAuxGPU::MarkUpdated(pGPU, newRadMultiE.pBaseRadX, true, false);
	CAuxGPU::MarkUpdated(pGPU, newRadMultiE.pBaseRadZ, true, false);
	newRadMultiE.pBaseRadX = (float*)CAuxGPU::GetHostPtr(pGPU, newRadMultiE.pBaseRadX);
	newRadMultiE.pBaseRadZ = (float*)CAuxGPU::GetHostPtr(pGPU, newRadMultiE.pBaseRadZ);
	
	return 0;
}

__global__ void SetupRadSliceConstE_Kernel(srTSRWRadStructAccessData* pRadAccessData, long ie, float* pInEx, float* pInEz)
{
	int ix = (blockIdx.x * blockDim.x + threadIdx.x); //nx range
	int iz = (blockIdx.y * blockDim.y + threadIdx.y); //nz range

	if(ix >= pRadAccessData->nx) return;
	if(iz >= pRadAccessData->nz) return;

	float *pEx0 = pRadAccessData->pBaseRadX + iz * pRadAccessData->nx * pRadAccessData->ne * 2 + ix * pRadAccessData->ne * 2 + ie * 2;
	float *pEz0 = pRadAccessData->pBaseRadZ + iz * pRadAccessData->nx * pRadAccessData->ne * 2 + ix * pRadAccessData->ne * 2 + ie * 2;
	float *tInEx = pInEx + iz * pRadAccessData->nx * 2 + ix * 2;
	float *tInEz = pInEz + iz * pRadAccessData->nx * 2 + ix * 2;
	
	*pEx0 = *tInEx;
	*(pEx0+1) = *(tInEx+1);
	*pEz0 = *tInEz;
	*(pEz0+1) = *(tInEz+1);
}

int srTGenOptElem::SetupRadSliceConstE_GPU(srTSRWRadStructAccessData* pRadAccessData, long ie, float* pInEx, float* pInEz, TGPUUsageArg* pGPU)
{
	pRadAccessData->pBaseRadX = (float*)CAuxGPU::ToDevice(pGPU, pRadAccessData->pBaseRadX, 2 * pRadAccessData->ne * pRadAccessData->nx * pRadAccessData->nz * sizeof(float));
	pRadAccessData->pBaseRadZ = (float*)CAuxGPU::ToDevice(pGPU, pRadAccessData->pBaseRadZ, 2 * pRadAccessData->ne * pRadAccessData->nx * pRadAccessData->nz * sizeof(float));
	pInEx = (float*)CAuxGPU::ToDevice(pGPU, pInEx, 2*pRadAccessData->nx * pRadAccessData->nz * sizeof(float));
	pInEz = (float*)CAuxGPU::ToDevice(pGPU, pInEz, 2*pRadAccessData->nx * pRadAccessData->nz * sizeof(float));
	
	CAuxGPU::EnsureDeviceMemoryReady(pGPU, pRadAccessData->pBaseRadX);
	CAuxGPU::EnsureDeviceMemoryReady(pGPU, pRadAccessData->pBaseRadZ);
	CAuxGPU::EnsureDeviceMemoryReady(pGPU, pInEx);
	CAuxGPU::EnsureDeviceMemoryReady(pGPU, pInEz);
	
	srTSRWRadStructAccessData* pRadAccessData_dev = (srTSRWRadStructAccessData*)CAuxGPU::ToDevice(pGPU, pRadAccessData, sizeof(srTSRWRadStructAccessData));
	CAuxGPU::EnsureDeviceMemoryReady(pGPU, pRadAccessData_dev);

	const int bs = 256;
	dim3 blocks(pRadAccessData->nx / bs + ((pRadAccessData->nx & (bs - 1)) != 0), pRadAccessData->nz, 1);
	dim3 threads(bs, 1);

	SetupRadSliceConstE_Kernel <<<blocks, threads >>> (pRadAccessData_dev, ie, pInEx, pInEz);

	CAuxGPU::ToHostAndFree(pGPU, pRadAccessData_dev, sizeof(srTSRWRadStructAccessData), true);
	
	CAuxGPU::MarkUpdated(pGPU, pRadAccessData->pBaseRadX, true, false);
	CAuxGPU::MarkUpdated(pGPU, pRadAccessData->pBaseRadZ, true, false);
	pRadAccessData->pBaseRadX = (float*)CAuxGPU::GetHostPtr(pGPU, pRadAccessData->pBaseRadX);
	pRadAccessData->pBaseRadZ = (float*)CAuxGPU::GetHostPtr(pGPU, pRadAccessData->pBaseRadZ);
	
	CAuxGPU::ToHostAndFree(pGPU, pInEx, 2*pRadAccessData->nx * pRadAccessData->nz * sizeof(float), true);
	CAuxGPU::ToHostAndFree(pGPU, pInEz, 2*pRadAccessData->nx * pRadAccessData->nz * sizeof(float), true);

	return 0;
}

template<bool IsCoordRepres, bool IsFreqRepres>
__global__ void ComputeRadMoments_Kernel(srTSRWRadStructAccessData* pSRWRadStructAccessData, int* IndLims, double* SumsZ, int ie)
{
	int ix = (blockIdx.x * blockDim.x + threadIdx.x); //nx range
	int iz = (blockIdx.y * blockDim.y + threadIdx.y); //nz range

	if (ix >= pSRWRadStructAccessData->nx) return;
	if (iz >= pSRWRadStructAccessData->nz) return;

	const double TwoPi = 3.141592653590*2.;
	const double FourPi = TwoPi*2.;
	const double Inv_eV_In_m = 1.239842E-06;

	float *fpX0 = pSRWRadStructAccessData->pBaseRadX;
	float *fpZ0 = pSRWRadStructAccessData->pBaseRadZ;
	bool ExIsOK = fpX0 != 0; //13112011
	bool EzIsOK = fpZ0 != 0;

	long long PerX = pSRWRadStructAccessData->ne << 1;
	long long PerZ = PerX*pSRWRadStructAccessData->nx;

	int nx_mi_1 = pSRWRadStructAccessData->nx - 1;
	int nz_mi_1 = pSRWRadStructAccessData->nz - 1;

	double ePh = pSRWRadStructAccessData->eStart + pSRWRadStructAccessData->eStep*ie; //This assumes wavefront in Time domain; Photon Energy in eV !

	if(!IsFreqRepres)
	{
		ePh = pSRWRadStructAccessData->avgPhotEn; //?? OC041108
	}

	long long Two_ie = ie << 1;
	
	double Lamb_d_FourPi = Inv_eV_In_m/(FourPi*ePh);
	double Lamb_m = Lamb_d_FourPi*FourPi;

	double FourPi_d_Lamb = 1./Lamb_d_FourPi;
	
	double LocRobsX = pSRWRadStructAccessData->RobsX; //OC030409
	if(LocRobsX == 0.) LocRobsX = 100.*Lamb_m;
	double LocRobsZ = pSRWRadStructAccessData->RobsZ;
	if(LocRobsZ == 0.) LocRobsZ = 100.*Lamb_m;

	double FourPi_d_Lamb_d_Rx = FourPi_d_Lamb/LocRobsX;
	double FourPi_d_Lamb_d_Rz = FourPi_d_Lamb/LocRobsZ;

	double FourPi_d_Lamb_d_Rx_xStep = pSRWRadStructAccessData->xStep*FourPi_d_Lamb_d_Rx;
	double FourPi_d_Lamb_d_Rz_zStep = pSRWRadStructAccessData->zStep*FourPi_d_Lamb_d_Rz;
	double TwoPi_d_Lamb_d_Rx_xStep = 0.5*FourPi_d_Lamb_d_Rx_xStep;
	double TwoPi_d_Lamb_d_Rz_zStep = 0.5*FourPi_d_Lamb_d_Rz_zStep;

	double TwoPi_d_Lamb_d_Rx_xStepE2 = TwoPi_d_Lamb_d_Rx_xStep*TwoPi_d_Lamb_d_Rx_xStep;
	double TwoPi_d_Lamb_d_Rz_zStepE2 = TwoPi_d_Lamb_d_Rz_zStep*TwoPi_d_Lamb_d_Rz_zStep;

	double ff[22];
	bool vertCoordInsidePowLim = ((iz >= IndLims[2]) && (iz <= IndLims[3]));
	//long izPerZ = iz*PerZ;
	long long izPerZ = iz*PerZ;
	float *fpX_StartForX = fpX0 + izPerZ;
	float *fpZ_StartForX = fpZ0 + izPerZ;

	double z = pSRWRadStructAccessData->zStart + iz*pSRWRadStructAccessData->zStep;
	bool horCoordInsidePowLim = ((ix >= IndLims[0]) && (ix <= IndLims[1]));
	bool coordInsidePowLim = vertCoordInsidePowLim && horCoordInsidePowLim;

	//long ixPerX_p_Two_ie = ix*PerX + Two_ie;
	long long ixPerX_p_Two_ie = ix*PerX + Two_ie;
	float *fpX = fpX_StartForX + ixPerX_p_Two_ie;
	float *fpZ = fpZ_StartForX + ixPerX_p_Two_ie;

	double ExRe = 0., ExIm = 0., EzRe = 0., EzIm = 0.;
	if(ExIsOK)
	{
		ExRe = *fpX;
		ExIm = *(fpX+1);
	}
	if(EzIsOK)
	{
		EzRe = *fpZ;
		EzIm = *(fpZ+1);
	}

	double x = pSRWRadStructAccessData->xStart + ix*pSRWRadStructAccessData->xStep;
	ff[0] = ExRe*ExRe + ExIm*ExIm; // NormX
	ff[11] = EzRe*EzRe + EzIm*EzIm; // NormZ

	ff[1] = x*ff[0]; // <x>
	ff[3] = z*ff[0]; // <z>
	ff[12] = x*ff[11]; // <x>
	ff[14] = z*ff[11]; // <z>

	if(coordInsidePowLim) //OC13112010
	{
		ff[5] = x*ff[1]; // <xx>
		ff[8] = z*ff[3]; // <zz>
		ff[16] = x*ff[12]; // <xx>
		ff[19] = z*ff[14]; // <zz>
	}
	else
	{
		ff[5] = 0.; // <xx>
		ff[8] = 0.; // <zz>
		ff[16] = 0.; // <xx>
		ff[19] = 0.; // <zz>
	}

	if(IsCoordRepres && (ix > 0))
	{
		float *fpX_Prev = fpX - PerX;
		float *fpZ_Prev = fpZ - PerX;

		double ExReM = 0., ExImM = 0., EzReM = 0., EzImM = 0.;
		if(ExIsOK)
		{
			ExReM = *fpX_Prev; ExImM = *(fpX_Prev+1);
		}
		if(EzIsOK)
		{
			EzReM = *fpZ_Prev; EzImM = *(fpZ_Prev+1);
		}

		double ExReP_mi_ExReM = ExRe - ExReM;
		double ExImP_mi_ExImM = ExIm - ExImM;
		double EzReP_mi_EzReM = EzRe - EzReM;
		double EzImP_mi_EzImM = EzIm - EzImM;

		double ExImP_mi_ExImM_ExRe_mi_ExReP_mi_ExReM_ExIm = ExImP_mi_ExImM*ExRe - ExReP_mi_ExReM*ExIm;
		ff[2] = ExImP_mi_ExImM_ExRe_mi_ExReP_mi_ExReM_ExIm + TwoPi_d_Lamb_d_Rx_xStep*x*ff[0]; // <x'>

		double EzImP_mi_EzImM_EzRe_mi_EzReP_mi_EzReM_EzIm = EzImP_mi_EzImM*EzRe - EzReP_mi_EzReM*EzIm;
		ff[13] = EzImP_mi_EzImM_EzRe_mi_EzReP_mi_EzReM_EzIm + TwoPi_d_Lamb_d_Rx_xStep*x*ff[11]; // <x'>

		if(coordInsidePowLim) //OC13112010
		{
			ff[6] = x*ff[2]; // <xx'>
			ff[7] = (ExReP_mi_ExReM*ExReP_mi_ExReM + ExImP_mi_ExImM*ExImP_mi_ExImM) 
					+ ExImP_mi_ExImM_ExRe_mi_ExReP_mi_ExReM_ExIm*TwoPi_d_Lamb_d_Rx_xStep*x
					+ TwoPi_d_Lamb_d_Rx_xStepE2*x*x*ff[0]; // <x'x'>
			ff[17] = x*ff[13]; // <xx'>
			ff[18] = EzReP_mi_EzReM*EzReP_mi_EzReM + EzImP_mi_EzImM*EzImP_mi_EzImM
					+ EzImP_mi_EzImM_EzRe_mi_EzReP_mi_EzReM_EzIm*TwoPi_d_Lamb_d_Rx_xStep*x
					+ TwoPi_d_Lamb_d_Rx_xStepE2*x*x*ff[11]; // <x'x'>
		}
		else
		{
			ff[6] = 0.; // <xx'>
			ff[7] = 0.; // <x'x'>
			ff[17] = 0.; // <xx'>
			ff[18] = 0.; // <x'x'>
		}
	}
	else
	{
		ff[2] = 0.; // <x'>
		ff[6] = 0.; // <xx'>
		ff[7] = 0.; // <x'x'>
		ff[13] = 0.; // <x'>
		ff[17] = 0.; // <xx'>
		ff[18] = 0.; // <x'x'>
	}

	if(IsCoordRepres && (iz > 0))
	{
		float *fpX_Prev = fpX - PerZ;
		float *fpZ_Prev = fpZ - PerZ;

		double ExReM = 0., ExImM = 0, EzReM = 0., EzImM = 0.;
		if(ExIsOK)
		{
			ExReM = *fpX_Prev; ExImM = *(fpX_Prev+1);
		}
		if(EzIsOK)
		{
			EzReM = *fpZ_Prev; EzImM = *(fpZ_Prev+1);
		}

		double ExReP_mi_ExReM = ExRe - ExReM;
		double ExImP_mi_ExImM = ExIm - ExImM;
		double EzReP_mi_EzReM = EzRe - EzReM;
		double EzImP_mi_EzImM = EzIm - EzImM;

		double ExImP_mi_ExImM_ExRe_mi_ExReP_mi_ExReM_ExIm = ExImP_mi_ExImM*ExRe - ExReP_mi_ExReM*ExIm;
		ff[4] = ExImP_mi_ExImM_ExRe_mi_ExReP_mi_ExReM_ExIm + TwoPi_d_Lamb_d_Rz_zStep*z*ff[0]; // <z'>

		double EzImP_mi_EzImM_EzRe_mi_EzReP_mi_EzReM_EzIm = EzImP_mi_EzImM*EzRe - EzReP_mi_EzReM*EzIm;
		ff[15] = EzImP_mi_EzImM_EzRe_mi_EzReP_mi_EzReM_EzIm + TwoPi_d_Lamb_d_Rz_zStep*z*ff[11]; // <z'>

		if(coordInsidePowLim) //OC13112010
		{
			ff[9] = z*ff[4]; // <zz'>
			ff[10] = ExReP_mi_ExReM*ExReP_mi_ExReM + ExImP_mi_ExImM*ExImP_mi_ExImM
					+ ExImP_mi_ExImM_ExRe_mi_ExReP_mi_ExReM_ExIm*TwoPi_d_Lamb_d_Rz_zStep*z
					+ TwoPi_d_Lamb_d_Rz_zStepE2*z*z*ff[0]; // <z'z'>
			ff[20] = z*ff[15]; // <zz'>
			ff[21] = EzReP_mi_EzReM*EzReP_mi_EzReM + EzImP_mi_EzImM*EzImP_mi_EzImM
					+ EzImP_mi_EzImM_EzRe_mi_EzReP_mi_EzReM_EzIm*TwoPi_d_Lamb_d_Rz_zStep*z
					+ TwoPi_d_Lamb_d_Rz_zStepE2*z*z*ff[11]; // <z'z'>
		}
		else
		{
			ff[9] = 0.; // <zz'>
			ff[10] = 0.; // <z'z'>
			ff[20] = 0.; // <zz'>
			ff[21] = 0.;
		}
	}
	else
	{
		ff[4] = 0.; // <z'>
		ff[9] = 0.; // <zz'>
		ff[10] = 0.; // <z'z'>
		ff[15] = 0.; // <z'>
		ff[20] = 0.; // <zz'>
		ff[21] = 0.; // <z'z'>
	}

	if((ix == 0) || (ix == nx_mi_1)) for(int k=0; k<22; k++) ff[k] *= 0.5;
	if(ix == 1)
	{
		ff[2] *= 0.5; // <x'>
		ff[6] *= 0.5; // <xx'>
		ff[7] *= 0.5; // <x'x'>
		ff[13] *= 0.5; // <x'>
		ff[17] *= 0.5; // <xx'>
		ff[18] *= 0.5; // <x'x'>
	}

	cg::coalesced_group g = cg::coalesced_threads();
	for(int kk=0; kk<22; kk++)
		ff[kk] = cg::reduce(g, ff[kk], cg::plus<double>());

	if(g.thread_rank() == 0)
	{
		if((iz == 0) || (iz == nz_mi_1)) for(int k2=0; k2<22; k2++) ff[k2] *= 0.5;
		if(iz == 1)
		{
			ff[4] *= 0.5; // <z'>
			ff[9] *= 0.5; // <zz'>
			ff[10] *= 0.5; // <z'z'>
			ff[15] *= 0.5; // <z'>
			ff[20] *= 0.5; // <zz'>
			ff[21] *= 0.5; // <z'z'>
		}

		//printf("%f\n", ff[0]);
		for (int kk = 0; kk < 22; kk++) atomicAdd(SumsZ + kk, ff[kk]);
	}
}

void srTGenOptElem::ComputeRadMoments_GPU(srTSRWRadStructAccessData* pSRWRadStructAccessData, int ie, double* SumsZ, int* IndLims, TGPUUsageArg* pGPU) //HG26072024
{
	pSRWRadStructAccessData->pBaseRadX = (float*)CAuxGPU::ToDevice(pGPU, pSRWRadStructAccessData->pBaseRadX, 2 * pSRWRadStructAccessData->ne * pSRWRadStructAccessData->nx * pSRWRadStructAccessData->nz * sizeof(float));
	pSRWRadStructAccessData->pBaseRadZ = (float*)CAuxGPU::ToDevice(pGPU, pSRWRadStructAccessData->pBaseRadZ, 2 * pSRWRadStructAccessData->ne * pSRWRadStructAccessData->nx * pSRWRadStructAccessData->nz * sizeof(float));
	SumsZ = (double*)CAuxGPU::ToDevice(pGPU, SumsZ, 22 * sizeof(double), true, false, 2);
	IndLims = (int*)CAuxGPU::ToDevice(pGPU, IndLims, 4 * sizeof(int));
	
	CAuxGPU::EnsureDeviceMemoryReady(pGPU, pSRWRadStructAccessData->pBaseRadX);
	CAuxGPU::EnsureDeviceMemoryReady(pGPU, pSRWRadStructAccessData->pBaseRadZ);
	CAuxGPU::EnsureDeviceMemoryReady(pGPU, SumsZ);
	CAuxGPU::EnsureDeviceMemoryReady(pGPU, IndLims);
	
	srTSRWRadStructAccessData* pSRWRadStructAccessData_dev = (srTSRWRadStructAccessData*)CAuxGPU::ToDevice(pGPU, pSRWRadStructAccessData, sizeof(srTSRWRadStructAccessData));
	CAuxGPU::EnsureDeviceMemoryReady(pGPU, pSRWRadStructAccessData_dev);

	const int bs = 256;
	dim3 blocks((pSRWRadStructAccessData->nx - 1) / bs + 1, pSRWRadStructAccessData->nz, 1);
	dim3 threads(bs, 1);

	bool IsCoordRepres = (pSRWRadStructAccessData->Pres == 0);
	bool IsFreqRepres = (pSRWRadStructAccessData->PresT == 0);

	if (IsCoordRepres && IsFreqRepres) ComputeRadMoments_Kernel<true, true> <<<blocks, threads >>> 			(pSRWRadStructAccessData_dev, IndLims, SumsZ, ie);
	else if (IsCoordRepres && !IsFreqRepres) ComputeRadMoments_Kernel<true, false> <<<blocks, threads >>>   (pSRWRadStructAccessData_dev, IndLims, SumsZ, ie);
	else if (!IsCoordRepres && IsFreqRepres) ComputeRadMoments_Kernel<false, true> <<<blocks, threads >>>   (pSRWRadStructAccessData_dev, IndLims, SumsZ, ie);
	else if (!IsCoordRepres && !IsFreqRepres) ComputeRadMoments_Kernel<false, false> <<<blocks, threads >>> (pSRWRadStructAccessData_dev, IndLims, SumsZ, ie);

	CAuxGPU::ToHostAndFree(pGPU, pSRWRadStructAccessData_dev, sizeof(srTSRWRadStructAccessData), true);
	
	pSRWRadStructAccessData->pBaseRadX = (float*)CAuxGPU::GetHostPtr(pGPU, pSRWRadStructAccessData->pBaseRadX);
	pSRWRadStructAccessData->pBaseRadZ = (float*)CAuxGPU::GetHostPtr(pGPU, pSRWRadStructAccessData->pBaseRadZ);
	
	CAuxGPU::MarkUpdated(pGPU, SumsZ, true, false);

	CAuxGPU::ToHostAndFree(pGPU, IndLims, 4 * sizeof(int), true);
	CAuxGPU::ToHostAndFree(pGPU, SumsZ, 22 * sizeof(double));
}
#endif