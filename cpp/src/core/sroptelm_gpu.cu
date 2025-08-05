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
    //int ix = (blockIdx.x * blockDim.x + threadIdx.x); //nx range
    //int iz = (blockIdx.y * blockDim.y + threadIdx.y); //nz range
    //int ie = (blockIdx.z * blockDim.z + threadIdx.z) + ieStart; //ne range
	int ie = (blockIdx.x * blockDim.x + threadIdx.x); //ne range //HG24042025
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
		RadAccessData.pBaseRadX = CAuxGPU::ToDevice(pGPU, RadAccessData.pBaseRadX, 2*RadAccessData.ne*RadAccessData.nx*RadAccessData.nz);
		CAuxGPU::EnsureDeviceMemoryReady(pGPU, RadAccessData.pBaseRadX);
	}
	if (RadAccessData.pBaseRadZ != NULL)
	{
		RadAccessData.pBaseRadZ = CAuxGPU::ToDevice(pGPU, RadAccessData.pBaseRadZ, 2*RadAccessData.ne*RadAccessData.nx*RadAccessData.nz);
		CAuxGPU::EnsureDeviceMemoryReady(pGPU, RadAccessData.pBaseRadZ);
	}

	srTSRWRadStructAccessData* pRadAccessData_dev = CAuxGPU::ToDevice(pGPU, &RadAccessData, 1); //HG27072024
	CAuxGPU::EnsureDeviceMemoryReady(pGPU, pRadAccessData_dev);

	int minGridSize;
    int bs = 256;
	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &bs, TreatStronglyOscillatingTerm_Kernel, 0, (ieBefEnd - ieStart) * RadAccessData.nx * RadAccessData.nz);
	dim3 blocks(ieBefEnd - ieStart, RadAccessData.nx, RadAccessData.nz);
	dim3 threads(1, 1, 1);
	CAuxGPU::CalcBlockSizeAndGridSize(bs, blocks, threads);

    //TreatStronglyOscillatingTerm_Kernel<< <blocks, threads >> > (RadAccessData, TreatPolCompX, TreatPolCompZ, ConstRx, ConstRz, ieStart);
    TreatStronglyOscillatingTerm_Kernel<< <blocks, threads >> > (pRadAccessData_dev, TreatPolCompX, TreatPolCompZ, ConstRx, ConstRz, ieStart, ieBefEnd); //HG27072024

	CAuxGPU::ToHostAndFree(pGPU, pRadAccessData_dev); //HG27072024
	
	CAuxGPU::MarkUpdated(pGPU, RadAccessData.pBaseRadX, CAuxGPU::DEVICE);
	CAuxGPU::MarkUpdated(pGPU, RadAccessData.pBaseRadZ, CAuxGPU::DEVICE);

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

//__global__ void MakeWfrEdgeCorrection_Kernel(srTSRWRadStructAccessData RadAccessData, float* pDataEx, float* pDataEz, srTDataPtrsForWfrEdgeCorr DataPtrs, float dxSt, float dxFi, float dzSt, float dzFi)
__global__ void MakeWfrEdgeCorrection_Kernel(srTSRWRadStructAccessData* pRadAccessData, float* __restrict__ pDataEx, float* __restrict__ pDataEz, srTDataPtrsForWfrEdgeCorr DataPtrs, float dxSt, float dxFi, float dzSt, float dzFi) //HG27072024
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

		//long TwoNz = RadAccessData.nz << 1; //OC25012024 (commented-out)
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
	pDataEx = CAuxGPU::ToDevice(pGPU, pDataEx, 2*RadAccessData->ne*RadAccessData->nx*RadAccessData->nz);
	pDataEz = CAuxGPU::ToDevice(pGPU, pDataEz, 2*RadAccessData->ne*RadAccessData->nx*RadAccessData->nz);
	DataPtrs.FFTArrXStEx = CAuxGPU::ToDevice(pGPU, DataPtrs.FFTArrXStEx, 2*RadAccessData->nz);
	DataPtrs.FFTArrXStEz = CAuxGPU::ToDevice(pGPU, DataPtrs.FFTArrXStEz, 2*RadAccessData->nz);
	DataPtrs.FFTArrXFiEx = CAuxGPU::ToDevice(pGPU, DataPtrs.FFTArrXFiEx, 2*RadAccessData->nz);
	DataPtrs.FFTArrXFiEz = CAuxGPU::ToDevice(pGPU, DataPtrs.FFTArrXFiEz, 2*RadAccessData->nz);
	DataPtrs.FFTArrZStEx = CAuxGPU::ToDevice(pGPU, DataPtrs.FFTArrZStEx, 2*RadAccessData->nx);
	DataPtrs.FFTArrZStEz = CAuxGPU::ToDevice(pGPU, DataPtrs.FFTArrZStEz, 2*RadAccessData->nx);
	DataPtrs.FFTArrZFiEx = CAuxGPU::ToDevice(pGPU, DataPtrs.FFTArrZFiEx, 2*RadAccessData->nx);
	DataPtrs.FFTArrZFiEz = CAuxGPU::ToDevice(pGPU, DataPtrs.FFTArrZFiEz, 2*RadAccessData->nx);
	DataPtrs.ExpArrXSt = CAuxGPU::ToDevice(pGPU, DataPtrs.ExpArrXSt, 2*RadAccessData->nx);
	DataPtrs.ExpArrXFi = CAuxGPU::ToDevice(pGPU, DataPtrs.ExpArrXFi, 2*RadAccessData->nx);
	DataPtrs.ExpArrZSt = CAuxGPU::ToDevice(pGPU, DataPtrs.ExpArrZSt, 2*RadAccessData->nz);
	DataPtrs.ExpArrZFi = CAuxGPU::ToDevice(pGPU, DataPtrs.ExpArrZFi, 2*RadAccessData->nz);

	CAuxGPU::EnsureDeviceMemoryReady(pGPU, 
		pDataEx, 
		pDataEz, 
		DataPtrs.FFTArrXStEx, 
		DataPtrs.FFTArrXStEz, 
		DataPtrs.FFTArrXFiEx, 
		DataPtrs.FFTArrXFiEz, 
		DataPtrs.FFTArrZStEx, 
		DataPtrs.FFTArrZStEz, 
		DataPtrs.FFTArrZFiEx, 
		DataPtrs.FFTArrZFiEz, 
		DataPtrs.ExpArrXSt, 
		DataPtrs.ExpArrXFi, 
		DataPtrs.ExpArrZSt, 
		DataPtrs.ExpArrZFi);

	srTSRWRadStructAccessData* pRadAccessData_dev = CAuxGPU::ToDevice(pGPU, RadAccessData, 1); //HG27072024
	CAuxGPU::EnsureDeviceMemoryReady(pGPU, pRadAccessData_dev);

	//const int bs = 256;
	//dim3 blocks(RadAccessData->nx / bs + ((RadAccessData->nx & (bs - 1)) != 0), RadAccessData->nz);
	//dim3 threads(bs, 1);
	int minGridSize;
	int bs = 256;
	dim3 blocks(RadAccessData->nx, RadAccessData->nz);
	dim3 threads(1, 1);
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &bs, MakeWfrEdgeCorrection_Kernel, 0, RadAccessData->nx);
    blocks.x = (RadAccessData->nx + bs - 1) / bs;
    threads.x = bs;

	//MakeWfrEdgeCorrection_Kernel << <blocks, threads >> > (*RadAccessData, pDataEx, pDataEz, DataPtrs, (float)DataPtrs.dxSt, (float)DataPtrs.dxFi, (float)DataPtrs.dzSt, (float)DataPtrs.dzFi);
	MakeWfrEdgeCorrection_Kernel <<<blocks, threads>>> (pRadAccessData_dev, pDataEx, pDataEz, DataPtrs, (float)DataPtrs.dxSt, (float)DataPtrs.dxFi, (float)DataPtrs.dzSt, (float)DataPtrs.dzFi); //HG27072024

	CAuxGPU::ToHostAndFree(pGPU, pRadAccessData_dev); //HG27072024

	DataPtrs.FFTArrXStEx = CAuxGPU::ToHostAndFree(pGPU, DataPtrs.FFTArrXStEx);
	DataPtrs.FFTArrXStEz = CAuxGPU::ToHostAndFree(pGPU, DataPtrs.FFTArrXStEz);
	DataPtrs.FFTArrXFiEx = CAuxGPU::ToHostAndFree(pGPU, DataPtrs.FFTArrXFiEx);
	DataPtrs.FFTArrXFiEz = CAuxGPU::ToHostAndFree(pGPU, DataPtrs.FFTArrXFiEz);
	DataPtrs.FFTArrZStEx = CAuxGPU::ToHostAndFree(pGPU, DataPtrs.FFTArrZStEx);
	DataPtrs.FFTArrZStEz = CAuxGPU::ToHostAndFree(pGPU, DataPtrs.FFTArrZStEz);
	DataPtrs.FFTArrZFiEx = CAuxGPU::ToHostAndFree(pGPU, DataPtrs.FFTArrZFiEx);
	DataPtrs.FFTArrZFiEz = CAuxGPU::ToHostAndFree(pGPU, DataPtrs.FFTArrZFiEz);
	DataPtrs.ExpArrXSt = CAuxGPU::ToHostAndFree(pGPU, DataPtrs.ExpArrXSt);
	DataPtrs.ExpArrXFi = CAuxGPU::ToHostAndFree(pGPU, DataPtrs.ExpArrXFi);
	DataPtrs.ExpArrZSt = CAuxGPU::ToHostAndFree(pGPU, DataPtrs.ExpArrZSt);
	DataPtrs.ExpArrZFi = CAuxGPU::ToHostAndFree(pGPU, DataPtrs.ExpArrZFi);

	CAuxGPU::MarkUpdated(pGPU, pDataEx, CAuxGPU::DEVICE);
	CAuxGPU::MarkUpdated(pGPU, pDataEz, CAuxGPU::DEVICE);

//#ifdef _DEBUG
//	CAuxGPU::ToHostAndFree(pGPU, pDataEx, 2*RadAccessData->ne*RadAccessData->nx*RadAccessData->nz*sizeof(float));
//	CAuxGPU::ToHostAndFree(pGPU, pDataEz, 2*RadAccessData->ne*RadAccessData->nx*RadAccessData->nz*sizeof(float));
//	cudaStreamSynchronize(0);
//	auto err = cudaGetLastError();
//	printf("%s\r\n", cudaGetErrorString(err));
//#endif
}

//template<bool TreatPolCompX, bool TreatPolCompZ> __global__ void RadResizeCore_Kernel(srTSRWRadStructAccessData OldRadAccessData, srTSRWRadStructAccessData NewRadAccessData)
template<bool TreatPolCompX, bool TreatPolCompZ> __global__ void RadResizeCore_Kernel(srTSRWRadStructAccessData* __restrict__ pOldRadAccessData, srTSRWRadStructAccessData* __restrict__ pNewRadAccessData) //HG27072024
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

	float * __restrict__ pEX0_New = 0, * __restrict__ pEZ0_New = 0;
	pEX0_New = pNewRadAccessData->pBaseRadX;
	pEZ0_New = pNewRadAccessData->pBaseRadZ;

	float* __restrict__ pEX0_Old = 0, * __restrict__ pEZ0_Old = 0;
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
	OldRadAccessData.pBaseRadX = CAuxGPU::ToDevice(pGPU, OldRadAccessData.pBaseRadX, 2*OldRadAccessData.ne*OldRadAccessData.nx*OldRadAccessData.nz);
	OldRadAccessData.pBaseRadZ = CAuxGPU::ToDevice(pGPU, OldRadAccessData.pBaseRadZ, 2*OldRadAccessData.ne*OldRadAccessData.nx*OldRadAccessData.nz);
	//NewRadAccessData.pBaseRadX = (float*)CAuxGPU::ToDevice(pGPU, NewRadAccessData.pBaseRadX, 2*NewRadAccessData.ne*NewRadAccessData.nx*NewRadAccessData.nz*sizeof(float), true);
	//NewRadAccessData.pBaseRadZ = (float*)CAuxGPU::ToDevice(pGPU, NewRadAccessData.pBaseRadZ, 2*NewRadAccessData.ne*NewRadAccessData.nx*NewRadAccessData.nz*sizeof(float), true);
	NewRadAccessData.pBaseRadX = CAuxGPU::ToDevice(pGPU, NewRadAccessData.pBaseRadX, 2*NewRadAccessData.ne*NewRadAccessData.nx*NewRadAccessData.nz, CAuxGPU::DONT_COPY);
	NewRadAccessData.pBaseRadZ = CAuxGPU::ToDevice(pGPU, NewRadAccessData.pBaseRadZ, 2*NewRadAccessData.ne*NewRadAccessData.nx*NewRadAccessData.nz, CAuxGPU::DONT_COPY);
	
	CAuxGPU::EnsureDeviceMemoryReady(pGPU, OldRadAccessData.pBaseRadX, OldRadAccessData.pBaseRadZ, NewRadAccessData.pBaseRadX, NewRadAccessData.pBaseRadZ); //HG27072024 Uncommented

	srTSRWRadStructAccessData* pOldRadAccessData_dev = CAuxGPU::ToDevice(pGPU, &OldRadAccessData, 1); //HG27072024
	srTSRWRadStructAccessData* pNewRadAccessData_dev = CAuxGPU::ToDevice(pGPU, &NewRadAccessData, 1); //HG27072024
	CAuxGPU::EnsureDeviceMemoryReady(pGPU, pOldRadAccessData_dev, pNewRadAccessData_dev); //HG27072024

	int minGridSize;
	int bs0, bs1;
	dim3 blocks0(nx, nz, ne);
	dim3 threads0(bs0, 1);
	dim3 blocks1(nx, nz, ne);
	dim3 threads1(bs1, 1);
	
	if (TreatPolCompX) cudaOccupancyMaxPotentialBlockSize(&minGridSize, &bs0, RadResizeCore_Kernel<true, false>, 0, nx);
	if (TreatPolCompZ) cudaOccupancyMaxPotentialBlockSize(&minGridSize, &bs1, RadResizeCore_Kernel<false, true>, 0, nx);
	if (bs0 > 16)
	{
		int bs0_rem = bs0 / 16;
		blocks0.y = (nz + bs0_rem - 1) / bs0_rem;
		threads0.y = bs0_rem;
		bs0 = 16;
	}
    blocks0.x = (nx + bs0 - 1) / bs0;
    threads0.x = bs0;
	if (bs1 > 16)
	{
		int bs1_rem = bs1 / 16;
		blocks1.y = (nz + bs1_rem - 1) / bs1_rem;
		threads1.y = bs1_rem;
		bs1 = 16;
	}
	blocks1.x = (nx + bs1 - 1) / bs1;
	threads1.x = bs1;

	long long stream1 = CAuxGPU::GetComputeStream(pGPU, 0);
	CAuxGPU::SyncComputeStream(pGPU, 0, stream1);

	if (TreatPolCompX) RadResizeCore_Kernel<true, false> << <blocks0, threads0, 0 >> > (pOldRadAccessData_dev, pNewRadAccessData_dev);
	if (TreatPolCompZ) RadResizeCore_Kernel<false, true> << <blocks1, threads1, 0, (cudaStream_t)stream1 >> > (pOldRadAccessData_dev, pNewRadAccessData_dev);

	CAuxGPU::SyncComputeStream(pGPU, stream1, 0);
	CAuxGPU::ToHostAndFree(pGPU, pOldRadAccessData_dev); //HG27072024
	CAuxGPU::ToHostAndFree(pGPU, pNewRadAccessData_dev); //HG27072024

	OldRadAccessData.pBaseRadX = (float*)CAuxGPU::ToHostAndFree(pGPU, OldRadAccessData.pBaseRadX);
	OldRadAccessData.pBaseRadZ = (float*)CAuxGPU::ToHostAndFree(pGPU, OldRadAccessData.pBaseRadZ);
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

template<bool TreatPolCompX, bool TreatPolCompZ> __global__ void RadResizeCore_OnlyLargerRange_Kernel(srTSRWRadStructAccessData* __restrict__ pOldRadAccessData, srTSRWRadStructAccessData* __restrict__ pNewRadAccessData)
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

	long long PerX_New = pNewRadAccessData->ne << 1;
	long long PerZ_New = PerX_New*pNewRadAccessData->nx;

	long long PerX_Old = PerX_New;
	long long PerZ_Old = PerX_Old*pOldRadAccessData->nx;

	double xStepInvOld = 1./pOldRadAccessData->xStep;
	double zStepInvOld = 1./pOldRadAccessData->zStep;
	
	long long Two_ie = ie << 1;
	
	long long izPerZ_New = iz*PerZ_New;
	float* pEX_StartForX_New = pEX0_New + izPerZ_New;
	float* pEZ_StartForX_New = pEZ0_New + izPerZ_New;

	double zAbs = pNewRadAccessData->zStart + iz*pNewRadAccessData->zStep;
	long izOld = long((zAbs - pOldRadAccessData->zStart)*zStepInvOld + 1.E-08);
	long long izPerZ_Old = izOld*PerZ_Old;

	float* pEX_StartForX_Old = pEX0_Old + izPerZ_Old;
	float* pEZ_StartForX_Old = pEZ0_Old + izPerZ_Old;

	long long ixPerX_New_p_Two_ie = ix*PerX_New + Two_ie;
	float* pEX_New = pEX_StartForX_New + ixPerX_New_p_Two_ie;
	float* pEZ_New = pEZ_StartForX_New + ixPerX_New_p_Two_ie;

	double xAbs = pNewRadAccessData->xStart + ix*pNewRadAccessData->xStep;
	long ixOld = long((xAbs - pOldRadAccessData->xStart)*xStepInvOld + 1.E-08);
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

	OldRadAccessData.pBaseRadX = CAuxGPU::ToDevice(pGPU, OldRadAccessData.pBaseRadX, 2 * OldRadAccessData.ne * OldRadAccessData.nx * OldRadAccessData.nz);
	OldRadAccessData.pBaseRadZ = CAuxGPU::ToDevice(pGPU, OldRadAccessData.pBaseRadZ, 2 * OldRadAccessData.ne * OldRadAccessData.nx * OldRadAccessData.nz);
	NewRadAccessData.pBaseRadX = CAuxGPU::ToDevice(pGPU, NewRadAccessData.pBaseRadX, 2 * NewRadAccessData.ne * NewRadAccessData.nx * NewRadAccessData.nz, CAuxGPU::DONT_COPY);
	NewRadAccessData.pBaseRadZ = CAuxGPU::ToDevice(pGPU, NewRadAccessData.pBaseRadZ, 2 * NewRadAccessData.ne * NewRadAccessData.nx * NewRadAccessData.nz, CAuxGPU::DONT_COPY);

	CAuxGPU::EnsureDeviceMemoryReady(pGPU, OldRadAccessData.pBaseRadX, OldRadAccessData.pBaseRadZ, NewRadAccessData.pBaseRadX, NewRadAccessData.pBaseRadZ);

	srTSRWRadStructAccessData* pOldRadAccessData_dev = CAuxGPU::ToDevice(pGPU, &OldRadAccessData, 1);
	srTSRWRadStructAccessData* pNewRadAccessData_dev = CAuxGPU::ToDevice(pGPU, &NewRadAccessData, 1);
	CAuxGPU::EnsureDeviceMemoryReady(pGPU, pOldRadAccessData_dev, pNewRadAccessData_dev);

	int minGridSize;
	int bs = 32;
	dim3 blocks(nx, nz, ne);
	dim3 threads(bs, 1);
	if (TreatPolCompX && TreatPolCompZ) cudaOccupancyMaxPotentialBlockSize(&minGridSize, &bs, RadResizeCore_OnlyLargerRange_Kernel<true, true>, 0, nx);
	else if (TreatPolCompX) cudaOccupancyMaxPotentialBlockSize(&minGridSize, &bs, RadResizeCore_OnlyLargerRange_Kernel<true, false>, 0, nx);
	else if (TreatPolCompZ) cudaOccupancyMaxPotentialBlockSize(&minGridSize, &bs, RadResizeCore_OnlyLargerRange_Kernel<false, true>, 0, nx);
    blocks.x = (nx + bs - 1) / bs;
    threads.x = bs;

	if (TreatPolCompX && TreatPolCompZ) RadResizeCore_OnlyLargerRange_Kernel<true, true> << <blocks, threads >> > (pOldRadAccessData_dev, pNewRadAccessData_dev);
	else if (TreatPolCompX) RadResizeCore_OnlyLargerRange_Kernel<true, false> << <blocks, threads >> > (pOldRadAccessData_dev, pNewRadAccessData_dev);
	else if (TreatPolCompZ) RadResizeCore_OnlyLargerRange_Kernel<false, true> << <blocks, threads >> > (pOldRadAccessData_dev, pNewRadAccessData_dev);

	CAuxGPU::ToHostAndFree(pGPU, pOldRadAccessData_dev);
	CAuxGPU::ToHostAndFree(pGPU, pNewRadAccessData_dev);

	OldRadAccessData.pBaseRadZ = (float*)CAuxGPU::ToHostAndFree(pGPU, OldRadAccessData.pBaseRadZ);
	OldRadAccessData.pBaseRadX = (float*)CAuxGPU::ToHostAndFree(pGPU, OldRadAccessData.pBaseRadX);
	
	CAuxGPU::MarkUpdated(pGPU, NewRadAccessData.pBaseRadX, CAuxGPU::DEVICE);
	CAuxGPU::MarkUpdated(pGPU, NewRadAccessData.pBaseRadZ, CAuxGPU::DEVICE);
	
	NewRadAccessData.pBaseRadX = (float*)CAuxGPU::GetHostPtr(pGPU, NewRadAccessData.pBaseRadX);
	NewRadAccessData.pBaseRadZ = (float*)CAuxGPU::GetHostPtr(pGPU, NewRadAccessData.pBaseRadZ);
	
	return 0;
}

template<bool TreatPolCompX, bool TreatPolCompZ> __global__ void RadResizeCore_OnlyLargerRangeE_Kernel(srTSRWRadStructAccessData* __restrict__ pOldRadAccessData, srTSRWRadStructAccessData* __restrict__ pNewRadAccessData)
{
	int ieStart = int(pNewRadAccessData->AuxLong1);
	
	int ix = (blockIdx.x * blockDim.x + threadIdx.x); //nx range
	int iz = (blockIdx.y * blockDim.y + threadIdx.y); //nz range
	int ie = (blockIdx.z * blockDim.z + threadIdx.z) + ieStart; //ne range

	if (ix > pNewRadAccessData->nx) return;
	if (iz > pNewRadAccessData->nz) return;
	
	float* __restrict__ pEX0_New = pNewRadAccessData->pBaseRadX;
	float* __restrict__ pEZ0_New = pNewRadAccessData->pBaseRadZ;

	float* __restrict__ pEX0_Old = pOldRadAccessData->pBaseRadX;
	float* __restrict__ pEZ0_Old = pOldRadAccessData->pBaseRadZ;

	long long PerX_New = pNewRadAccessData->ne << 1;
	long long PerZ_New = PerX_New * pNewRadAccessData->nx;

	long long PerX_Old = pOldRadAccessData->ne << 1;
	long long PerZ_Old = PerX_Old * pOldRadAccessData->nx;

	double eStepInvOld = 1. / pOldRadAccessData->eStep;

	long long iz_PerZ_New = iz * PerZ_New;
	long long iz_PerZ_Old = iz * PerZ_Old;

	long long iz_PerZ_New_p_ix_PerX_New = iz_PerZ_New + ix * PerX_New;
	long long iz_PerZ_Old_p_ix_PerX_Old = iz_PerZ_Old + ix * PerX_Old;

	long long ofstNew = iz_PerZ_New_p_ix_PerX_New + (ie << 1);
	float* pEX_New = pEX0_New + ofstNew;
	float* pEZ_New = pEZ0_New + ofstNew;

	double eAbs = pNewRadAccessData->eStart + ie * pNewRadAccessData->eStep;
	long ieOld = long((eAbs - pOldRadAccessData->eStart) * eStepInvOld + 1.E-08);

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

	OldRadAccessData.pBaseRadX = CAuxGPU::ToDevice(pGPU, OldRadAccessData.pBaseRadX, 2 * OldRadAccessData.ne * OldRadAccessData.nx * OldRadAccessData.nz);
	OldRadAccessData.pBaseRadZ = CAuxGPU::ToDevice(pGPU, OldRadAccessData.pBaseRadZ, 2 * OldRadAccessData.ne * OldRadAccessData.nx * OldRadAccessData.nz);
	NewRadAccessData.pBaseRadX = CAuxGPU::ToDevice(pGPU, NewRadAccessData.pBaseRadX, 2 * NewRadAccessData.ne * NewRadAccessData.nx * NewRadAccessData.nz, CAuxGPU::DONT_COPY);
	NewRadAccessData.pBaseRadZ = CAuxGPU::ToDevice(pGPU, NewRadAccessData.pBaseRadZ, 2 * NewRadAccessData.ne * NewRadAccessData.nx * NewRadAccessData.nz, CAuxGPU::DONT_COPY);

	CAuxGPU::EnsureDeviceMemoryReady(pGPU, OldRadAccessData.pBaseRadX, OldRadAccessData.pBaseRadZ, NewRadAccessData.pBaseRadX, NewRadAccessData.pBaseRadZ);

	srTSRWRadStructAccessData* pOldRadAccessData_dev = CAuxGPU::ToDevice(pGPU, &OldRadAccessData, 1); //HG27072024
	srTSRWRadStructAccessData* pNewRadAccessData_dev = CAuxGPU::ToDevice(pGPU, &NewRadAccessData, 1); //HG27072024
	CAuxGPU::EnsureDeviceMemoryReady(pGPU, pOldRadAccessData_dev, pNewRadAccessData_dev);

	int minGridSize;
	int bs = 32;
	dim3 blocks(NewRadAccessData.nx, NewRadAccessData.nz, ne);
	dim3 threads(bs, 1);
	if (TreatPolCompX && TreatPolCompZ) cudaOccupancyMaxPotentialBlockSize(&minGridSize, &bs, RadResizeCore_OnlyLargerRangeE_Kernel<true, true>, 0, NewRadAccessData.nx);
	else if (TreatPolCompX) cudaOccupancyMaxPotentialBlockSize(&minGridSize, &bs, RadResizeCore_OnlyLargerRangeE_Kernel<true, false>, 0, NewRadAccessData.nx);
	else if (TreatPolCompZ) cudaOccupancyMaxPotentialBlockSize(&minGridSize, &bs, RadResizeCore_OnlyLargerRangeE_Kernel<false, true>, 0, NewRadAccessData.nx);
    blocks.x = (NewRadAccessData.nx + bs - 1) / bs;
    threads.x = bs;

	if (TreatPolCompX && TreatPolCompZ) RadResizeCore_OnlyLargerRangeE_Kernel<true, true> << <blocks, threads >> > (pOldRadAccessData_dev, pNewRadAccessData_dev);
	else if (TreatPolCompX) RadResizeCore_OnlyLargerRangeE_Kernel<true, false> << <blocks, threads >> > (pOldRadAccessData_dev, pNewRadAccessData_dev);
	else if (TreatPolCompZ) RadResizeCore_OnlyLargerRangeE_Kernel<false, true> << <blocks, threads >> > (pOldRadAccessData_dev, pNewRadAccessData_dev);

	CAuxGPU::ToHostAndFree(pGPU, pOldRadAccessData_dev); //HG27072024
	CAuxGPU::ToHostAndFree(pGPU, pNewRadAccessData_dev); //HG27072024

	OldRadAccessData.pBaseRadX = CAuxGPU::ToHostAndFree(pGPU, OldRadAccessData.pBaseRadX);
	OldRadAccessData.pBaseRadZ = CAuxGPU::ToHostAndFree(pGPU, OldRadAccessData.pBaseRadZ);
	CAuxGPU::MarkUpdated(pGPU, NewRadAccessData.pBaseRadX, CAuxGPU::DEVICE);
	CAuxGPU::MarkUpdated(pGPU, NewRadAccessData.pBaseRadZ, CAuxGPU::DEVICE);
	NewRadAccessData.pBaseRadX = (float*)CAuxGPU::GetHostPtr(pGPU, NewRadAccessData.pBaseRadX);
	NewRadAccessData.pBaseRadZ = (float*)CAuxGPU::GetHostPtr(pGPU, NewRadAccessData.pBaseRadZ);

	return 0;
}

#endif