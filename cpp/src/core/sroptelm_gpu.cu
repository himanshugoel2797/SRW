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
	int ie = (blockIdx.x * blockDim.x + threadIdx.x) + ieStart; //ne range //HG24042025
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
		srTGenOptElem::CosAndSinPi(Phase, CosPh, SinPh);

        long long PerX = pRadAccessData->ne << 1;
        long long PerZ = PerX * pRadAccessData->nx;
        long long offset = ie * 2 + iz * PerZ + ix * PerX;
        
		if (TreatPolCompX)
		{
			float* pExRe = pRadAccessData->pBaseRadX + offset;
			float* pExIm = pExRe + 1;
			//double ExReNew = (*pExRe) * CosPh - (*pExIm) * SinPh;
			//double ExImNew = (*pExRe) * SinPh + (*pExIm) * CosPh;
			
					double tmp0 = (*pExIm)*SinPh;
					double ExReNew = fma(*pExRe, CosPh, -tmp0) + fma(-SinPh, *pExIm, tmp0); // To improve accuracy

					double tmp1 = (*pExIm)*CosPh;
					double ExImNew = fma(*pExRe, SinPh, -tmp1) + fma(CosPh, *pExIm, tmp1); // To improve accuracy
			
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

	dim3 blocks(ieBefEnd - ieStart, RadAccessData.nx, RadAccessData.nz);
	dim3 threads(1, 1, 1);
	CAuxGPU::CalcLaunchDims(TreatStronglyOscillatingTerm_Kernel, blocks, blocks, threads);//, 0, (ieBefEnd - ieStart) * RadAccessData.nx * RadAccessData.nz);

    //TreatStronglyOscillatingTerm_Kernel<< <blocks, threads >> > (RadAccessData, TreatPolCompX, TreatPolCompZ, ConstRx, ConstRz, ieStart);
    TreatStronglyOscillatingTerm_Kernel<<<blocks, threads>>> (pRadAccessData_dev, TreatPolCompX, TreatPolCompZ, ConstRx, ConstRz, ieStart, ieBefEnd); //HG27072024

	CAuxGPU::ToHostAndFree(pGPU, pRadAccessData_dev); //HG27072024
	
	CAuxGPU::MarkUpdated(pGPU, RadAccessData.pBaseRadX, CAuxGPU::DEVICE);
	CAuxGPU::MarkUpdated(pGPU, RadAccessData.pBaseRadZ, CAuxGPU::DEVICE);

//#ifndef _DEBUG
	if (RadAccessData.pBaseRadX != NULL)
		RadAccessData.pBaseRadX = CAuxGPU::GetHostPtr(pGPU, RadAccessData.pBaseRadX);
	if (RadAccessData.pBaseRadZ != NULL)
		RadAccessData.pBaseRadZ = CAuxGPU::GetHostPtr(pGPU, RadAccessData.pBaseRadZ);
//#endif

//#ifdef _DEBUG
//	if (RadAccessData.pBaseRadX != NULL)
//		RadAccessData.pBaseRadX = (float*)CAuxGPU::ToHostAndFree(pGPU, RadAccessData.pBaseRadX, 2*RadAccessData.ne*RadAccessData.nx*RadAccessData.nz*sizeof(float));
//	if (RadAccessData.pBaseRadZ != NULL)
//		RadAccessData.pBaseRadZ = (float*)CAuxGPU::ToHostAndFree(pGPU, RadAccessData.pBaseRadZ, 2*RadAccessData.ne*RadAccessData.nx*RadAccessData.nz*sizeof(float));
//	cudaStreamSynchronize(0);
//	auto err = cudaGetLastError();
//	printf("\r\n%s\r\n", cudaGetErrorString(err));
//#endif
}

//__global__ void MakeWfrEdgeCorrection_Kernel(srTSRWRadStructAccessData RadAccessData, float* pDataEx, float* pDataEz, srTDataPtrsForWfrEdgeCorr DataPtrs, float dxSt, float dxFi, float dzSt, float dzFi)
__global__ void MakeWfrEdgeCorrection_Kernel(srTSRWRadStructAccessData* pRadAccessData, float* __restrict__ pDataEx, float* __restrict__ pDataEz, srTDataPtrsForWfrEdgeCorr DataPtrs) //HG27072024
{
    int ix = (blockIdx.x * blockDim.x + threadIdx.x); //nx range
    int iz = (blockIdx.y * blockDim.y + threadIdx.y); //nz range

    if (ix < pRadAccessData->nx && iz < pRadAccessData->nz)
    {
		double dxSt = DataPtrs.dxSt;
		double dxFi = DataPtrs.dxFi;
		double dzSt = DataPtrs.dzSt;
		double dzFi = DataPtrs.dzFi;
		double dxSt_dzSt = dxSt * dzSt;
		double dxSt_dzFi = dxSt * dzFi;
		double dxFi_dzSt = dxFi * dzSt;
		double dxFi_dzFi = dxFi * dzFi;

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
	printf("\r\n%s\r\n", __func__);
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
	dim3 blocks(RadAccessData->nx, RadAccessData->nz);
	dim3 threads(1);
	CAuxGPU::CalcLaunchDims(MakeWfrEdgeCorrection_Kernel, blocks, blocks, threads);
    
	//MakeWfrEdgeCorrection_Kernel << <blocks, threads >> > (*RadAccessData, pDataEx, pDataEz, DataPtrs, (float)DataPtrs.dxSt, (float)DataPtrs.dxFi, (float)DataPtrs.dzSt, (float)DataPtrs.dzFi);
	MakeWfrEdgeCorrection_Kernel <<<blocks, threads>>> (pRadAccessData_dev, pDataEx, pDataEz, DataPtrs); //HG27072024

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
//	printf("\r\n%s\r\n", cudaGetErrorString(err));
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
			//if(ix == 541 && iz == 252)
			//{
				//printf("TotOffsetOld=%lld PerX_Old=%lld PerZ_Old=%lld izStOld=%d ixStOld=%d LowOrderInterp=%d\n", TotOffsetOld, PerX_Old, PerZ_Old, izStOld, ixStOld, UseLowOrderInterp_PolCompX);
				//printf("%.10f,%.10f,%.10f,%.10f\r\n", AuxF[0].f00, AuxF[0].f01, AuxF[0].f02, AuxF[0].f03);
				//printf("%.10f,%.10f,%.10f,%.10f\r\n", AuxF[0].f10, AuxF[0].f11, AuxF[0].f12, AuxF[0].f13);
				//printf("%.10f,%.10f,%.10f,%.10f\r\n", AuxF[0].f20, AuxF[0].f21, AuxF[0].f22, AuxF[0].f23);
				//printf("%.10f,%.10f,%.10f,%.10f\r\n", AuxF[0].f30, AuxF[0].f31, AuxF[0].f32, AuxF[0].f33);
				//printf("\r\n");
				//printf("%.10f,%.10f,%.10f,%.10f\r\n", AuxF[1].f00, AuxF[1].f01, AuxF[1].f02, AuxF[1].f03);
				//printf("%.10f,%.10f,%.10f,%.10f\r\n", AuxF[1].f10, AuxF[1].f11, AuxF[1].f12, AuxF[1].f13);
				//printf("%.10f,%.10f,%.10f,%.10f\r\n", AuxF[1].f20, AuxF[1].f21, AuxF[1].f22, AuxF[1].f23);
				//printf("%.10f,%.10f,%.10f,%.10f\r\n", AuxF[1].f30, AuxF[1].f31, AuxF[1].f32, AuxF[1].f33);
				//printf("\r\n");
				//printf("%.10f,%.10f,%.10f,%.10f\r\n", AuxFI[0].f00, AuxFI[0].f01, AuxFI[0].f02, AuxFI[0].f03);
				//printf("%.10f,%.10f,%.10f,%.10f\r\n", AuxFI[0].f10, AuxFI[0].f11, AuxFI[0].f12, AuxFI[0].f13);
				//printf("%.10f,%.10f,%.10f,%.10f\r\n", AuxFI[0].f20, AuxFI[0].f21, AuxFI[0].f22, AuxFI[0].f23);
				//printf("%.10f,%.10f,%.10f,%.10f\r\n", AuxFI[0].f30, AuxFI[0].f31, AuxFI[0].f32, AuxFI[0].f33);
				//printf("\r\n");
				//printf("%.10f,%.10f,%.10f,%.10f\r\n", 0.0, InterpolAux01.cAx1z1, InterpolAux01.cAx2z1, InterpolAux01.cAx3z1);
				//printf("%.10f,%.10f,%.10f,%.10f\r\n", InterpolAux01.cAx0z1, InterpolAux01.cAx1z1, InterpolAux01.cAx2z1, InterpolAux01.cAx3z1);
				//printf("%.10f,%.10f,%.10f,%.10f\r\n", InterpolAux01.cAx0z2, InterpolAux01.cAx1z2, InterpolAux01.cAx2z2, InterpolAux01.cAx3z2);
				//printf("%.10f,%.10f,%.10f,%.10f\r\n", InterpolAux01.cAx0z3, InterpolAux01.cAx1z3, InterpolAux01.cAx2z3, InterpolAux01.cAx3z3);
			//}

			if (UseLowOrderInterp_PolCompX)
			{
				srTGenOptElem::InterpolF_LowOrder(InterpolAux02, xRel, zRel, BufF, 0);
				srTGenOptElem::InterpolFI_LowOrder(InterpolAux02I, xRel, zRel, BufFI, 0);
			}
			else
			{
				for (int i = 0; i < 2; i++)
				{
					srTGenOptElem::SetupInterpolAux02(AuxF + i, &InterpolAux01, InterpolAux02 + i);
				}
				srTGenOptElem::SetupInterpolAux02(AuxFI, &InterpolAux01, InterpolAux02I);
				srTGenOptElem::InterpolF(InterpolAux02, xRel, zRel, BufF, 0);
				srTGenOptElem::InterpolFI(InterpolAux02I, xRel, zRel, BufFI, 0);
			}

			(*BufFI) *= AuxFI->fNorm;
			//srTGenOptElem::ImproveReAndIm(BufF, BufFI);
			//if(ix == 541 && iz == 252)
			//{
				//a0 = -pF->f00 + pF->f13 - pF->f20
				//a1 = -pF->f21 - pF->f01
				//a2 = pF->f02 + pF->f11 + pF->f22
				//a3 = pF->f10
				//a4 = -pF->f12
				//a5 = -pF->f23-pF->f03

				//(2*(a0) + 3*(a1) + 6*(a2) + 4*a3 12*a4 +a5)*pC->cAx2z1
				//(3*(a1 + 2*a2) + 2*(a0 + 2*(a3 + 3*a4)) + a5)*pC->cAx2z1
				//printf("2*(-%f + %f - %f) - 3*(%f + %f) + 6*(%f + %f + %f) + 4*%f - 12*%f - %f - %f) * %f\r\n", AuxF[0].f00, AuxF[0].f13, AuxF[0].f20, AuxF[0].f21, AuxF[0].f01, AuxF[0].f02, AuxF[0].f11, AuxF[0].f22, AuxF[0].f10, AuxF[0].f12, AuxF[0].f23, AuxF[0].f03, InterpolAux01.cAx2z1);
				//printf("2*%f - 3*%f + 6*%f + 4*%f - 12*%f + %f) * %f\r\n", -AuxF[0].f00 +AuxF[0].f13 -AuxF[0].f20, AuxF[0].f21+AuxF[0].f01, AuxF[0].f02+AuxF[0].f11+AuxF[0].f22, AuxF[0].f10, AuxF[0].f12, - AuxF[0].f23 - AuxF[0].f03, InterpolAux01.cAx2z1);
				//printf("%f + %f + %f + %f + %f + %f) * %f\r\n", 2*(-AuxF[0].f00 +AuxF[0].f13 -AuxF[0].f20), -3*(AuxF[0].f21+AuxF[0].f01), 6*(AuxF[0].f02+AuxF[0].f11+AuxF[0].f22), 4*AuxF[0].f10, -12*AuxF[0].f12, - AuxF[0].f23 - AuxF[0].f03, InterpolAux01.cAx2z1);
				//printf("%f * %f\r\n", 2*(-AuxF[0].f00 +AuxF[0].f13 -AuxF[0].f20) -3*(AuxF[0].f21+AuxF[0].f01) + 6*(AuxF[0].f02+AuxF[0].f11+AuxF[0].f22) + 4*AuxF[0].f10 -12*AuxF[0].f12 - AuxF[0].f23 - AuxF[0].f03, InterpolAux01.cAx2z1);
				//printf("%f\r\n", (2*(-AuxF[0].f00 +AuxF[0].f13 -AuxF[0].f20) -3*(AuxF[0].f21+AuxF[0].f01) + 6*(AuxF[0].f02+AuxF[0].f11+AuxF[0].f22) + 4*AuxF[0].f10 -12*AuxF[0].f12 - AuxF[0].f23 - AuxF[0].f03) * InterpolAux01.cAx2z1);
				//printf("\r\n");
				//printf("%.10f,%.10f,%.10f,%.10f\r\n", InterpolAux02[0].Ax0z0, InterpolAux02[0].Ax1z0, InterpolAux02[0].Ax2z0, InterpolAux02[0].Ax3z0);
				//printf("%.10f,%.10f,%.10f,%.10f\r\n", InterpolAux02[0].Ax0z1, InterpolAux02[0].Ax1z1, InterpolAux02[0].Ax2z1, InterpolAux02[0].Ax3z1);
				//printf("%.10f,%.10f,%.10f,%.10f\r\n", InterpolAux02[0].Ax0z2, InterpolAux02[0].Ax1z2, InterpolAux02[0].Ax2z2, InterpolAux02[0].Ax3z2);
				//printf("%.10f,%.10f,%.10f,%.10f\r\n", InterpolAux02[0].Ax0z3, InterpolAux02[0].Ax1z3, InterpolAux02[0].Ax2z3, InterpolAux02[0].Ax3z3);
				//printf("\r\n");
				//printf("%f,%f,%f,%f,%f\r\n", BufF[0], BufF[1], BufFI[0], BufFI[1], AuxFI->fNorm);
			//}

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
			if (UseLowOrderInterp_PolCompZ)
			{
				srTGenOptElem::InterpolF_LowOrder(InterpolAux02, xRel, zRel, BufF, 2);
				srTGenOptElem::InterpolFI_LowOrder(InterpolAux02I, xRel, zRel, BufFI, 1);
			}
			else
			{
				for (int i = 0; i < 2; i++)
				{
					srTGenOptElem::SetupInterpolAux02(AuxF + 2 + i, &InterpolAux01, InterpolAux02 + 2 + i);
				}
				srTGenOptElem::SetupInterpolAux02(AuxFI + 1, &InterpolAux01, InterpolAux02I + 1);
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
	NewRadAccessData.pBaseRadX = CAuxGPU::ToDevice(pGPU, NewRadAccessData.pBaseRadX, 2*NewRadAccessData.ne*NewRadAccessData.nx*NewRadAccessData.nz, CAuxGPU::DONT_COPY);
	NewRadAccessData.pBaseRadZ = CAuxGPU::ToDevice(pGPU, NewRadAccessData.pBaseRadZ, 2*NewRadAccessData.ne*NewRadAccessData.nx*NewRadAccessData.nz, CAuxGPU::DONT_COPY);
	
	CAuxGPU::EnsureDeviceMemoryReady(pGPU, OldRadAccessData.pBaseRadX, OldRadAccessData.pBaseRadZ, NewRadAccessData.pBaseRadX, NewRadAccessData.pBaseRadZ); //HG27072024 Uncommented

	srTSRWRadStructAccessData* pOldRadAccessData_dev = CAuxGPU::ToDevice(pGPU, &OldRadAccessData, 1); //HG27072024
	srTSRWRadStructAccessData* pNewRadAccessData_dev = CAuxGPU::ToDevice(pGPU, &NewRadAccessData, 1); //HG27072024
	CAuxGPU::EnsureDeviceMemoryReady(pGPU, pOldRadAccessData_dev, pNewRadAccessData_dev); //HG27072024

	dim3 blocks0(nx, nz, ne);
	dim3 threads0(1);
	dim3 blocks1(nx, nz, ne);
	dim3 threads1(1);
	if (TreatPolCompX) CAuxGPU::CalcLaunchDims(RadResizeCore_Kernel<true, false>, blocks0, blocks0, threads0, 16);
	if (TreatPolCompZ) CAuxGPU::CalcLaunchDims(RadResizeCore_Kernel<false, true>, blocks1, blocks1, threads1, 16);
	long long stream1 = CAuxGPU::GetComputeStream(pGPU, 0);
	CAuxGPU::SyncComputeStream(pGPU, 0, stream1);

	if (TreatPolCompX) RadResizeCore_Kernel<true, false> <<<blocks0, threads0, 0 >>> (pOldRadAccessData_dev, pNewRadAccessData_dev);
	if (TreatPolCompZ) RadResizeCore_Kernel<false, true> <<<blocks1, threads1, 0, (cudaStream_t)stream1 >>> (pOldRadAccessData_dev, pNewRadAccessData_dev);

	CAuxGPU::SyncComputeStream(pGPU, stream1, 0);
	CAuxGPU::ToHostAndFree(pGPU, pOldRadAccessData_dev); //HG27072024
	CAuxGPU::ToHostAndFree(pGPU, pNewRadAccessData_dev); //HG27072024

	OldRadAccessData.pBaseRadX = CAuxGPU::ToHostAndFree(pGPU, OldRadAccessData.pBaseRadX);
	OldRadAccessData.pBaseRadZ = CAuxGPU::ToHostAndFree(pGPU, OldRadAccessData.pBaseRadZ);
	//NewRadAccessData.pBaseRadX = CAuxGPU::ToHostAndFree(pGPU, NewRadAccessData.pBaseRadX, 2*NewRadAccessData.ne*NewRadAccessData.nx*NewRadAccessData.nz*sizeof(float));
	//NewRadAccessData.pBaseRadZ = CAuxGPU::ToHostAndFree(pGPU, NewRadAccessData.pBaseRadZ, 2*NewRadAccessData.ne*NewRadAccessData.nx*NewRadAccessData.nz*sizeof(float));
	CAuxGPU::MarkUpdated(pGPU, NewRadAccessData.pBaseRadX, CAuxGPU::DEVICE);
	CAuxGPU::MarkUpdated(pGPU, NewRadAccessData.pBaseRadZ, CAuxGPU::DEVICE);
//#ifndef _DEBUG
	NewRadAccessData.pBaseRadX = CAuxGPU::GetHostPtr(pGPU, NewRadAccessData.pBaseRadX);
	NewRadAccessData.pBaseRadZ = CAuxGPU::GetHostPtr(pGPU, NewRadAccessData.pBaseRadZ);
//#endif

//#ifdef _DEBUG
	//NewRadAccessData.pBaseRadX = (float*)CAuxGPU::ToHostAndFree(pGPU, NewRadAccessData.pBaseRadX, 2*NewRadAccessData.ne*NewRadAccessData.nx*NewRadAccessData.nz*sizeof(float), false);
	//NewRadAccessData.pBaseRadZ = (float*)CAuxGPU::ToHostAndFree(pGPU, NewRadAccessData.pBaseRadZ, 2*NewRadAccessData.ne*NewRadAccessData.nx*NewRadAccessData.nz*sizeof(float), false);
	//cudaStreamSynchronize(0);
	//auto err = cudaGetLastError();
	//printf("\r\n%s\r\n", cudaGetErrorString(err));

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

	CAuxGPU::Memset(pGPU, NewRadAccessData.pBaseRadX, 0.0f, 2 * NewRadAccessData.ne * NewRadAccessData.nx * NewRadAccessData.nz);
	CAuxGPU::Memset(pGPU, NewRadAccessData.pBaseRadZ, 0.0f, 2 * NewRadAccessData.ne * NewRadAccessData.nx * NewRadAccessData.nz);

	CAuxGPU::EnsureDeviceMemoryReady(pGPU, OldRadAccessData.pBaseRadX, OldRadAccessData.pBaseRadZ, NewRadAccessData.pBaseRadX, NewRadAccessData.pBaseRadZ);

	srTSRWRadStructAccessData* pOldRadAccessData_dev = CAuxGPU::ToDevice(pGPU, &OldRadAccessData, 1);
	srTSRWRadStructAccessData* pNewRadAccessData_dev = CAuxGPU::ToDevice(pGPU, &NewRadAccessData, 1);
	CAuxGPU::EnsureDeviceMemoryReady(pGPU, pOldRadAccessData_dev, pNewRadAccessData_dev);

	dim3 blocks(nx, nz, ne);
	dim3 threads(32, 1);

	//Select the right kernel
	decltype(RadResizeCore_OnlyLargerRange_Kernel<true, true>) *kern = NULL;
	if (TreatPolCompX && TreatPolCompZ) kern = RadResizeCore_OnlyLargerRange_Kernel<true, true>;
	else if (TreatPolCompX) kern = RadResizeCore_OnlyLargerRange_Kernel<true, false>;
	else if (TreatPolCompZ) kern = RadResizeCore_OnlyLargerRange_Kernel<false, true>;
	
	CAuxGPU::CalcLaunchDims(kern, blocks, blocks, threads);
	kern<<<blocks, threads>>> (pOldRadAccessData_dev, pNewRadAccessData_dev);

	CAuxGPU::ToHostAndFree(pGPU, pOldRadAccessData_dev);
	CAuxGPU::ToHostAndFree(pGPU, pNewRadAccessData_dev);

	OldRadAccessData.pBaseRadZ = CAuxGPU::ToHostAndFree(pGPU, OldRadAccessData.pBaseRadZ);
	OldRadAccessData.pBaseRadX = CAuxGPU::ToHostAndFree(pGPU, OldRadAccessData.pBaseRadX);

	CAuxGPU::MarkUpdated(pGPU, NewRadAccessData.pBaseRadX, CAuxGPU::DEVICE);
	CAuxGPU::MarkUpdated(pGPU, NewRadAccessData.pBaseRadZ, CAuxGPU::DEVICE);
	
	NewRadAccessData.pBaseRadX = CAuxGPU::GetHostPtr(pGPU, NewRadAccessData.pBaseRadX);
	NewRadAccessData.pBaseRadZ = CAuxGPU::GetHostPtr(pGPU, NewRadAccessData.pBaseRadZ);
	
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
	printf("\r\n%s\r\n", __func__);
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

	dim3 blocks(NewRadAccessData.nx, NewRadAccessData.nz, ne);
	dim3 threads(32, 1);
	
	//Select the right kernel
	decltype(RadResizeCore_OnlyLargerRangeE_Kernel<true, true>) *kern = NULL;
	if (TreatPolCompX && TreatPolCompZ) kern = RadResizeCore_OnlyLargerRangeE_Kernel<true, true>;
	else if (TreatPolCompX) kern = RadResizeCore_OnlyLargerRangeE_Kernel<true, false>;
	else if (TreatPolCompZ) kern = RadResizeCore_OnlyLargerRangeE_Kernel<false, true>;
	
	CAuxGPU::CalcLaunchDims(kern, blocks, blocks, threads);
	kern<<<blocks, threads>>> (pOldRadAccessData_dev, pNewRadAccessData_dev);
	
	CAuxGPU::ToHostAndFree(pGPU, pOldRadAccessData_dev); //HG27072024
	CAuxGPU::ToHostAndFree(pGPU, pNewRadAccessData_dev); //HG27072024

	OldRadAccessData.pBaseRadX = CAuxGPU::ToHostAndFree(pGPU, OldRadAccessData.pBaseRadX);
	OldRadAccessData.pBaseRadZ = CAuxGPU::ToHostAndFree(pGPU, OldRadAccessData.pBaseRadZ);
	CAuxGPU::MarkUpdated(pGPU, NewRadAccessData.pBaseRadX, CAuxGPU::DEVICE);
	CAuxGPU::MarkUpdated(pGPU, NewRadAccessData.pBaseRadZ, CAuxGPU::DEVICE);
	NewRadAccessData.pBaseRadX = CAuxGPU::GetHostPtr(pGPU, NewRadAccessData.pBaseRadX);
	NewRadAccessData.pBaseRadZ = CAuxGPU::GetHostPtr(pGPU, NewRadAccessData.pBaseRadZ);

	return 0;
}


//KernelMode:
//0 - Init
//1 - Common
//2 - X
//4 - Z
template<int KernelMode, bool ExIsOK, bool EzIsOK>
__global__ void ComputeRadMoments_Kernel(const srTSRWRadStructAccessData* pSRWRadStructAccessData, int4 IndLims, double* SumsZ, int ie, double TwoPi_d_Lamb_d_Rx_xStep, double TwoPi_d_Lamb_d_Rz_zStep)
{
	int ix = (blockIdx.x * blockDim.x + threadIdx.x); //nx range
	int iz = (blockIdx.y * blockDim.y + threadIdx.y); //nz range

	if (KernelMode == 0)
	{
		if (ix >= pSRWRadStructAccessData->nx) return;
		if (iz >= pSRWRadStructAccessData->nz) return;
	}
	else
	{
		ix += IndLims.x;
		iz += IndLims.z;
		if (ix > IndLims.y) return;
		if (iz > IndLims.w) return;
	}

	float* __restrict__ fpX0 = pSRWRadStructAccessData->pBaseRadX;
	float* __restrict__ fpZ0 = pSRWRadStructAccessData->pBaseRadZ;
	
	long PerX = pSRWRadStructAccessData->ne << 1;
	long long PerZ = PerX*pSRWRadStructAccessData->nx;
	
	int nx_mi_1 = pSRWRadStructAccessData->nx - 1;
	int nz_mi_1 = pSRWRadStructAccessData->nz - 1;
	
	long long izPerZ = iz*PerZ;
	float *fpX_StartForX = fpX0 + izPerZ;
	float *fpZ_StartForX = fpZ0 + izPerZ;
	

	long Two_ie = ie << 1;
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

	double TwoPi_d_Lamb_d_Rx_xStepE2 = TwoPi_d_Lamb_d_Rx_xStep*TwoPi_d_Lamb_d_Rx_xStep;
	double TwoPi_d_Lamb_d_Rz_zStepE2 = TwoPi_d_Lamb_d_Rz_zStep*TwoPi_d_Lamb_d_Rz_zStep;

	double ff[22] = {0.};
	
	double z = pSRWRadStructAccessData->zStart + iz*pSRWRadStructAccessData->zStep;
	double x = pSRWRadStructAccessData->xStart + ix*pSRWRadStructAccessData->xStep;
	ff[0] = ExRe*ExRe + ExIm*ExIm; // NormX
	ff[11] = EzRe*EzRe + EzIm*EzIm; // NormZ

	ff[1] = x* ff[0]; // <x>
	ff[3] = z* ff[0]; // <z>
	ff[12] = x*ff[11]; // <x>
	ff[14] = z*ff[11]; // <z>

	ff[5] = x*ff[1]; // <xx>
	ff[8] = z*ff[3]; // <zz>
	ff[16] = x*ff[12]; // <xx>
	ff[19] = z*ff[14]; // <zz>

	if(ix > 0)
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
		ff[2] = 0.; // <x'>
		ff[6] = 0.; // <xx'>
		ff[7] = 0.; // <x'x'>
		ff[13] = 0.; // <x'>
		ff[17] = 0.; // <xx'>
		ff[18] = 0.; // <x'x'>
	}

	if(iz > 0)
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
		ff[4] = 0.; // <z'>
		ff[9] = 0.; // <zz'>
		ff[10] = 0.; // <z'z'>
		ff[15] = 0.; // <z'>
		ff[20] = 0.; // <zz'>
		ff[21] = 0.; // <z'z'>
	}

	if((ix == 0) || (ix == nx_mi_1))
	{
		ff[0] *= 0.5;
		ff[1] *= 0.5;
		ff[2] *= 0.5;
		ff[3] *= 0.5;
		ff[4] *= 0.5;
		ff[5] *= 0.5;
		ff[6] *= 0.5;
		ff[7] *= 0.5;
		ff[8] *= 0.5;
		ff[9] *= 0.5;
		ff[10] *= 0.5;
		ff[11] *= 0.5;
		ff[12] *= 0.5;
		ff[13] *= 0.5;
		ff[14] *= 0.5;
		ff[15] *= 0.5;
		ff[16] *= 0.5;
		ff[17] *= 0.5;
		ff[18] *= 0.5;
		ff[19] *= 0.5;
		ff[20] *= 0.5;
		ff[21] *= 0.5;
	}
	if(ix == 1)
	{
		ff[2] *= 0.5; // <x'>>
		ff[6] *= 0.5; // <xx'>
		ff[7] *= 0.5; // <x'x'>
		ff[13] *= 0.5; // <x'>
		ff[17] *= 0.5; // <xx'>
		ff[18] *= 0.5; // <x'x'>
	}

	if((iz == 0) || (iz == nz_mi_1))
	{
		ff[0] *= 0.5;
		ff[1] *= 0.5;
		ff[2] *= 0.5;
		ff[3] *= 0.5;
		ff[4] *= 0.5;
		ff[5] *= 0.5;
		ff[6] *= 0.5;
		ff[7] *= 0.5;
		ff[8] *= 0.5;
		ff[9] *= 0.5;
		ff[10] *= 0.5;
		ff[11] *= 0.5;
		ff[12] *= 0.5;
		ff[13] *= 0.5;
		ff[14] *= 0.5;
		ff[15] *= 0.5;
		ff[16] *= 0.5;
		ff[17] *= 0.5;
		ff[18] *= 0.5;
		ff[19] *= 0.5;
		ff[20] *= 0.5;
		ff[21] *= 0.5;
	}
	if(iz == 1)
	{
		ff[4] *= 0.5; // <z'>
		ff[9] *= 0.5; // <zz'>
		ff[10] *= 0.5; // <z'z'>
		ff[15] *= 0.5; // <z'>
		ff[20] *= 0.5; // <zz'>
		ff[21] *= 0.5; // <z'z'>
	}

	cg::coalesced_group g = cg::coalesced_threads();
	if (KernelMode == 0)
	{
		//Computed for whole mesh
		ff[0] = cg::reduce(g, ff[0], cg::plus<double>());
		ff[11] = cg::reduce(g, ff[11], cg::plus<double>());
		ff[1] = cg::reduce(g, ff[1], cg::plus<double>());
		ff[3] = cg::reduce(g, ff[3], cg::plus<double>());
		ff[12] = cg::reduce(g, ff[12], cg::plus<double>());
		ff[14] = cg::reduce(g, ff[14], cg::plus<double>());

		//Computed for ix > 0
		ff[2] = cg::reduce(g, ff[2], cg::plus<double>());
		ff[13] = cg::reduce(g, ff[13], cg::plus<double>());

		//Computed for iz > 0
		ff[4] = cg::reduce(g, ff[4], cg::plus<double>());
		ff[15] = cg::reduce(g, ff[15], cg::plus<double>());
	}
	else
	{
		//All only computed within IndLims range
		if (KernelMode & 1)
		{
			//Just within IndLims
			ff[5] = cg::reduce(g, ff[5], cg::plus<double>());
			ff[8] = cg::reduce(g, ff[8], cg::plus<double>());
			ff[16] = cg::reduce(g, ff[16], cg::plus<double>());
			ff[19] = cg::reduce(g, ff[19], cg::plus<double>());
		}
		if (KernelMode & 2)
		{
			//ix > 0
			ff[6] = cg::reduce(g, ff[6], cg::plus<double>());
			ff[7] = cg::reduce(g, ff[7], cg::plus<double>());
			ff[17] = cg::reduce(g, ff[17], cg::plus<double>());
			ff[18] = cg::reduce(g, ff[18], cg::plus<double>());
		}
		if (KernelMode & 4)
		{
			//iz > 0
			ff[9] = cg::reduce(g, ff[9], cg::plus<double>());
			ff[10] = cg::reduce(g, ff[10], cg::plus<double>());
			ff[20] = cg::reduce(g, ff[20], cg::plus<double>());
			ff[21] = cg::reduce(g, ff[21], cg::plus<double>());
		}
	}

	if(g.thread_rank() == 0)
	{
		if (KernelMode == 0)
		{
			atomicAdd(SumsZ, ff[0]);
			atomicAdd(SumsZ + 11, ff[11]);
			atomicAdd(SumsZ + 1, ff[1]);
			atomicAdd(SumsZ + 3, ff[3]);
			atomicAdd(SumsZ + 12, ff[12]);
			atomicAdd(SumsZ + 14, ff[14]);
			atomicAdd(SumsZ + 2, ff[2]);
			atomicAdd(SumsZ + 13, ff[13]);
			atomicAdd(SumsZ + 4, ff[4]);
			atomicAdd(SumsZ + 15, ff[15]);
		}
		else
		{
			if (KernelMode & 1)
			{
				atomicAdd(SumsZ + 5, ff[5]);
				atomicAdd(SumsZ + 8, ff[8]);
				atomicAdd(SumsZ + 16, ff[16]);
				atomicAdd(SumsZ + 19, ff[19]);
			}
			if (KernelMode & 2)
			{
				atomicAdd(SumsZ + 6, ff[6]);
				atomicAdd(SumsZ + 7, ff[7]);
				atomicAdd(SumsZ + 17, ff[17]);
				atomicAdd(SumsZ + 18, ff[18]);
			}
			if (KernelMode & 4)
			{
				atomicAdd(SumsZ + 9, ff[9]);
				atomicAdd(SumsZ + 10, ff[10]);
				atomicAdd(SumsZ + 20, ff[20]);
				atomicAdd(SumsZ + 21, ff[21]);
			}
		}
	}
}

void srTGenOptElem::ComputeRadMoments_GPU(srTSRWRadStructAccessData* pSRWRadStructAccessData, int ie, double* SumsZ, int* IndLims, TGPUUsageArg* pGPU) //HG26072024
{
	#define GEN_MEMBERS(i) \
	ComputeRadMoments_Kernel <i, false, false>, \
	ComputeRadMoments_Kernel <i, false, true>, \
	ComputeRadMoments_Kernel <i, true, false>, \
	ComputeRadMoments_Kernel <i, true, true>,
	
	decltype(ComputeRadMoments_Kernel <0, false, false>) *ComputeRadMoments_tbl[] = {
		GEN_MEMBERS(0)
		GEN_MEMBERS(1)
		GEN_MEMBERS(2)
		GEN_MEMBERS(4)
	};
	#undef GEN_MEMBERS
	
	bool ExIsOK = pSRWRadStructAccessData->pBaseRadX != 0;
	bool EzIsOK = pSRWRadStructAccessData->pBaseRadZ != 0;
	bool IsFreqRepres = (pSRWRadStructAccessData->PresT == 0);
	bool IsCoordRepres = (pSRWRadStructAccessData->Pres == 0);
	const double TwoPi = 3.141592653590*2.;
	const double FourPi = TwoPi*2.;
	const double Inv_eV_In_m = 1.239842E-06;
	double ePh = pSRWRadStructAccessData->eStart + pSRWRadStructAccessData->eStep*ie; //This assumes wavefront in Time domain; Photon Energy in eV !
	if(!IsFreqRepres)
	{
		ePh = pSRWRadStructAccessData->avgPhotEn; //?? OC041108
	}
	double Lamb_d_FourPi = Inv_eV_In_m/(FourPi*ePh);
	double Lamb_m = Lamb_d_FourPi*FourPi;
	double FourPi_d_Lamb = 1./Lamb_d_FourPi;
	double LocRobsX = pSRWRadStructAccessData->RobsX; //OC030409
	if(LocRobsX == 0.) LocRobsX = 100.*Lamb_m;
	double LocRobsZ = pSRWRadStructAccessData->RobsZ;
	if(LocRobsZ == 0.) LocRobsZ = 100.*Lamb_m;
	double FourPi_d_Lamb_d_Rz = FourPi_d_Lamb/LocRobsZ;
	double FourPi_d_Lamb_d_Rz_zStep = pSRWRadStructAccessData->zStep*FourPi_d_Lamb_d_Rz;
	double TwoPi_d_Lamb_d_Rz_zStep = 0.5*FourPi_d_Lamb_d_Rz_zStep;
	double FourPi_d_Lamb_d_Rx = FourPi_d_Lamb/LocRobsX;
	double FourPi_d_Lamb_d_Rx_xStep = pSRWRadStructAccessData->xStep*FourPi_d_Lamb_d_Rx;
	double TwoPi_d_Lamb_d_Rx_xStep = 0.5*FourPi_d_Lamb_d_Rx_xStep;
	
	dim3 blocks0(IndLims[1] - IndLims[0] + 1, IndLims[3] - IndLims[2] + 1, 1);
	dim3 blocks1(IndLims[1] - IndLims[0] + 1, IndLims[3] - IndLims[2] + 1, 1);
	dim3 blocks2(IndLims[1] - IndLims[0] + 1, IndLims[3] - IndLims[2] + 1, 1);
	dim3 blocks3(pSRWRadStructAccessData->nx, pSRWRadStructAccessData->nz, 1);
	dim3 blocks00(IndLims[1] - IndLims[0] + 1, IndLims[3] - IndLims[2] + 1, 1);
	dim3 blocks01(IndLims[1] - IndLims[0] + 1, IndLims[3] - IndLims[2] + 1, 1);
	dim3 blocks02(IndLims[1] - IndLims[0] + 1, IndLims[3] - IndLims[2] + 1, 1);
	dim3 blocks03(pSRWRadStructAccessData->nx, pSRWRadStructAccessData->nz, 1);
	dim3 threads0(1);
	dim3 threads1(1);
	dim3 threads2(1);
	dim3 threads3(1);
	CAuxGPU::CalcLaunchDims(ComputeRadMoments_tbl[(1 << 2) | ((ExIsOK & 1) << 1) | (EzIsOK & 1)], blocks0, blocks0, threads0, 16);
	CAuxGPU::CalcLaunchDims(ComputeRadMoments_tbl[(2 << 2) | ((ExIsOK & 1) << 1) | (EzIsOK & 1)], blocks1, blocks1, threads1, 16);
	CAuxGPU::CalcLaunchDims(ComputeRadMoments_tbl[(3 << 2) | ((ExIsOK & 1) << 1) | (EzIsOK & 1)], blocks2, blocks2, threads2, 16);
	CAuxGPU::CalcLaunchDims(ComputeRadMoments_tbl[(0 << 2) | ((ExIsOK & 1) << 1) | (EzIsOK & 1)], blocks3, blocks3, threads3, 16);
	
	double* sumz = SumsZ;
	pSRWRadStructAccessData->pBaseRadX = CAuxGPU::ToDevice(pGPU, pSRWRadStructAccessData->pBaseRadX, 2 * pSRWRadStructAccessData->ne * pSRWRadStructAccessData->nx * pSRWRadStructAccessData->nz);
	pSRWRadStructAccessData->pBaseRadZ = CAuxGPU::ToDevice(pGPU, pSRWRadStructAccessData->pBaseRadZ, 2 * pSRWRadStructAccessData->ne * pSRWRadStructAccessData->nx * pSRWRadStructAccessData->nz);
	SumsZ = CAuxGPU::ToDevice(pGPU, SumsZ, 22, CAuxGPU::DONT_COPY);
	CAuxGPU::Memset(pGPU, SumsZ, 0.0, 22);
	srTSRWRadStructAccessData* pSRWRadStructAccessData_dev = CAuxGPU::ToDevice(pGPU, pSRWRadStructAccessData, 1);
	
	CAuxGPU::EnsureDeviceMemoryReady(pGPU, 
		pSRWRadStructAccessData->pBaseRadX,
		pSRWRadStructAccessData->pBaseRadZ,
		SumsZ,
		pSRWRadStructAccessData_dev
	);
	
	int4 IndLims_dev = { IndLims[0], IndLims[1], IndLims[2], IndLims[3] };

	cudaStream_t stream1 = (cudaStream_t)CAuxGPU::GetComputeStream(pGPU, 0);
	cudaStream_t stream2 = (cudaStream_t)CAuxGPU::GetComputeStream(pGPU, 1);
	cudaStream_t stream3 = (cudaStream_t)CAuxGPU::GetComputeStream(pGPU, 2);

	CAuxGPU::SyncComputeStream(pGPU, 0, (long long)stream1);
	CAuxGPU::SyncComputeStream(pGPU, 0, (long long)stream2);
	CAuxGPU::SyncComputeStream(pGPU, 0, (long long)stream3);

	ComputeRadMoments_tbl[(1 << 2) | ((ExIsOK & 1) << 1) | (EzIsOK & 1)] <<<blocks0, threads0 >>>(pSRWRadStructAccessData_dev, IndLims_dev, SumsZ, ie, TwoPi_d_Lamb_d_Rx_xStep, TwoPi_d_Lamb_d_Rz_zStep);
	if(IsCoordRepres) 
	{
		ComputeRadMoments_tbl[(2 << 2) | ((ExIsOK & 1) << 1) | (EzIsOK & 1)] <<<blocks1, threads1, 0, stream1 >>>(pSRWRadStructAccessData_dev, IndLims_dev, SumsZ, ie, TwoPi_d_Lamb_d_Rx_xStep, TwoPi_d_Lamb_d_Rz_zStep);
		ComputeRadMoments_tbl[(3 << 2) | ((ExIsOK & 1) << 1) | (EzIsOK & 1)] <<<blocks2, threads2, 0, stream2 >>>(pSRWRadStructAccessData_dev, IndLims_dev, SumsZ, ie, TwoPi_d_Lamb_d_Rx_xStep, TwoPi_d_Lamb_d_Rz_zStep);
	}
	ComputeRadMoments_tbl[(0 << 2) | ((ExIsOK & 1) << 1) | (EzIsOK & 1)] <<<blocks3, threads3, 0, stream3>>>(pSRWRadStructAccessData_dev, IndLims_dev, SumsZ, ie, TwoPi_d_Lamb_d_Rx_xStep, TwoPi_d_Lamb_d_Rz_zStep);

	CAuxGPU::SyncComputeStream(pGPU, (long long)stream1, 0);
	CAuxGPU::SyncComputeStream(pGPU, (long long)stream2, 0);
	CAuxGPU::SyncComputeStream(pGPU, (long long)stream3, 0);
	
	CAuxGPU::ToHostAndFree(pGPU, pSRWRadStructAccessData_dev);
	
	pSRWRadStructAccessData->pBaseRadX = CAuxGPU::GetHostPtr(pGPU, pSRWRadStructAccessData->pBaseRadX);
	pSRWRadStructAccessData->pBaseRadZ = CAuxGPU::GetHostPtr(pGPU, pSRWRadStructAccessData->pBaseRadZ);
	
	CAuxGPU::MarkUpdated(pGPU, SumsZ, CAuxGPU::DEVICE);
	CAuxGPU::ToHostAndFree(pGPU, SumsZ);
}

const int PerThreadOps = 16;

__global__ void ExtractRadSliceConstE_Kernel(srTSRWRadStructAccessData *pRadAccessData, long ie, float* __restrict__ pOutEx, float* __restrict__ pOutEz)
{
	int ix = (blockIdx.x * blockDim.x + threadIdx.x) * PerThreadOps; //nx range
	int iz = (blockIdx.y * blockDim.y + threadIdx.y); //nz range

	if(iz >= pRadAccessData->nz) return;

	for (int i = 0; i < PerThreadOps; i++)
	{
		if(ix >= pRadAccessData->nx) return;

		float* __restrict__ pEx0 = pRadAccessData->pBaseRadX;
		float* __restrict__ pEz0 = pRadAccessData->pBaseRadZ;
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

		ix++;
	}
}

int srTGenOptElem::ExtractRadSliceConstE_GPU(srTSRWRadStructAccessData* pRadAccessData, long ie, float* pOutEx, float* pOutEz, TGPUUsageArg* pGPU)
{
	printf("\r\n%s\r\n", __func__);
	pRadAccessData->pBaseRadX = CAuxGPU::ToDevice(pGPU, pRadAccessData->pBaseRadX, 2 * pRadAccessData->ne * pRadAccessData->nx * pRadAccessData->nz);
	pRadAccessData->pBaseRadZ = CAuxGPU::ToDevice(pGPU, pRadAccessData->pBaseRadZ, 2 * pRadAccessData->ne * pRadAccessData->nx * pRadAccessData->nz);
	pOutEx = CAuxGPU::ToDevice(pGPU, pOutEx, 2 * pRadAccessData->nx * pRadAccessData->nz, CAuxGPU::DONT_COPY);
	pOutEz = CAuxGPU::ToDevice(pGPU, pOutEz, 2 * pRadAccessData->nx * pRadAccessData->nz, CAuxGPU::DONT_COPY);

	CAuxGPU::EnsureDeviceMemoryReady(pGPU, 
		pRadAccessData->pBaseRadX,
		pRadAccessData->pBaseRadZ,
		pOutEx,
		pOutEz
	);

	srTSRWRadStructAccessData* pRadAccessData_dev = CAuxGPU::ToDevice(pGPU, pRadAccessData, 1);
	CAuxGPU::EnsureDeviceMemoryReady(pGPU, pRadAccessData_dev);

	dim3 blocks(pRadAccessData->nx / PerThreadOps + !!(pRadAccessData->nx % PerThreadOps), pRadAccessData->nz, 1);
	dim3 threads(1);
	CAuxGPU::CalcLaunchDims(ExtractRadSliceConstE_Kernel, blocks, blocks, threads);

	ExtractRadSliceConstE_Kernel <<<blocks, threads >>> (pRadAccessData_dev, ie, pOutEx, pOutEz);

	CAuxGPU::ToHostAndFree(pGPU, pRadAccessData_dev);

	CAuxGPU::MarkUpdated(pGPU, pOutEx, CAuxGPU::DEVICE);
	CAuxGPU::MarkUpdated(pGPU, pOutEz, CAuxGPU::DEVICE);

	pRadAccessData->pBaseRadX = CAuxGPU::GetHostPtr(pGPU, pRadAccessData->pBaseRadX);
	pRadAccessData->pBaseRadZ = CAuxGPU::GetHostPtr(pGPU, pRadAccessData->pBaseRadZ);
	pOutEx = CAuxGPU::GetHostPtr(pGPU, pOutEx);
	pOutEz = CAuxGPU::GetHostPtr(pGPU, pOutEz);
	return 0;
}

__global__ void UpdateGenRadStructSliceConstE_Meth_0_Kernel(srTSRWRadStructAccessData* pRadDataSliceConstE, int ie, srTSRWRadStructAccessData* pRadAccessData)
{
	int ix = (blockIdx.x * blockDim.x + threadIdx.x) * PerThreadOps; //nx range
	int iz = (blockIdx.y * blockDim.y + threadIdx.y); //nz range

	int neCom = pRadAccessData->ne;
	int nxCom = pRadAccessData->nx;
	int nzCom = pRadAccessData->nz;

	if(ix >= nxCom) return;
	if(iz >= nzCom) return;

	for (int i = 0; i < PerThreadOps; i++)
	{
		if(ix >= nxCom) return;

		float* __restrict__ pEx0 = pRadAccessData->pBaseRadX;
		float* __restrict__ pEz0 = pRadAccessData->pBaseRadZ;

		long long PerX = neCom << 1;
		long long PerZ = PerX*nxCom;
		long long iePerE = ie << 1;

		float* __restrict__ tSliceEx = pRadDataSliceConstE->pBaseRadX + iz*nxCom*2 + ix*2;
		float* __restrict__ tSliceEz = pRadDataSliceConstE->pBaseRadZ + iz*nxCom*2 + ix*2;

		float *pEx = pEx0 + iz*PerZ + ix*PerX + iePerE;
		float *pEz = pEz0 + iz*PerZ + ix*PerX + iePerE;

		*(pEx++) = *(tSliceEx++); *pEx = *(tSliceEx);
		*(pEz++) = *(tSliceEz++); *pEz = *(tSliceEz);

		ix++;
	}
}

int srTGenOptElem::UpdateGenRadStructSliceConstE_Meth_0_GPU(srTSRWRadStructAccessData* pRadDataSliceConstE, int ie, srTSRWRadStructAccessData* pRadAccessData, TGPUUsageArg* pGPU)
{
	pRadAccessData->pBaseRadX = CAuxGPU::ToDevice(pGPU, pRadAccessData->pBaseRadX, 2 * pRadAccessData->ne * pRadAccessData->nx * pRadAccessData->nz);
	pRadAccessData->pBaseRadZ = CAuxGPU::ToDevice(pGPU, pRadAccessData->pBaseRadZ, 2 * pRadAccessData->ne * pRadAccessData->nx * pRadAccessData->nz);
	pRadDataSliceConstE->pBaseRadX = CAuxGPU::ToDevice(pGPU, pRadDataSliceConstE->pBaseRadX, 2 * pRadAccessData->nx * pRadAccessData->nz);
	pRadDataSliceConstE->pBaseRadZ = CAuxGPU::ToDevice(pGPU, pRadDataSliceConstE->pBaseRadZ, 2 * pRadAccessData->nx * pRadAccessData->nz);

	CAuxGPU::EnsureDeviceMemoryReady(pGPU, 
		pRadAccessData->pBaseRadX,
		pRadAccessData->pBaseRadZ,
		pRadDataSliceConstE->pBaseRadX,
		pRadDataSliceConstE->pBaseRadZ
	);

	srTSRWRadStructAccessData* pRadDataSliceConstE_dev = CAuxGPU::ToDevice(pGPU, pRadDataSliceConstE, 1);
	srTSRWRadStructAccessData* pRadAccessData_dev = CAuxGPU::ToDevice(pGPU, pRadAccessData, 1);
	CAuxGPU::EnsureDeviceMemoryReady(pGPU, pRadDataSliceConstE_dev, pRadAccessData_dev);

	dim3 blocks(pRadAccessData->nx / PerThreadOps + !!(pRadAccessData->nx % PerThreadOps), pRadAccessData->nz, 1);
	dim3 threads(1);
	CAuxGPU::CalcLaunchDims(UpdateGenRadStructSliceConstE_Meth_0_Kernel, blocks, blocks, threads);

	UpdateGenRadStructSliceConstE_Meth_0_Kernel <<<blocks, threads >>> (pRadDataSliceConstE_dev, ie, pRadAccessData_dev);

	CAuxGPU::ToHostAndFree(pGPU, pRadAccessData_dev);
	CAuxGPU::ToHostAndFree(pGPU, pRadDataSliceConstE_dev);

	pRadDataSliceConstE->pBaseRadX = CAuxGPU::ToHostAndFree(pGPU, pRadDataSliceConstE->pBaseRadX);
	pRadDataSliceConstE->pBaseRadZ = CAuxGPU::ToHostAndFree(pGPU, pRadDataSliceConstE->pBaseRadZ);

	CAuxGPU::MarkUpdated(pGPU, pRadAccessData->pBaseRadX, CAuxGPU::DEVICE);
	CAuxGPU::MarkUpdated(pGPU, pRadAccessData->pBaseRadZ, CAuxGPU::DEVICE);
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
	printf("\r\n%s\r\n", __func__);
	oldRadSingleE.pBaseRadX = CAuxGPU::ToDevice(pGPU, oldRadSingleE.pBaseRadX, 2 * oldRadSingleE.ne * oldRadSingleE.nx * oldRadSingleE.nz);
	oldRadSingleE.pBaseRadZ = CAuxGPU::ToDevice(pGPU, oldRadSingleE.pBaseRadZ, 2 * oldRadSingleE.ne * oldRadSingleE.nx * oldRadSingleE.nz);
	newRadMultiE.pBaseRadX = CAuxGPU::ToDevice(pGPU, newRadMultiE.pBaseRadX, 2 * newRadMultiE.ne * newRadMultiE.nx * newRadMultiE.nz);
	newRadMultiE.pBaseRadZ = CAuxGPU::ToDevice(pGPU, newRadMultiE.pBaseRadZ, 2 * newRadMultiE.ne * newRadMultiE.nx * newRadMultiE.nz);

	CAuxGPU::EnsureDeviceMemoryReady(pGPU, 
		oldRadSingleE.pBaseRadX,
		oldRadSingleE.pBaseRadZ,
		newRadMultiE.pBaseRadX,
		newRadMultiE.pBaseRadZ
	);

	dim3 blocks(newRadMultiE.nx, newRadMultiE.nz, 1);
	dim3 threads(1);
	CAuxGPU::CalcLaunchDims(ReInterpolateWfrSliceSingleE_Kernel, blocks, blocks, threads);
	
	srTSRWRadStructAccessData* pOldRadSingleE_dev = CAuxGPU::ToDevice(pGPU, &oldRadSingleE, 1);
	CAuxGPU::EnsureDeviceMemoryReady(pGPU, pOldRadSingleE_dev);

	srTSRWRadStructAccessData* pNewRadMultiE_dev = CAuxGPU::ToDevice(pGPU, &newRadMultiE, 1);
	CAuxGPU::EnsureDeviceMemoryReady(pGPU, pNewRadMultiE_dev);

	ReInterpolateWfrSliceSingleE_Kernel <<<blocks, threads >>> (pOldRadSingleE_dev, pNewRadMultiE_dev, ie);

	CAuxGPU::ToHostAndFree(pGPU, pNewRadMultiE_dev);
	CAuxGPU::ToHostAndFree(pGPU, pOldRadSingleE_dev);

	oldRadSingleE.pBaseRadX = CAuxGPU::ToHostAndFree(pGPU, oldRadSingleE.pBaseRadX);
	oldRadSingleE.pBaseRadZ = CAuxGPU::ToHostAndFree(pGPU, oldRadSingleE.pBaseRadZ);

	CAuxGPU::MarkUpdated(pGPU, newRadMultiE.pBaseRadX, CAuxGPU::DEVICE);
	CAuxGPU::MarkUpdated(pGPU, newRadMultiE.pBaseRadZ, CAuxGPU::DEVICE);
	newRadMultiE.pBaseRadX = CAuxGPU::GetHostPtr(pGPU, newRadMultiE.pBaseRadX);
	newRadMultiE.pBaseRadZ = CAuxGPU::GetHostPtr(pGPU, newRadMultiE.pBaseRadZ);

	return 0;
}

__global__ void SetupRadSliceConstE_Kernel(srTSRWRadStructAccessData* pRadAccessData, long ie, float* __restrict__ pInEx, float* __restrict__ pInEz)
{
	int ix = (blockIdx.x * blockDim.x + threadIdx.x) * PerThreadOps; //nx range
	int iz = (blockIdx.y * blockDim.y + threadIdx.y); //nz range

	if(iz >= pRadAccessData->nz) return;

	for (int i = 0; i < PerThreadOps; i++)
	{
		if(ix >= pRadAccessData->nx) return;
		
		float *pEx0 = pRadAccessData->pBaseRadX + iz * pRadAccessData->nx * pRadAccessData->ne * 2 + ix * pRadAccessData->ne * 2 + ie * 2;
		float *pEz0 = pRadAccessData->pBaseRadZ + iz * pRadAccessData->nx * pRadAccessData->ne * 2 + ix * pRadAccessData->ne * 2 + ie * 2;
		float *tInEx = pInEx + iz * pRadAccessData->nx * 2 + ix * 2;
		float *tInEz = pInEz + iz * pRadAccessData->nx * 2 + ix * 2;
		
		*pEx0 = *tInEx;
		*(pEx0+1) = *(tInEx+1);
		*pEz0 = *tInEz;
		*(pEz0+1) = *(tInEz+1);

		ix++;
	}
}

int srTGenOptElem::SetupRadSliceConstE_GPU(srTSRWRadStructAccessData* pRadAccessData, long ie, float* pInEx, float* pInEz, TGPUUsageArg* pGPU)
{
	printf("\r\n%s\r\n", __func__);
	pRadAccessData->pBaseRadX = CAuxGPU::ToDevice(pGPU, pRadAccessData->pBaseRadX, 2 * pRadAccessData->ne * pRadAccessData->nx * pRadAccessData->nz);
	pRadAccessData->pBaseRadZ = CAuxGPU::ToDevice(pGPU, pRadAccessData->pBaseRadZ, 2 * pRadAccessData->ne * pRadAccessData->nx * pRadAccessData->nz);
	pInEx = CAuxGPU::ToDevice(pGPU, pInEx, 2*pRadAccessData->nx * pRadAccessData->nz);
	pInEz = CAuxGPU::ToDevice(pGPU, pInEz, 2*pRadAccessData->nx * pRadAccessData->nz);

	CAuxGPU::EnsureDeviceMemoryReady(pGPU, 
		pRadAccessData->pBaseRadX,
		pRadAccessData->pBaseRadZ,
		pInEx,
		pInEz
	);

	srTSRWRadStructAccessData* pRadAccessData_dev = CAuxGPU::ToDevice(pGPU, pRadAccessData, 1);
	CAuxGPU::EnsureDeviceMemoryReady(pGPU, pRadAccessData_dev);
	dim3 blocks(pRadAccessData->nx / PerThreadOps + !!(pRadAccessData->nx % PerThreadOps), pRadAccessData->nz, 1);
	dim3 threads(1);
	CAuxGPU::CalcLaunchDims(SetupRadSliceConstE_Kernel, blocks, blocks, threads);
	
	SetupRadSliceConstE_Kernel <<<blocks, threads >>> (pRadAccessData_dev, ie, pInEx, pInEz);

	CAuxGPU::ToHostAndFree(pGPU, pRadAccessData_dev);

	CAuxGPU::MarkUpdated(pGPU, pRadAccessData->pBaseRadX, CAuxGPU::DEVICE);
	CAuxGPU::MarkUpdated(pGPU, pRadAccessData->pBaseRadZ, CAuxGPU::DEVICE);
	pRadAccessData->pBaseRadX = CAuxGPU::GetHostPtr(pGPU, pRadAccessData->pBaseRadX);
	pRadAccessData->pBaseRadZ = CAuxGPU::GetHostPtr(pGPU, pRadAccessData->pBaseRadZ);

	CAuxGPU::ToHostAndFree(pGPU, pInEx);
	CAuxGPU::ToHostAndFree(pGPU, pInEz);

	return 0;
}

template <char vsX_or_vsZ>
__global__ void SetupRadXorZSectFromSliceConstE_Kernel(float* pInEx, float* pInEz, long nx, long nz, long iSect, float* pOutEx, float* pOutEz)
{
	int ix = (blockIdx.x * blockDim.x + threadIdx.x) * PerThreadOps;
	
	long long Per = (vsX_or_vsZ == 'x')? 2 : (nx << 1);
	long long StartOffset = (vsX_or_vsZ == 'x')? iSect*(nx << 1) : (iSect << 1);
	long long nPt = (vsX_or_vsZ == 'x')? nx : nz;

	for(int i = 0; i < PerThreadOps; i++)
	{
		if (ix >= nPt) return;
	
		float *tOutEx = &pOutEx[ix * 2];
		float *tOutEz = &pOutEz[ix * 2];
		float *tEx = &pInEx[StartOffset + ix * Per];
		float *tEz = &pInEz[StartOffset + ix * Per];

		tOutEx[0] = tEx[0];
		tOutEx[1] = tEx[1];
		tOutEz[0] = tEz[0];
		tOutEz[1] = tEz[1];

		ix++;
	}
}

void srTGenOptElem::SetupRadXorZSectFromSliceConstE_GPU(float* pInEx, float* pInEz, long nx, long nz, char vsX_or_vsZ, long iSect, float* pOutEx, float* pOutEz, TGPUUsageArg* pGPU)
{
	printf("\r\n%s\r\n", __func__);
	pInEx = CAuxGPU::ToDevice(pGPU, pInEx, 2*nx*nz);
	pInEz = CAuxGPU::ToDevice(pGPU, pInEz, 2*nx*nz);
	pOutEx = CAuxGPU::ToDevice(pGPU, pOutEx, (vsX_or_vsZ == 'x')? 2*nx : 2*nz, CAuxGPU::DONT_COPY);
	pOutEz = CAuxGPU::ToDevice(pGPU, pOutEz, (vsX_or_vsZ == 'x')? 2*nx : 2*nz, CAuxGPU::DONT_COPY);

	CAuxGPU::EnsureDeviceMemoryReady(pGPU, 
		pInEx,
		pInEz
	);

	long long nPt = (vsX_or_vsZ == 'x')? nx : nz;

	dim3 blocks(nPt / PerThreadOps + !!(nPt % PerThreadOps), 1, 1);
	dim3 threads(1);

	if (vsX_or_vsZ == 'x')
	{
		CAuxGPU::CalcLaunchDims(SetupRadXorZSectFromSliceConstE_Kernel<'x'>, blocks, blocks, threads);
		SetupRadXorZSectFromSliceConstE_Kernel<'x'> <<<blocks, threads >>> (pInEx, pInEz, nx, nz, iSect, pOutEx, pOutEz);
	} 
	else
	{
		CAuxGPU::CalcLaunchDims(SetupRadXorZSectFromSliceConstE_Kernel<'z'>, blocks, blocks, threads);
		SetupRadXorZSectFromSliceConstE_Kernel<'z'> <<<blocks, threads >>> (pInEx, pInEz, nx, nz, iSect, pOutEx, pOutEz);
	}

	CAuxGPU::MarkUpdated(pGPU, pOutEx, CAuxGPU::DEVICE);
	CAuxGPU::MarkUpdated(pGPU, pOutEz, CAuxGPU::DEVICE);
}

#endif