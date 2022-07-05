/************************************************************************//**
 * File: gmfftgpu0.h
 * Description: Auxiliary utilities to perform CUDA accelerated FFTs (header)
 * Project: 
 * First release: 2022
 *
 * Copyright (C) European Synchrotron Radiation Facility, Grenoble, France
 * All Rights Reserved
 *
 * @author O.Chubar, P.Elleaume, H.Goel
 * @version 1.0
 ***************************************************************************/

#ifndef __GMFFTGPU0_H
#define __GMFFTGPU0_H

void RepairSignAfter1DFFT_CUDA(float* pAfterFFT, long HowMany, long Nx);
void RotateDataAfter1DFFT_CUDA(float* pAfterFFT, long HowMany, long Nx);
void RepairAndRotateDataAfter1DFFT_CUDA(float* pAfterFFT, long HowMany, long Nx, float Mult=1.f);
void NormalizeDataAfter1DFFT_CUDA(float* pAfterFFT, long HowMany, long Nx, double Mult);
void FillArrayShift_CUDA(double t0, double tStep, long Nx, float* tShiftX);
void TreatShift_CUDA(float* pData, long HowMany, long Nx, float* tShiftX);

void RepairSignAfter1DFFT_CUDA(double* pAfterFFT, long HowMany, long Nx);
void RotateDataAfter1DFFT_CUDA(double* pAfterFFT, long HowMany, long Nx);
void RepairAndRotateDataAfter1DFFT_CUDA(double* pAfterFFT, long HowMany, long Nx, double Mult=1.);
void NormalizeDataAfter1DFFT_CUDA(double* pAfterFFT, long HowMany, long Nx, double Mult);
void FillArrayShift_CUDA(double t0, double tStep, long Nx, double* tShiftX);
void TreatShift_CUDA(double* pData, long HowMany, long Nx, double* tShiftX);

void RepairSignAfter2DFFT_CUDA(float* pAfterFFT, long Nx, long Ny, long howMany);
void RotateDataAfter2DFFT_CUDA(float* pAfterFFT, long Nx, long Ny, long howMany);
void RepairSignAndRotateDataAfter2DFFT_CUDA(float* pAfterFFT, long Nx, long Ny, long howMany, float Mult=1.f);
void NormalizeDataAfter2DFFT_CUDA(float* pAfterFFT, long Nx, long Ny, long howMany, double Mult);
void TreatShifts2D_CUDA(float* pData, long Nx, long Ny, long howMany, bool NeedsShiftX, bool NeedsShiftY, float* m_ArrayShiftX, float* m_ArrayShiftY);

void RepairSignAfter2DFFT_CUDA(double* pAfterFFT, long Nx, long Ny, long howMany);
void RotateDataAfter2DFFT_CUDA(double* pAfterFFT, long Nx, long Ny, long howMany);
void RepairSignAndRotateDataAfter2DFFT_CUDA(double* pAfterFFT, long Nx, long Ny, long howMany, double Mult=1.);
void NormalizeDataAfter2DFFT_CUDA(double* pAfterFFT, long Nx, long Ny, long howMany, double Mult);
void TreatShifts2D_CUDA(double* pData, long Nx, long Ny, long howMany, bool NeedsShiftX, bool NeedsShiftY, double* m_ArrayShiftX, double* m_ArrayShiftY);

#endif // __GMFFTGPU0_H