/************************************************************************//**
 * File: sroptdrf_gpu.cu
 * Description: Optical element: Aperture (CUDA implementation)
 * Project: Synchrotron Radiation Workshop
 * First release: 2024
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

#include <stdio.h>
#include <iostream>
#include <chrono>
#include "sroptapt.h"




//Implementation of the RadPointModifier's GPU function for the srTRectAperture class
int srTRectAperture::RadPointModifierParallel(srTSRWRadStructAccessData* pRadAccessData, void* pBufVars, long pBufVarsSz, TGPUUsageArg *pGPU) 
{ 
    if (TransHndl.rep != 0)
        return RadPointModifierParallelImpl<srTRectAperture>(pRadAccessData, pBufVars, pBufVarsSz, this, pGPU, 0, true); 
    else
    {
        //Calculate the bounds of the aperture
        const double SmallOffset = 1.E-10;
        const int Margin = 5;
        double EffHalfDx = HalfDx + SmallOffset, EffHalfDz = HalfDz + SmallOffset;

        int xStart = (-EffHalfDx - pRadAccessData->xStart) / pRadAccessData->xStep;
        int xEnd = (EffHalfDx - pRadAccessData->xStart) / pRadAccessData->xStep;
        int zStart = (-EffHalfDz - pRadAccessData->zStart) / pRadAccessData->zStep;
        int zEnd = (EffHalfDz - pRadAccessData->zStart) / pRadAccessData->zStep;

        xStart += Margin; xEnd -= Margin;
        zStart += Margin; zEnd -= Margin;
        if (xStart < 0) xStart = 0;
        if (xEnd >= pRadAccessData->nx) xEnd = pRadAccessData->nx - 1;
        if (zStart < 0) zStart = 0;
        if (zEnd >= pRadAccessData->nz) zEnd = pRadAccessData->nz - 1;

        if (xStart == 0 && xEnd == pRadAccessData->nx - 1 && zStart == 0 && zEnd == pRadAccessData->nz - 1)
        {
            return 0;
        }

        int region_params[5] = { xStart, xEnd, zStart, zEnd, 1 };
        return RadPointModifierParallelImpl<srTRectAperture>(pRadAccessData, pBufVars, pBufVarsSz, this, pGPU, region_params, true);
    }
}

//Implementation of the RadPointModifier's GPU function for the srTRectAperture class
int srTRectObstacle::RadPointModifierParallel(srTSRWRadStructAccessData* pRadAccessData, void* pBufVars, long pBufVarsSz, TGPUUsageArg* pGPU)
{
    if (TransHndl.rep != 0)
        return RadPointModifierParallelImpl<srTRectObstacle>(pRadAccessData, pBufVars, pBufVarsSz, this, pGPU,0, true); 
    else
    {
        //Calculate the bounds of the aperture
        const double SmallOffset = 1.E-10;
        const int Margin = 5;
        double EffHalfDx = HalfDx + SmallOffset, EffHalfDz = HalfDz + SmallOffset;

        int xStart = (-EffHalfDx - pRadAccessData->xStart) / pRadAccessData->xStep;
        int xEnd = (EffHalfDx - pRadAccessData->xStart) / pRadAccessData->xStep;
        int zStart = (-EffHalfDz - pRadAccessData->zStart) / pRadAccessData->zStep;
        int zEnd = (EffHalfDz - pRadAccessData->zStart) / pRadAccessData->zStep;

        xStart -= Margin; xEnd += Margin;
        zStart -= Margin; zEnd += Margin;
        if (xStart < 0) xStart = 0;
        if (xEnd >= pRadAccessData->nx) xEnd = pRadAccessData->nx - 1;
        if (zStart < 0) zStart = 0;
        if (zEnd >= pRadAccessData->nz) zEnd = pRadAccessData->nz - 1;

        int region_params[5] = { xStart, xEnd, zStart, zEnd, 0 };
        return RadPointModifierParallelImpl<srTRectObstacle>(pRadAccessData, pBufVars, pBufVarsSz, this, pGPU, region_params, true);
    }
}

//Implementation of the RadPointModifier's GPU function for the srTRectAperture class
int srTCircAperture::RadPointModifierParallel(srTSRWRadStructAccessData* pRadAccessData, void* pBufVars, long pBufVarsSz, TGPUUsageArg* pGPU)
{
    if (TransHndl.rep != 0)
        return RadPointModifierParallelImpl<srTCircAperture>(pRadAccessData, pBufVars, pBufVarsSz, this, pGPU, 0, true); 
    else
    {
        //Calculate the bounds of the aperture
        const double SmallOffset = 1.E-10;
        const int Margin = 5;

        //Calculate the bounds of the aperture as the square fully embedded in the circle
        double Side = (1.414213562373095 * R)/2 + SmallOffset;

        int xStart = ((TransvCenPoint.x-Side) - pRadAccessData->xStart) / pRadAccessData->xStep;
        int xEnd = ((TransvCenPoint.x+Side)  - pRadAccessData->xStart) / pRadAccessData->xStep;
        int zStart = ((TransvCenPoint.y-Side)  - pRadAccessData->zStart) / pRadAccessData->zStep;
        int zEnd = ((TransvCenPoint.y+Side)  - pRadAccessData->zStart) / pRadAccessData->zStep;

        xStart += Margin; xEnd -= Margin;
        zStart += Margin; zEnd -= Margin;
        if (xStart < 0) xStart = 0;
        if (xEnd >= pRadAccessData->nx) xEnd = pRadAccessData->nx - 1;
        if (zStart < 0) zStart = 0;
        if (zEnd >= pRadAccessData->nz) zEnd = pRadAccessData->nz - 1;

        if (xStart == 0 && xEnd == pRadAccessData->nx - 1 && zStart == 0 && zEnd == pRadAccessData->nz - 1)
        {
            return 0;
        }

        int region_params[5] = { xStart, xEnd, zStart, zEnd, 1 };
        return RadPointModifierParallelImpl<srTCircAperture>(pRadAccessData, pBufVars, pBufVarsSz, this, pGPU, region_params, true);
    }
}

//Implementation of the RadPointModifier's GPU function for the srTRectAperture class
int srTCircObstacle::RadPointModifierParallel(srTSRWRadStructAccessData* pRadAccessData, void* pBufVars, long pBufVarsSz, TGPUUsageArg* pGPU)
{
    if (TransHndl.rep != 0)
        return RadPointModifierParallelImpl<srTCircObstacle>(pRadAccessData, pBufVars, pBufVarsSz, this, pGPU, 0, true); 
    else
    {
        //Calculate the bounds of the aperture
        const double SmallOffset = 1.E-10;
        const int Margin = 5;

        //Calculate the bounds of the aperture as the square fully embedded in the circle
        double Side = R + SmallOffset;

        int xStart = ((TransvCenPoint.x-Side) - pRadAccessData->xStart) / pRadAccessData->xStep;
        int xEnd = ((TransvCenPoint.x+Side)  - pRadAccessData->xStart) / pRadAccessData->xStep;
        int zStart = ((TransvCenPoint.y-Side)  - pRadAccessData->zStart) / pRadAccessData->zStep;
        int zEnd = ((TransvCenPoint.y+Side)  - pRadAccessData->zStart) / pRadAccessData->zStep;

        xStart -= Margin; xEnd += Margin;
        zStart -= Margin; zEnd += Margin;
        if (xStart < 0) xStart = 0;
        if (xEnd >= pRadAccessData->nx) xEnd = pRadAccessData->nx - 1;
        if (zStart < 0) zStart = 0;
        if (zEnd >= pRadAccessData->nz) zEnd = pRadAccessData->nz - 1;

        int region_params[5] = { xStart, xEnd, zStart, zEnd, 0 };
        return RadPointModifierParallelImpl<srTCircObstacle>(pRadAccessData, pBufVars, pBufVarsSz, this, pGPU, region_params, true);
    }
}
#endif