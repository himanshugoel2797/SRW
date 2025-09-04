/************************************************************************//**
 * File: sremitpr.cpp
 * Description: SR calculation, case when single-electron emission couples with wavefront propagation (to be implemented?)
 * Project: Synchrotron Radiation Workshop
 * First release: 
 *
 * Copyright (C) European Synchrotron Radiation Facility, Grenoble, France
 * All Rights Reserved
 *
 * @author O.Chubar
 * @version 1.0
 ***************************************************************************/

#ifndef __SREMITPR_H
#include "sremitpr.h"
#endif

//*************************************************************************

int srTEmitPropag::ComputeRadiation(srTTrjDat& TrjDat, srTGenOptElemHndl& OptElemHndl, srTSRWRadStructAccessData& Wfr, double* PrecPar)
{
	int result = 0;

	// TODO: Implement ComputeRadiation method for coupling single-electron emission with wavefront propagation

	//if(result = OptElemHndl.rep->CheckRadStructForPropagation(&Wfr)) return result;

	return result;
}

//*************************************************************************
