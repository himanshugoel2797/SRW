/************************************************************************//**
 * File: sremitpr.h
 * Description: SR calculation, the case when single-electron emission couples with wavefront propagation (header)
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
#define __SREMITPR_H

#include "srstraux.h"
#include "sroptelm.h"

//*************************************************************************

class srTTrjDat;

//*************************************************************************

class srTEmitPropag {

public:

	static int ComputeRadiation(srTTrjDat&, srTGenOptElemHndl&, srTSRWRadStructAccessData&, double*);
	
	// TODO: Implement methods for coupling single-electron emission with wavefront propagation

};

//*************************************************************************

#endif
