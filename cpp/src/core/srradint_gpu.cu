/************************************************************************//**
 * File: srradint_gpu.cu
 * Description: SR undulator radiation integral (method "Auto1") - CUDA implementation
 * Project: Synchrotron Radiation Workshop
 * First release: 2026
 *
 * The CPU adaptive integrator (srTRadInt::RadIntegrationAuto1) carries a
 * loop-carried dependency across observation points: the convergence threshold
 * for point N is CurrentAbsPrec = sIntegRelPrec * max(SqNorm over points 0..N-1)
 * (members MaxFluxDensVal / ProbablyTheSameLoop). The GPU reformulation below is
 * parallel over observation points and breaks that dependency in two passes:
 *
 *   Pass A: every point integrates with the PER-POINT RELATIVE criterion
 *           |dSqNorm| <= relPrec*SqNorm (this is exactly the CPU criterion for
 *           the first / brightest point). Points that converge this way are at
 *           least as deeply converged as on the CPU.
 *   Pass B: points that did not converge relatively by the level cap are
 *           re-integrated with the ABSOLUTE criterion |dSqNorm| <= relPrec*MaxEst,
 *           where MaxEst = max SqNorm over all points from pass A. This is the
 *           CPU criterion with the running maximum replaced by the global
 *           maximum - i.e. CPU semantics with a different (order-independent)
 *           definition of the same tolerance.
 *
 * Deviation from CPU is therefore bounded by the integrator's own tolerance
 * (sIntegRelPrec * global peak SqNorm); it is NOT bit-identical. Points whose
 * endpoint residual needs the near-field treatment (or whose far-field expansion
 * fails), and points that do not converge by the level cap, are flagged and
 * recomputed on the CPU through the unchanged code path.
 *
 * Trajectory data: the CPU fills per-level midpoint arrays lazily; the union of
 * levels 0..L is a uniform grid of 4*2^L+1 points. The host precomputes that
 * finest uniform grid once per call (values are evaluated pointwise from the
 * interpolating structure, so they are grid-independent) and the kernel indexes
 * it with per-level strides.
 *
 * @author H.Goel (GPU port pattern), agent/undulator session
 ***************************************************************************/

#ifdef _OFFLOAD_GPU

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "srradint.h"
#include "srtrjdat.h"
#include "auxgpu.h"

//*************************************************************************
// Status codes per observation point
#define SRRADINT_GPU_ST_CONVERGED   0
#define SRRADINT_GPU_ST_NEED_PASS_B 1
#define SRRADINT_GPU_ST_CPU         2

struct SRadIntAuto1GPUArgs
{
	// observation mesh (effective, i.e. after symmetry reduction)
	double xStart, xStep; int nx;
	double zStart, zStep; int nz;
	double eStart, eStep; int ne;
	double yObs;
	long long nPts;

	// integration interval and precision
	double sStart, sEnd;
	double relPrec;
	double absPrec;     // used in pass B
	int pass;           // 1 or 2
	int maxLevel;       // deepest level available on the uploaded grid
	long long nGrid;    // 4*2^maxLevel + 1

	// trajectory on the finest uniform grid [sStart, sEnd]
	const double *Btx, *X, *IntBtxE2, *Btz, *Z, *IntBtzE2;

	// physics constants
	double GmEm2;             // TrjDatPtr->EbmDat.GammaEm2
	double BetaNormConst;     // TrjDatPtr->BetaNormConst
	double NormalizingConst;
	double PIm10e6, PIm10e6dEnCon;
	int TreatLambdaAsEnergyIn_eV;

	// endpoint trajectory/field data for the residual terms: [0]=left(sStart), [1]=right(sEnd)
	// from CompTrjDataAndFieldWithDerAtPoint('x'/'z', s, ...)
	double end_dBzds[2], end_Bz[2], end_Btx[2], end_Crdx[2], end_IntBtE2x[2];
	double end_dBxds[2], end_Bx[2], end_Btz[2], end_Crdz[2], end_IntBtE2z[2];

	// outputs
	float *outEx, *outEz;      // interleaved re,im per point
	double *sqNorm;
	unsigned char *status;
};

//*************************************************************************
// Bit-compatible replica of srTRadInt::CosAndSin (polynomial after range reduction)

__device__ __forceinline__ void RadIntCosAndSin_GPU(double x, double& Cos, double& Sin)
{
	const double locPI = 3.141592653590;
	const double locTwoPI = 2.*locPI;
	const double locThreePIdTwo = 3.*locPI/2.; // matches ThreePIdTwo = 1.5*PI set in Initialize()
	const double locHalfPI = 0.5*locPI;
	const double locOne_dTwoPI = 1./locTwoPI;
	const double a2c = -0.5, a4c = 1./24., a6c = -1./720., a8c = 1./40320., a10c = -1./3628800.;
	const double a3s = -1./6., a5s = 1./120., a7s = -1./5040., a9s = 1./362880., a11s = -1./39916800.;

	x -= locTwoPI*int(x*locOne_dTwoPI);
	if(x < 0.) x += locTwoPI;

	char ChangeSign = 0;
	if(x > locThreePIdTwo) x -= locTwoPI;
	else if(x > locHalfPI) { x -= locPI; ChangeSign = 1;}

	double xe2 = x*x;
	Cos = 1. + xe2*(a2c + xe2*(a4c + xe2*(a6c + xe2*(a8c + xe2*a10c))));
	Sin = x*(1. + xe2*(a3s + xe2*(a5s + xe2*(a7s + xe2*(a9s + xe2*a11s)))));
	if(ChangeSign) { Cos = -Cos; Sin = -Sin;}
}

//*************************************************************************
// Residual (terminating term) at one integration limit, mirroring
// srTRadInt::ComputeNormalResidual for CoordPres, NumberOfTerms=3,
// ComputeNormalDerivative=0. Returns false if the far-field expansion cannot
// be applied (caller must flag the point for CPU computation).
// side: 0 = left (ComputeDer=1), 1 = right (ComputeDer=2)

__device__ bool RadIntResidualOneSide_GPU(const SRadIntAuto1GPUArgs& p, int side,
	double s, double xObs, double zObs, double ActNormConst, double PIm10e9_d_Lamb,
	double& ResXRe, double& ResXIm, double& ResZRe, double& ResZIm,
	double& DerXRe, double& DerXIm, double& DerZRe, double& DerZIm)
{
	double dBzds = p.end_dBzds[side], Bz = p.end_Bz[side], Btx = p.end_Btx[side], Crdx = p.end_Crdx[side], IntBtE2x = p.end_IntBtE2x[side];
	double dBxds = p.end_dBxds[side], Bx = p.end_Bx[side], Btz = p.end_Btz[side], Crdz = p.end_Crdz[side], IntBtE2z = p.end_IntBtE2z[side];

	double One_d_ymis = 1./(p.yObs - s);
	double xObs_mi_x = xObs - Crdx, zObs_mi_z = zObs - Crdz;

	double LongTerm = IntBtE2x + IntBtE2z;
	double a0 = LongTerm + (xObs_mi_x*xObs_mi_x + zObs_mi_z*zObs_mi_z)*One_d_ymis;
	double Ph = PIm10e9_d_Lamb*(s*p.GmEm2 + a0);

	double CosPh, SinPh;
	RadIntCosAndSin_GPU(Ph, CosPh, SinPh);

	double Nx = xObs_mi_x*One_d_ymis, Btx_mi_Nx = Btx - Nx;
	double Nz = zObs_mi_z*One_d_ymis, Btz_mi_Nz = Btz - Nz;
	double dPhds = PIm10e9_d_Lamb*(p.GmEm2 + Btx_mi_Nx*Btx_mi_Nx + Btz_mi_Nz*Btz_mi_Nz);
	double One_d_dPhds = 1./dPhds;

	double Ax = Btx_mi_Nx*One_d_ymis;
	double Az = Btz_mi_Nz*One_d_ymis;

	// PreExp accumulated as complex; keep 2-term snapshot as CPU does
	double PreExpXRe = Ax, PreExpXIm = 0., PreExpZRe = Az, PreExpZIm = 0.;
	double t1xd = Ax;

	// 2nd term
	double ConBtxBz = p.BetaNormConst*Bz;
	double ConBtzBx = (-p.BetaNormConst)*Bx;
	double ConBtxBzpAx = ConBtxBz + Ax;
	double ConBtzBxpAz = ConBtzBx + Az;
	double TwoPIm10e9_d_Lamb = 2.*PIm10e9_d_Lamb;

	double d2Phds2 = TwoPIm10e9_d_Lamb*(Btx_mi_Nx*ConBtxBzpAx + Btz_mi_Nz*ConBtzBxpAz);
	double dAxds = (2.*Ax + ConBtxBz)*One_d_ymis;
	double dAzds = (2.*Az + ConBtzBx)*One_d_ymis;

	double d2Phds2_d_dPhds = d2Phds2*One_d_dPhds;
	double t2xd = (dAxds - Ax*d2Phds2_d_dPhds)*One_d_dPhds;
	double t2zd = (dAzds - Az*d2Phds2_d_dPhds)*One_d_dPhds;

	PreExpXIm += t2xd; PreExpZIm += t2zd;
	double PreExpX02Re = PreExpXRe, PreExpX02Im = PreExpXIm;
	double PreExpZ02Re = PreExpZRe, PreExpZ02Im = PreExpZIm;

	// 3rd term
	double ConBtxdBzds = p.BetaNormConst*dBzds;
	double ConBtzdBxds = (-p.BetaNormConst)*dBxds;

	double d3Phds3 = TwoPIm10e9_d_Lamb*(ConBtxBzpAx*ConBtxBzpAx + ConBtzBxpAz*ConBtzBxpAz
		+ Btx_mi_Nx*(One_d_ymis*(2.*Btx_mi_Nx*One_d_ymis + ConBtxBz) + ConBtxdBzds)
		+ Btz_mi_Nz*(One_d_ymis*(2.*Btz_mi_Nz*One_d_ymis + ConBtzBx) + ConBtzdBxds));

	double d2Axds2 = One_d_ymis*(3.*dAxds + ConBtxdBzds);
	double d2Azds2 = One_d_ymis*(3.*dAzds + ConBtzdBxds);

	double One_d_dPhdsE2 = One_d_dPhds*One_d_dPhds;
	double d2Phds2_mu_d2Phds2_d_dPhdsE2 = d2Phds2*d2Phds2*One_d_dPhdsE2;

	double t3xd = (-d2Axds2 + (3.*dAxds*d2Phds2 + Ax*d3Phds3)*One_d_dPhds - 3.*Ax*d2Phds2_mu_d2Phds2_d_dPhdsE2)*One_d_dPhdsE2;
	double t3zd = (-d2Azds2 + (3.*dAzds*d2Phds2 + Az*d3Phds3)*One_d_dPhds - 3.*Az*d2Phds2_mu_d2Phds2_d_dPhdsE2)*One_d_dPhdsE2;

	PreExpXRe += t3xd; PreExpZRe += t3zd;

	// expansion applicability checks (x components only, as on CPU)
	bool TwoTermsOK = ((t2xd == 0.) && (t1xd == 0.)) || ((t2xd != 0.) && (fabs(t2xd) < 0.7*fabs(t1xd)));
	bool ThreeTermsOK = TwoTermsOK && (((t3xd == 0.) && (t2xd == 0.)) || ((t3xd != 0.) && (fabs(t3xd) < 0.7*fabs(t2xd))));

	if(!ThreeTermsOK)
	{
		if(!TwoTermsOK) return false; // far-field expansion does not work -> CPU
		PreExpXRe = PreExpX02Re; PreExpXIm = PreExpX02Im;
		PreExpZRe = PreExpZ02Re; PreExpZIm = PreExpZ02Im;
	}

	// PreExp *= i/dPhds ; Resid = ActNormConst*PreExp*exp(i*Ph)
	double bRe, bIm;
	bRe = -PreExpXIm*One_d_dPhds; bIm = PreExpXRe*One_d_dPhds;
	ResXRe = ActNormConst*(bRe*CosPh - bIm*SinPh);
	ResXIm = ActNormConst*(bRe*SinPh + bIm*CosPh);
	bRe = -PreExpZIm*One_d_dPhds; bIm = PreExpZRe*One_d_dPhds;
	ResZRe = ActNormConst*(bRe*CosPh - bIm*SinPh);
	ResZIm = ActNormConst*(bRe*SinPh + bIm*CosPh);

	// edge derivative of the integrand (InitDerMan/FinDerMan on CPU):
	// (dAxds + i*Ax*dPhds) * exp(i*Ph)
	DerXRe = dAxds*CosPh - Ax*dPhds*SinPh;
	DerXIm = dAxds*SinPh + Ax*dPhds*CosPh;
	DerZRe = dAzds*CosPh - Az*dPhds*SinPh;
	DerZIm = dAzds*SinPh + Az*dPhds*CosPh;
	return true;
}

//*************************************************************************

__global__ void RadIntAuto1Kernel(SRadIntAuto1GPUArgs p)
{
	long long ipt = (long long)blockIdx.x*blockDim.x + threadIdx.x;
	if(ipt >= p.nPts) return;
	if(p.pass == 2)
	{
		if(p.status[ipt] != SRRADINT_GPU_ST_NEED_PASS_B) return;
	}

	// decode (iz, ix, ie); linear index = (iz*nx + ix)*ne + ie
	int ie = (int)(ipt % p.ne);
	long long izx = ipt/p.ne;
	int ix = (int)(izx % p.nx);
	int iz = (int)(izx/p.nx);

	double xObs = p.xStart + ix*p.xStep;
	double zObs = p.zStart + iz*p.zStep;
	double Lamb = p.eStart + ie*p.eStep;
	double yObs = p.yObs;

	double ActNormConst = (p.TreatLambdaAsEnergyIn_eV)? p.NormalizingConst*Lamb*0.80654658E-03 : p.NormalizingConst/Lamb;
	double PIm10e9_d_Lamb = (p.TreatLambdaAsEnergyIn_eV)? p.PIm10e6dEnCon*Lamb : p.PIm10e6*1000./Lamb;

	const double locPI = 3.141592653590;
	const double wfe = 7./15., wf1 = 16./15., wf2 = 14./15., wd = 1./15.;

	double sStart = p.sStart, sEnd = p.sEnd;
	double GmEm2 = p.GmEm2;

	// ---- residual (terminating) terms and edge derivatives ----
	double L_ResXRe, L_ResXIm, L_ResZRe, L_ResZIm, L_DerXRe, L_DerXIm, L_DerZRe, L_DerZIm;
	double R_ResXRe, R_ResXIm, R_ResZRe, R_ResZIm, R_DerXRe, R_DerXIm, R_DerZRe, R_DerZIm;

	if(!RadIntResidualOneSide_GPU(p, 0, sStart, xObs, zObs, ActNormConst, PIm10e9_d_Lamb,
		L_ResXRe, L_ResXIm, L_ResZRe, L_ResZIm, L_DerXRe, L_DerXIm, L_DerZRe, L_DerZIm))
	{
		p.status[ipt] = SRRADINT_GPU_ST_CPU; p.sqNorm[ipt] = 0.; return;
	}
	if(!RadIntResidualOneSide_GPU(p, 1, sEnd, xObs, zObs, ActNormConst, PIm10e9_d_Lamb,
		R_ResXRe, R_ResXIm, R_ResZRe, R_ResZIm, R_DerXRe, R_DerXIm, R_DerZRe, R_DerZIm))
	{
		p.status[ipt] = SRRADINT_GPU_ST_CPU; p.sqNorm[ipt] = 0.; return;
	}

	double OutIntXRe = R_ResXRe - L_ResXRe, OutIntXIm = R_ResXIm - L_ResXIm;
	double OutIntZRe = R_ResZRe - L_ResZRe, OutIntZIm = R_ResZIm - L_ResZIm;

	// DifDer = InitDer - FinDer (left - right)
	double wDifDerXRe = wd*(L_DerXRe - R_DerXRe), wDifDerXIm = wd*(L_DerXIm - R_DerXIm);
	double wDifDerZRe = wd*(L_DerZRe - R_DerZRe), wDifDerZIm = wd*(L_DerZIm - R_DerZIm);

	// ---- level 0: 5 points across [sStart, sEnd] ----
	const double *pBtx = p.Btx, *pX = p.X, *pIntBtxE2 = p.IntBtxE2;
	const double *pBtz = p.Btz, *pZ = p.Z, *pIntBtzE2 = p.IntBtzE2;

	long long G = p.nGrid - 1;            // = 4*2^maxLevel
	long long stride0 = G >> 2;           // level-0 spacing in grid indices

	double sStep = (sEnd - sStart)*0.25;
	double Ax, Az, Ph, CosPh, SinPh, PhPrev, PhInit;
	double One_d_ymis, xObs_mi_x, zObs_mi_z, Nx, Nz, LongTerm, a0;

	double Sum1XRe=0., Sum1XIm=0., Sum1ZRe=0., Sum1ZIm=0., Sum2XRe=0., Sum2XIm=0., Sum2ZRe=0., Sum2ZIm=0.;
	double wFxRe, wFxIm, wFzRe, wFzIm;

	// helper macro: near-field integrand at grid index g and longitudinal position sv
#define SRRADINT_EVAL(g, sv) \
	{ \
		One_d_ymis = 1./(yObs - (sv)); \
		xObs_mi_x = xObs - pX[(g)]; zObs_mi_z = zObs - pZ[(g)]; \
		Nx = xObs_mi_x*One_d_ymis; Nz = zObs_mi_z*One_d_ymis; \
		LongTerm = pIntBtxE2[(g)] + pIntBtzE2[(g)]; \
		a0 = LongTerm + xObs_mi_x*Nx + zObs_mi_z*Nz; \
		Ph = PIm10e9_d_Lamb*((sv)*GmEm2 + a0); \
		Ax = (pBtx[(g)] - Nx)*One_d_ymis; Az = (pBtz[(g)] - Nz)*One_d_ymis; \
		RadIntCosAndSin_GPU(Ph, CosPh, SinPh); \
	}

	double s = sStart;
	SRRADINT_EVAL(0, s);
	wFxRe = Ax*CosPh; wFxIm = Ax*SinPh; wFzRe = Az*CosPh; wFzIm = Az*SinPh;
	PhInit = Ph;

	s = sStart + sStep;
	SRRADINT_EVAL(stride0, s);
	Sum1XRe += Ax*CosPh; Sum1XIm += Ax*SinPh; Sum1ZRe += Az*CosPh; Sum1ZIm += Az*SinPh; s += sStep;

	SRRADINT_EVAL(2*stride0, s);
	Sum2XRe += Ax*CosPh; Sum2XIm += Ax*SinPh; Sum2ZRe += Az*CosPh; Sum2ZIm += Az*SinPh; s += sStep;

	SRRADINT_EVAL(3*stride0, s);
	Sum1XRe += Ax*CosPh; Sum1XIm += Ax*SinPh; Sum1ZRe += Az*CosPh; Sum1ZIm += Az*SinPh; s += sStep;

	SRRADINT_EVAL(4*stride0, s);
	wFxRe += Ax*CosPh; wFxIm += Ax*SinPh; wFzRe += Az*CosPh; wFzIm += Az*SinPh;
	wFxRe *= wfe; wFxIm *= wfe; wFzRe *= wfe; wFzIm *= wfe;

	double ActNormConst_sStep = ActNormConst*sStep;
	double IntXRe = OutIntXRe + ActNormConst_sStep*(wFxRe + wf1*Sum1XRe + wf2*Sum2XRe + sStep*wDifDerXRe);
	double IntXIm = OutIntXIm + ActNormConst_sStep*(wFxIm + wf1*Sum1XIm + wf2*Sum2XIm + sStep*wDifDerXIm);
	double IntZRe = OutIntZRe + ActNormConst_sStep*(wFzRe + wf1*Sum1ZRe + wf2*Sum2ZRe + sStep*wDifDerZRe);
	double IntZIm = OutIntZIm + ActNormConst_sStep*(wFzIm + wf1*Sum1ZIm + wf2*Sum2ZIm + sStep*wDifDerZIm);
	double SqNorm = IntXRe*IntXRe + IntXIm*IntXIm + IntZRe*IntZRe + IntZIm*IntZIm;

	// ---- adaptive level doubling ----
	long long NpOnLevel = 4;
	int LevelNo = 0;
	char NotFinishedYet = 1;
	unsigned char st = SRRADINT_GPU_ST_CONVERGED;

	while(NotFinishedYet)
	{
		Sum2XRe += Sum1XRe; Sum2XIm += Sum1XIm; Sum2ZRe += Sum1ZRe; Sum2ZIm += Sum1ZIm;
		Sum1XRe = Sum1XIm = Sum1ZRe = Sum1ZIm = 0.;
		char ThisMayBeTheLastLoop = 1;
		PhPrev = PhInit;
		LevelNo++;

		if(LevelNo > p.maxLevel)
		{
			st = (p.pass == 1)? SRRADINT_GPU_ST_NEED_PASS_B : SRRADINT_GPU_ST_CPU;
			break;
		}

		double HalfStep = 0.5*sStep;
		s = sStart + HalfStep;

		long long gStride = 1LL << (p.maxLevel - LevelNo); // grid index of point i on this level: (2i+1)*gStride
		long long g = gStride;
		long long gInc = 2*gStride;

		for(long long i=0; i<NpOnLevel; i++)
		{
			SRRADINT_EVAL(g, s);
			Sum1XRe += Ax*CosPh; Sum1XIm += Ax*SinPh; Sum1ZRe += Az*CosPh; Sum1ZIm += Az*SinPh;
			s += sStep; g += gInc;

			if(Ph - PhPrev > locPI) ThisMayBeTheLastLoop = 0;
			PhPrev = Ph;
		}

		double ActNormConstHalfStep = ActNormConst*HalfStep;
		double LocIntXRe = OutIntXRe + ActNormConstHalfStep*(wFxRe + wf1*Sum1XRe + wf2*Sum2XRe + HalfStep*wDifDerXRe);
		double LocIntXIm = OutIntXIm + ActNormConstHalfStep*(wFxIm + wf1*Sum1XIm + wf2*Sum2XIm + HalfStep*wDifDerXIm);
		double LocIntZRe = OutIntZRe + ActNormConstHalfStep*(wFzRe + wf1*Sum1ZRe + wf2*Sum2ZRe + HalfStep*wDifDerZRe);
		double LocIntZIm = OutIntZIm + ActNormConstHalfStep*(wFzIm + wf1*Sum1ZIm + wf2*Sum2ZIm + HalfStep*wDifDerZIm);
		double LocSqNorm = LocIntXRe*LocIntXRe + LocIntXIm*LocIntXIm + LocIntZRe*LocIntZRe + LocIntZIm*LocIntZIm;

		if(ThisMayBeTheLastLoop)
		{
			double TestVal = fabs(LocSqNorm - SqNorm);
			char NotFinishedYetFirstTest;
			if(p.pass == 2) NotFinishedYetFirstTest = (TestVal > p.absPrec);
			else NotFinishedYetFirstTest = (TestVal > p.relPrec*LocSqNorm);
			if(!NotFinishedYetFirstTest) NotFinishedYet = 0;
		}

		IntXRe = LocIntXRe; IntXIm = LocIntXIm; IntZRe = LocIntZRe; IntZIm = LocIntZIm;
		SqNorm = LocSqNorm;
		sStep = HalfStep; NpOnLevel *= 2;
	}
#undef SRRADINT_EVAL

	p.status[ipt] = st;
	p.sqNorm[ipt] = SqNorm;
	p.outEx[2*ipt] = (float)IntXRe; p.outEx[2*ipt+1] = (float)IntXIm;
	p.outEz[2*ipt] = (float)IntZRe; p.outEz[2*ipt+1] = (float)IntZIm;
}

//*************************************************************************
// Host-side driver. Returns 0 if the whole mesh was computed (possibly with a
// per-point CPU fallback for flagged points), -1 if the configuration is not
// supported (caller falls through to the unchanged CPU loop), >0 on error.

int srTRadInt::ComputeTotalRadDistrDirectOutGPU(srTSRWRadStructAccessData& SRWRadStructAccessData, void* pvGPU, char FinalResAreSymOverX, char FinalResAreSymOverZ)
{
	const bool verb = (getenv("SRW_RADINT_GPU_VERBOSE") != 0);
#define SRRADINT_GPU_REFUSE(why) { if(verb) { fprintf(stderr, "srradint_gpu: refusing: %s\n", why); fflush(stderr); } return -1; }

	TGPUUsageArg gpuArg(pvGPU);
	if(!CAuxGPU::GPUEnabled(&gpuArg)) SRRADINT_GPU_REFUSE("GPU not enabled");

	// supported configuration (narrow, verified slice)
	if(sIntegMethod != 10) SRRADINT_GPU_REFUSE("sIntegMethod != 10");            // Auto1 only
	if(DistrInfoDat.CoordOrAngPresentation != CoordPres) SRRADINT_GPU_REFUSE("not CoordPres"); // near-field presentation only
	if(ComputeNormalDerivative) SRRADINT_GPU_REFUSE("ComputeNormalDerivative");
	if(m_CalcResidTerminTerms != 1) SRRADINT_GPU_REFUSE("m_CalcResidTerminTerms != 1"); // both terminating terms (the default)
	if(DistrInfoDat.ShowPhaseOnly) SRRADINT_GPU_REFUSE("ShowPhaseOnly");
	if(TrjDatPtr == 0) SRRADINT_GPU_REFUSE("no TrjDat");

	const int maxLev = 14;               // level cap on GPU: 4*2^14+1 = 65537 trajectory points
	const long long nGrid = (4LL << maxLev) + 1;

	double StepLambda = (DistrInfoDat.nLamb > 1)? (DistrInfoDat.LambEnd - DistrInfoDat.LambStart)/(DistrInfoDat.nLamb - 1) : 0.;
	double StepX = (DistrInfoDat.nx > 1)? (DistrInfoDat.xEnd - DistrInfoDat.xStart)/(DistrInfoDat.nx - 1) : 0.;
	double StepZ = (DistrInfoDat.nz > 1)? (DistrInfoDat.zEnd - DistrInfoDat.zStart)/(DistrInfoDat.nz - 1) : 0.;

	// effective (symmetry-reduced) loop bounds, mirroring the CPU loop breaks
	double xc = TrjDatPtr->EbmDat.x0, zc = TrjDatPtr->EbmDat.z0;
	double xTol = StepX*0.001, zTol = StepZ*0.001;

	int nzEff = DistrInfoDat.nz, nxEff = DistrInfoDat.nx;
	if(FinalResAreSymOverZ)
	{
		double z = DistrInfoDat.zStart;
		for(int iz=0; iz<DistrInfoDat.nz; iz++) { if((z - zc) > zTol) { nzEff = iz; break;} z += StepZ; }
	}
	if(FinalResAreSymOverX)
	{
		double x = DistrInfoDat.xStart;
		for(int ix=0; ix<DistrInfoDat.nx; ix++) { if((x - xc) > xTol) { nxEff = ix; break;} x += StepX; }
	}

	long long nPts = ((long long)nzEff)*nxEff*DistrInfoDat.nLamb;
	if(nPts < 128) SRRADINT_GPU_REFUSE("mesh too small"); // not worth a GPU round-trip
#undef SRRADINT_GPU_REFUSE

	// ---- host trajectory grid (pointwise from the interpolating structure) ----
	double *hTraj = new double[nGrid*8];
	if(hTraj == 0) return MEMORY_ALLOCATION_FAILURE;
	double *hBtx = hTraj, *hBtz = hTraj + nGrid, *hX = hTraj + 2*nGrid, *hZ = hTraj + 3*nGrid;
	double *hIntBtxE2 = hTraj + 4*nGrid, *hIntBtzE2 = hTraj + 5*nGrid, *hBx = hTraj + 6*nGrid, *hBz = hTraj + 7*nGrid;
	TrjDatPtr->CompTotalTrjData(sIntegStart, sIntegFin, nGrid, hBtx, hBtz, hX, hZ, hIntBtxE2, hIntBtzE2, hBx, hBz);

	// ---- kernel arguments ----
	SRadIntAuto1GPUArgs p;
	p.xStart = DistrInfoDat.xStart; p.xStep = StepX; p.nx = nxEff;
	p.zStart = DistrInfoDat.zStart; p.zStep = StepZ; p.nz = nzEff;
	p.eStart = DistrInfoDat.LambStart; p.eStep = StepLambda; p.ne = DistrInfoDat.nLamb;
	p.yObs = DistrInfoDat.yStart;
	p.nPts = nPts;
	p.sStart = sIntegStart; p.sEnd = sIntegFin;
	p.relPrec = sIntegRelPrec;
	p.absPrec = 0.;
	p.pass = 1;
	p.maxLevel = maxLev;
	p.nGrid = nGrid;
	p.GmEm2 = TrjDatPtr->EbmDat.GammaEm2;
	p.BetaNormConst = TrjDatPtr->BetaNormConst;
	p.NormalizingConst = NormalizingConst;
	p.PIm10e6 = PIm10e6; p.PIm10e6dEnCon = PIm10e6dEnCon;
	p.TreatLambdaAsEnergyIn_eV = (DistrInfoDat.TreatLambdaAsEnergyIn_eV)? 1 : 0;

	// endpoint data for the residual terms
	double sEndArr[2]; sEndArr[0] = sIntegStart; sEndArr[1] = sIntegFin;
	for(int side=0; side<2; side++)
	{
		double dBzds=0., Bz=0., Btx=0., Crdx=0., IntBtE2x=0.;
		double dBxds=0., Bx=0., Btz=0., Crdz=0., IntBtE2z=0.;
		TrjDatPtr->CompTrjDataAndFieldWithDerAtPoint('x', sEndArr[side], dBzds, Bz, Btx, Crdx, IntBtE2x);
		TrjDatPtr->CompTrjDataAndFieldWithDerAtPoint('z', sEndArr[side], dBxds, Bx, Btz, Crdz, IntBtE2z);
		p.end_dBzds[side] = dBzds; p.end_Bz[side] = Bz; p.end_Btx[side] = Btx; p.end_Crdx[side] = Crdx; p.end_IntBtE2x[side] = IntBtE2x;
		p.end_dBxds[side] = dBxds; p.end_Bx[side] = Bx; p.end_Btz[side] = Btz; p.end_Crdz[side] = Crdz; p.end_IntBtE2z[side] = IntBtE2z;
	}

	// ---- device buffers ----
	double *dTraj = 0; float *dEx = 0, *dEz = 0; double *dSqNorm = 0; unsigned char *dStatus = 0;
	cudaError_t cuErr = cudaSuccess;
	int result = 0;
	float *hEx = 0, *hEz = 0; double *hSqNorm = 0; unsigned char *hStatus = 0;

	do {
		if((cuErr = cudaMalloc(&dTraj, sizeof(double)*nGrid*6)) != cudaSuccess) break;
		if((cuErr = cudaMalloc(&dEx, sizeof(float)*2*nPts)) != cudaSuccess) break;
		if((cuErr = cudaMalloc(&dEz, sizeof(float)*2*nPts)) != cudaSuccess) break;
		if((cuErr = cudaMalloc(&dSqNorm, sizeof(double)*nPts)) != cudaSuccess) break;
		if((cuErr = cudaMalloc(&dStatus, sizeof(unsigned char)*nPts)) != cudaSuccess) break;

		// upload the 6 arrays the kernel needs, in kernel order
		if((cuErr = cudaMemcpy(dTraj,           hBtx,      sizeof(double)*nGrid, cudaMemcpyHostToDevice)) != cudaSuccess) break;
		if((cuErr = cudaMemcpy(dTraj +   nGrid, hX,        sizeof(double)*nGrid, cudaMemcpyHostToDevice)) != cudaSuccess) break;
		if((cuErr = cudaMemcpy(dTraj + 2*nGrid, hIntBtxE2, sizeof(double)*nGrid, cudaMemcpyHostToDevice)) != cudaSuccess) break;
		if((cuErr = cudaMemcpy(dTraj + 3*nGrid, hBtz,      sizeof(double)*nGrid, cudaMemcpyHostToDevice)) != cudaSuccess) break;
		if((cuErr = cudaMemcpy(dTraj + 4*nGrid, hZ,        sizeof(double)*nGrid, cudaMemcpyHostToDevice)) != cudaSuccess) break;
		if((cuErr = cudaMemcpy(dTraj + 5*nGrid, hIntBtzE2, sizeof(double)*nGrid, cudaMemcpyHostToDevice)) != cudaSuccess) break;

		p.Btx = dTraj; p.X = dTraj + nGrid; p.IntBtxE2 = dTraj + 2*nGrid;
		p.Btz = dTraj + 3*nGrid; p.Z = dTraj + 4*nGrid; p.IntBtzE2 = dTraj + 5*nGrid;
		p.outEx = dEx; p.outEz = dEz; p.sqNorm = dSqNorm; p.status = dStatus;

		int threads = 128;
		long long blocks = (nPts + threads - 1)/threads;

		// ---- pass A: per-point relative convergence ----
		RadIntAuto1Kernel<<<(unsigned int)blocks, threads>>>(p);
		if((cuErr = cudaGetLastError()) != cudaSuccess) break;
		if((cuErr = cudaDeviceSynchronize()) != cudaSuccess) break;

		hSqNorm = new double[nPts];
		hStatus = new unsigned char[nPts];
		if((cuErr = cudaMemcpy(hSqNorm, dSqNorm, sizeof(double)*nPts, cudaMemcpyDeviceToHost)) != cudaSuccess) break;
		if((cuErr = cudaMemcpy(hStatus, dStatus, sizeof(unsigned char)*nPts, cudaMemcpyDeviceToHost)) != cudaSuccess) break;

		double MaxEst = 0.;
		long long nNeedB = 0;
		for(long long i=0; i<nPts; i++)
		{
			if(hStatus[i] != SRRADINT_GPU_ST_CPU) { if(hSqNorm[i] > MaxEst) MaxEst = hSqNorm[i]; }
			if(hStatus[i] == SRRADINT_GPU_ST_NEED_PASS_B) nNeedB++;
		}

		// ---- pass B: absolute criterion from the global maximum ----
		if((nNeedB > 0) && (MaxEst > 0.))
		{
			p.pass = 2;
			p.absPrec = sIntegRelPrec*MaxEst;
			RadIntAuto1Kernel<<<(unsigned int)blocks, threads>>>(p);
			if((cuErr = cudaGetLastError()) != cudaSuccess) break;
			if((cuErr = cudaDeviceSynchronize()) != cudaSuccess) break;
			if((cuErr = cudaMemcpy(hStatus, dStatus, sizeof(unsigned char)*nPts, cudaMemcpyDeviceToHost)) != cudaSuccess) break;
			if((cuErr = cudaMemcpy(hSqNorm, dSqNorm, sizeof(double)*nPts, cudaMemcpyDeviceToHost)) != cudaSuccess) break;
			for(long long i=0; i<nPts; i++)
				if((hStatus[i] != SRRADINT_GPU_ST_CPU) && (hSqNorm[i] > MaxEst)) MaxEst = hSqNorm[i];
		}
		else if(nNeedB > 0)
		{
			for(long long i=0; i<nPts; i++)
				if(hStatus[i] == SRRADINT_GPU_ST_NEED_PASS_B) hStatus[i] = SRRADINT_GPU_ST_CPU;
		}

		hEx = new float[2*nPts];
		hEz = new float[2*nPts];
		if((cuErr = cudaMemcpy(hEx, dEx, sizeof(float)*2*nPts, cudaMemcpyDeviceToHost)) != cudaSuccess) break;
		if((cuErr = cudaMemcpy(hEz, dEz, sizeof(float)*2*nPts, cudaMemcpyDeviceToHost)) != cudaSuccess) break;

		// ---- scatter into the wavefront and CPU-recompute flagged points ----
		long long PerX = DistrInfoDat.nLamb << 1;
		long long PerZ = DistrInfoDat.nx*PerX;
		float *pEx0 = SRWRadStructAccessData.pBaseRadX;
		float *pEz0 = SRWRadStructAccessData.pBaseRadZ;

		// seed the (order-carried) CPU tolerance state with the global maximum,
		// so per-point CPU fallbacks see the same tolerance the GPU used
		MaxFluxDensVal = MaxEst;
		CurrentAbsPrec = sIntegRelPrec*MaxEst;
		ProbablyTheSameLoop = 1;

		long long nCPUFallback = 0;
		long long i = 0;
		ObsCoor.y = DistrInfoDat.yStart;
		for(int iz=0; iz<nzEff; iz++)
		{
			long long izPerZ = iz*PerZ;
			for(int ix=0; ix<nxEff; ix++)
			{
				long long ixPerX = ix*PerX;
				for(int iLamb=0; iLamb<DistrInfoDat.nLamb; iLamb++)
				{
					long long Offset = izPerZ + ixPerX + (iLamb << 1);
					float *pEx = pEx0 + Offset, *pEz = pEz0 + Offset;
					if(hStatus[i] == SRRADINT_GPU_ST_CPU)
					{
						nCPUFallback++;
						ObsCoor.z = DistrInfoDat.zStart + iz*StepZ;
						ObsCoor.x = DistrInfoDat.xStart + ix*StepX;
						ObsCoor.Lamb = DistrInfoDat.LambStart + iLamb*StepLambda;
						complex<double> RadIntegValues[2];
						srTEFourier EwNormDer;
						if(result = GenRadIntegration(RadIntegValues, &EwNormDer)) break;
						*pEx = (float)RadIntegValues->real();
						*(pEx+1) = (float)RadIntegValues->imag();
						*pEz = (float)RadIntegValues[1].real();
						*(pEz+1) = (float)RadIntegValues[1].imag();
					}
					else
					{
						*pEx = hEx[2*i]; *(pEx+1) = hEx[2*i+1];
						*pEz = hEz[2*i]; *(pEz+1) = hEz[2*i+1];
					}
					i++;
				}
				if(result) break;
			}
			if(result) break;
		}

		if((result == 0) && (FinalResAreSymOverZ || FinalResAreSymOverX))
			FillInSymPartsOfResults(FinalResAreSymOverX, FinalResAreSymOverZ, SRWRadStructAccessData);

		if(getenv("SRW_RADINT_GPU_VERBOSE") != 0)
		{
			long long nB = 0;
			for(long long j=0; j<nPts; j++) if(hStatus[j] == SRRADINT_GPU_ST_NEED_PASS_B) nB++;
			fprintf(stderr, "srradint_gpu: nPts=%lld passB=%lld (still flagged: %lld) cpuFallback=%lld MaxEst=%.6e\n",
				nPts, nNeedB, nB, nCPUFallback, MaxEst);
			fflush(stderr);
		}
	} while(0);

	if(cuErr != cudaSuccess)
	{
		fprintf(stderr, "srradint_gpu: CUDA error %d - %s; falling back to CPU\n", (int)cuErr, cudaGetErrorString(cuErr));
		fflush(stderr);
		result = -1; // let the CPU loop compute everything
	}

	if(dTraj) cudaFree(dTraj);
	if(dEx) cudaFree(dEx);
	if(dEz) cudaFree(dEz);
	if(dSqNorm) cudaFree(dSqNorm);
	if(dStatus) cudaFree(dStatus);
	if(hEx) delete[] hEx;
	if(hEz) delete[] hEz;
	if(hSqNorm) delete[] hSqNorm;
	if(hStatus) delete[] hStatus;
	delete[] hTraj;

	return result;
}

#endif //_OFFLOAD_GPU
