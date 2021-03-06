#include "Vec2D.h"
#include "params.h"

void setparams(Params & mypars)
{
	// box length defined as unity
	if (mypars.cellNx > mypars.cellNy) {
		int tmp = mypars.cellNx;
		mypars.cellNx = mypars.cellNy;
		mypars.cellNy = tmp;
	}

	mypars.sqrtDens = 4;
	mypars.cellNx = 5;
	mypars.cellNy = 5; // keep in mind we want square cells
	mypars.AR = 1;
	mypars.length = 3.0;
	mypars.crossEval = 50;
	mypars.crossSweepLim = 1000000;
	mypars.nProc = 1e6;
	mypars.nEquil = 1e5;
	mypars.sweepEval = 1e3;
	//mypars.sweepEval = mypars.nProc/10;
	mypars.boxEdge = 10.0;
	mypars.transFactor = 0.04;
	mypars.angMag = 0.04; // fraction of pi
	mypars.onefile = true;

	double wallRatio = mypars.cellNx / mypars.cellNy;
	mypars.box.set_values(mypars.boxEdge*wallRatio, mypars.boxEdge);
	mypars.cellWidth = mypars.boxEdge / mypars.cellNy;
	mypars.molWidth = mypars.AR * mypars.length;
	mypars.dr = (double)(mypars.cellWidth / mypars.sqrtDens);
	mypars.Nx = mypars.sqrtDens * mypars.cellNx;
	mypars.Ny = mypars.sqrtDens * mypars.cellNy;
	mypars.nObj = mypars.Nx * mypars.Ny;
	mypars.transMag = mypars.transFactor*mypars.dr;
	mypars.nCell = mypars.cellNx * mypars.cellNy;
	mypars.stepsInSweep = mypars.nObj;
}
