#include "Vec2D.h"
#include "params.h"
#include "config_main.h"

void setparams(Params & mypars)
{
	ConfigFile prms("conf.param");

	mypars.sqrtDens = prms.getValueOfKey4;
	mypars.cellNx = 5;
	mypars.cellNy = 5; // keep in mind we want square cells
	mypars.AR = 1;
	mypars.length = 3.0;
	mypars.sweepEval = 50;
	mypars.sweepLimit = 100000;
	mypars.nProc = 1e5;
	mypars.sweepEvalProc = 1e3;
	mypars.maxAttempts = 1;
	mypars.boxEdge = 20.0;
	mypars.transFactor = 0.04;
	mypars.angMag = 0.04; // fraction of pi
	mypars.onefile = false;

	// box length defined as unity
	if (mypars.cellNx > mypars.cellNy) {
		int tmp = mypars.cellNx;
		mypars.cellNx = mypars.cellNy;
		mypars.cellNy = tmp;
	}

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