#ifndef PARAMS
#define PARAMS

#include "config_main.h"
#include "Vec2D.h"

struct Params
{
	
	int sqrtDens;
	int cellNx;
	int cellNy;
	int AR;
	double length;
	int sweepEval;
	int sweepLimit;
	int nProc;
	int sweepEvalProc;
	int maxAttempts;
	double boxEdge;
	double transFactor;
	double angMag;
	bool onefile;
	double wallRatio;
	Vec<double> box;
	double cellWidth;
	double molWidth;
	double dr;
	int Nx;
	int Ny;
	int nObj;
	double transMag;
	int nCell;
	int stepsInSweep;

	Params() {
		cout << "OK" << endl;
		ConfigFile prms("conf.param");

		cout << "sqd " << sqrtDens << endl;
		sqrtDens = prms.getValueOfKey<int>("sqrtDens");
		cout << "sqd " << sqrtDens << endl;
		cellNx = prms.getValueOfKey<int>("cellNx");
		cellNy = prms.getValueOfKey<int>("cellNy");
		AR = prms.getValueOfKey<int>("AR");
		length = prms.getValueOfKey<double>("length");
		sweepEval = prms.getValueOfKey<int>("sweepEval");
		sweepLimit = prms.getValueOfKey<int>("sweepLimit");
		nProc = prms.getValueOfKey<int>("nProc");
		sweepEvalProc = prms.getValueOfKey<int>("sweepEvalProc");
		maxAttempts = prms.getValueOfKey<int>("maxAttempts");
		boxEdge = prms.getValueOfKey<double>("boxEdge");
		transFactor = prms.getValueOfKey<double>("transFactor");
		angMag = prms.getValueOfKey<double>("angMag");
		cout << "sqd " << sqrtDens << endl;
		onefile = prms.getValueOfKey<int>("onefile");
		cout << "onefile" << onefile << endl;
		

		// box length defined as unity
		if (cellNx > cellNy) {
			int tmp = cellNx;
			cellNx = cellNy;
			cellNy = tmp;
		};

		wallRatio = cellNx / cellNy;
		box.set_values(boxEdge*wallRatio, boxEdge);
		cellWidth = boxEdge / cellNy;
		molWidth = AR * length;
		dr = (double)(cellWidth / sqrtDens);
		Nx = sqrtDens * cellNx;
		Ny = sqrtDens * cellNy;
		nObj = Nx * Ny;
		transMag = transFactor*dr;
		nCell = cellNx * cellNy;
		stepsInSweep = nObj;
	};



	void printParams() {
		cout << "Nx " << Nx << endl
				 << "Ny " << Ny << endl
				 << "nObj " << nObj << endl
				 << "sqrtDens " << sqrtDens << endl
				 << "dr " << dr  << endl
				 << "cellNx " << cellNx << endl
				 << "cellNy " << cellNy  << endl
				 << "nCell " << nCell << endl
				 << "boxEdge " << boxEdge << endl
				 << "box.x " << box.x << endl
				 << "box.y " << box.y << endl
				 << "cellWidth " << cellWidth << endl
				 << "AR " << AR << endl
				 << "molWidth " << molWidth << endl
				 << "length " << length << endl
				 << "transFactor " << transFactor << endl
				 << "transMag " << transMag  << endl
				 << "sweepEval " << sweepEval << endl
				 << "sweepEvalProc " << sweepEvalProc << endl
				 << "nProc " << nProc << endl
				 << "sweepLimit " << sweepLimit << endl;
	}
};
#endif
