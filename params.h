#ifndef PARAMS
#define PARAMS

#include "config_main.h"
#include "Vec2D.h"

struct Params
{
	
	int cellNx;
	int cellNy;
	int AR;
	double length;
	int sweepEval;
	int sweepLimit;
	int nProc;
	int nEquil;
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

	Params(std::string cnfFile) {
		ConfigFile prms(cnfFile);

		Nx = prms.getValueOfKey<int>("Nx");
		Ny = prms.getValueOfKey<int>("Ny");
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
		onefile = prms.getValueOfKey<int>("onefile");
		

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
		dr = (double) boxEdge/Nx;
		//dr = (double)(cellWidth / sqrtDens);
		nObj = Nx * Ny;
		transMag = transFactor*dr;
		nCell = cellNx * cellNy;
	};



	void printParams() {
		double rho = (double) nObj / (boxEdge*boxEdge);
		double redRho = length*length*rho;
		cout << "Nx " << Nx << endl
				 << "Ny " << Ny << endl
				 << "nObj " << nObj << endl
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
				 << "sweepLimit " << sweepLimit << endl
				 << "sweepEval " << sweepEval << endl
				 << "nEquil " << nEquil << endl
				 << "nProc " << nProc << endl
				 << "sweepEvalProc " << sweepEvalProc << endl
				 << endl
				 << "Rho " << rho << endl
				 << "Reduced rho " << redRho << endl;
	}
};
#endif
