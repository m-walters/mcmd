#ifndef PARAMS
#define PARAMS

#include <iostream>
#include "config_main.h"
#include "Vec2D.h"

struct Params
{
	
	int cellNx;
	int cellNy;
	int AR;
	double length;
	int crossEval;
	int nProc;
	int nEquil;
	int sweepEval;
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
	bool printUncross;
	std::string shape;

	Params(std::string cnfFile) {
		ConfigFile prms(cnfFile);

		Nx = prms.getValueOfKey<int>("Nx");
		Ny = prms.getValueOfKey<int>("Ny");
		cellNx = prms.getValueOfKey<int>("cellNx");
		cellNy = prms.getValueOfKey<int>("cellNy");
		AR = prms.getValueOfKey<int>("AR");
		length = prms.getValueOfKey<double>("length");
		crossEval = prms.getValueOfKey<int>("crossEval");
		nEquil = prms.getValueOfKey<int>("nEquil");
		nProc = prms.getValueOfKey<int>("nProc");
		sweepEval = prms.getValueOfKey<int>("sweepEval");
		boxEdge = prms.getValueOfKey<double>("boxEdge");
		transFactor = prms.getValueOfKey<double>("transFactor");
		angMag = prms.getValueOfKey<double>("angMag");
		onefile = prms.getValueOfKey<int>("onefile");
		printUncross = prms.getValueOfKey<bool>("printUncross");
		shape = prms.getValueOfKey<std::string>("shape");
		

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



	void writeParams(string fname) {
		ofstream fout;
		fout.open(fname, ios::out | ios::app | ios::binary);
		if (!fout.is_open()) {
			cout << "Could not open file for writing!" << endl;
			return;
		}
		double rho = (double) nObj / (boxEdge*boxEdge);
		double redRho = length*length*rho;
		fout << "Nx " << Nx << "|"
				 << "Ny " << Ny << "|"
				 << "nObj " << nObj << "|"
				 << "dr " << dr  << "|"
				 << "cellNx " << cellNx << "|"
				 << "cellNy " << cellNy  << "|"
				 << "nCell " << nCell << "|"
				 << "boxEdge " << boxEdge << "|"
				 << "box.x " << box.x << "|"
				 << "box.y " << box.y << "|"
				 << "cellWidth " << cellWidth << "|"
				 << "AR " << AR << "|"
				 << "shape " << shape << "|"
				 << "molWidth " << molWidth << "|"
				 << "length " << length << "|"
				 << "transFactor " << transFactor << "|"
				 << "transMag " << transMag  << "|"
				 << "angMag " << angMag << "|"
				 << "crossEval " << crossEval << "|"
				 << "printUncross " << printUncross << "|"
				 << "nEquil " << nEquil << "|"
				 << "nProc " << nProc << "|"
				 << "sweepEval " << sweepEval << "|"
				 << "Rho " << rho << "|"
				 << "ReducedRho " << redRho << endl;
		fout.close();
	}

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
				 << "shape " << shape << endl
				 << "molWidth " << molWidth << endl
				 << "length " << length << endl
				 << "transFactor " << transFactor << endl
				 << "transMag " << transMag  << endl
				 << "angMag " << angMag << endl
				 << "crossEval " << crossEval << endl
				 << "printUncross " << printUncross << endl
				 << "nEquil " << nEquil << endl
				 << "nProc " << nProc << endl
				 << "sweepEval " << sweepEval << endl
				 << endl
				 << "Rho " << rho << endl
				 << "Reduced rho " << redRho << endl;
	}
};
#endif
