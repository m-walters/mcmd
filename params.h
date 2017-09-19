#ifndef PARAMS
#define PARAMS
struct Params
{
	int Nx, Ny, nObj;
	// density defined as num per cell
	int sqrtDens;
 	double dr;
	int cellNx, cellNy, nCell;
	Vec<double> box;
	double AR;
	double molWidth;
	double length; // <1, fraction of cellwidth
	double cellWidth;
	double transFactor;
	double transMag;
	double angMag;
	int maxAttempts;
	int sweepEval;
	int sweepEvalProc;
	int sweepLimit;
	int nProc;
	int stepsInSweep;
	double boxEdge;

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
