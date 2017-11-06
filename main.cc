#include "master.h"
#include <chrono>
#include <fstream>

using namespace chrono;

int main(int argc, const char *argv[])
{     

	typedef high_resolution_clock myclock;
	Params *simparams = new Params(argv[argc-1]);

	simparams->printParams();
	if (simparams->cellWidth < simparams->length) {
		cout << "Error. Cell width is smaller than obj length" << endl;
		return 1;
	}
	// Write params to file
	simparams->writeParams(argv[1]);

	Master<Rod> master(simparams);
	master.InitializeSim();
	master.WriteSweep(argv[1]); // Write init map
	int sweep = 1;
	for (; sweep <= simparams->sweepLimit; sweep++) {
		if (!master.noOverlap)
			master.MCSweep();
		else 
			break;
	}
	master.WriteSweep(argv[1]); // Write first uncrossed img
	myclock::time_point beginning = myclock::now();
	if (master.noOverlap) {
		// Processing sweeps
		cout << "Beginning processing run at sweep " << sweep << endl
				 << "Performing " << simparams->nProc << " more sweeps..." << endl
				 << "Writing to " << argv[1] << endl;
		for (int i=1; i<simparams->nProc; i++) {
			master.MCSweep();
			if (i > simparams->nEquil) {
				if (i%simparams->sweepEvalProc==0) {
					master.WriteSweep(argv[1]);
					if (!simparams->shape.compare("T") || !simparams->shape.compare("X")) {
						double l = master.EvalOrder();
						cout << "Lambda " << l << endl;
						if (l>0.5) {
							cout << "Lambda exceeded 0.5, resetting map" << endl;
							master.ReInitializeSim();
							while (!master.noOverlap) master.MCSweep();
							for (int ii=0; ii<simparams->nEquil; ii++) master.MCSweep();
						}
					}
				}
			}
		}
	}



	auto dur = duration_cast<seconds>(myclock::now() - beginning).count();
	cout << endl << "Processing time: " << dur << " seconds" <<  endl;

	return 0;
}
