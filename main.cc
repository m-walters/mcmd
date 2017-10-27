#include "master.h"
#include <chrono>
#include <fstream>

using namespace chrono;
//void setparams(Params & mypars);

int main(int argc, const char *argv[])
{     

	typedef high_resolution_clock myclock;
	Params *simparams = new Params(argv[argc-1]);

	simparams->printParams();
	if (simparams->cellWidth < simparams->length) {
		cout << "Error. Cell width is smaller than obj length" << endl;
		return 1;
	}
	Master<Rod> master(simparams);
	master.InitializeSim();
	master.WriteSweep(argv[1], argv[2]);
	int sweep = 1;
	for (; sweep <= simparams->sweepLimit; sweep++) {
		if (!master.noOverlap)
			master.MCSweep();
		else 
			break;
	}
	master.WriteSweep(argv[1], argv[2]);
	myclock::time_point beginning = myclock::now();
	if (master.noOverlap) {
		// Processing sweeps
		cout << "Beginning processing run at sweep " << sweep << endl
				 << "Performing " << simparams->nProc << " more sweeps..." << endl
				 << "Writing to " << argv[2] << argv[1] << endl;
		for (int i=1; i<simparams->nProc; i++) {
			master.MCSweep();
			if (i > simparams->nEquil) {
				if (i%simparams->sweepEvalProc==0) {
					if (argc == 2) {
						master.WriteSweep(to_string(*argv[1]), "");
					} else if (argc > 2) {
						master.WriteSweep(argv[1], argv[2]);
					}
				}
			}
		}
	}
	auto dur = duration_cast<seconds>(myclock::now() - beginning).count();
	cout << endl << "Processing time: " << dur << " seconds" <<  endl;

	return 0;
}
