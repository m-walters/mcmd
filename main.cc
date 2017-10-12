#include "master.h"
#include <chrono>
#include <fstream>

using namespace chrono;
void readin(Params & mypars);
void setparams(Params & mypars);

int main(int argc, const char *argv[])
{     

	typedef high_resolution_clock myclock;
	Params simparams;

	setparams(simparams);
	simparams.printParams();
	Master<Rod> master(simparams);
	master.InitializeSim();
	int sweep = 1;
	for (; sweep <= simparams.sweepLimit; sweep++) {
		if (!master.noOverlap)
			master.MCSweep();
		else {
			master.finalSweep = true;
			break;
		}
	}
	myclock::time_point beginning = myclock::now();
	if (master.noOverlap) {
		// Processing sweeps
		cout << "Beginning processing run at sweep " << sweep << endl
				 << "Performing " << simparams.nProc << " more sweeps..." << endl;
		for (int i=1; i<simparams.nProc; i++) {
			master.MCSweep();
			if (i%simparams.sweepEvalProc==0) {
				if (argc == 2) {
					master.WriteSweep(to_string(*argv[1]), "");
				} else if (argc == 3) {
					cout << "Writing to path " << argv[2] << endl;
					master.WriteSweep(argv[1], argv[2]);
				}
			}
		}
	}
	auto dur = duration_cast<seconds>(myclock::now() - beginning).count();
	cout << endl << dur << endl;

	return 0;
}
