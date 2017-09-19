#include "master.h"
#include <fstream>

void readin(Params & mypars);
void setparams(Params & mypars);

int main()
{     
	Params simparams;
	//readin(simparams);

	setparams(simparams);
	simparams.printParams();
	Master<Rod> master(simparams);
	master.InitializeSim();
	int sweep = 1;
	for (; sweep <= simparams.sweepLimit; sweep++) {
		if (!master.noOverlap)
			master.MCSweep();
		else
			break;
	}
	if (master.noOverlap) {
		// Final shakeup
		cout << "Beginning processing run at sweep " << sweep << endl
				 << "Performing " << simparams.nProc << " more sweeps..." << endl;
		for (int i=1; i<simparams.nProc; i++) {
			master.MCSweep();
		}
		master.finalSweep = true;
		master.MCSweep();
	}

	return 0;
}
