#include "master.h"
#include <chrono>
#include <fstream>

using namespace chrono;

int main(int argc, const char *argv[])
{     

	typedef high_resolution_clock myclock;
	Params *simparams = new Params(argv[argc-1]);

	// Create fresh file for output
	ofstream fout;
	fout.open(argv[1], ios::out | ios::trunc | ios::binary);
	if (!fout.is_open()) {
		cout << "Could not open file for writing!" << endl;
	}

	simparams->printParams();
	if (simparams->cellWidth < simparams->length) {
		cout << "Error. Cell width is smaller than obj length" << endl;
		return 1;
	}
	
	// Write params to file
	simparams->writeParams(argv[1]);

	myclock::time_point start_uncross = myclock::now();
	Master<Rod> master(simparams);
	master.InitializeSim();
	master.WriteSweep(argv[1]); // Write init map

	// Uncrossing phase
	int presweep = 1;
	while (!master.noOverlap) {
		master.MCSweep(presweep);
		presweep++;
	}

	// Write uncrossed img if desired
	if (simparams->printUncross) {
		master.WriteSweep(argv[1]);
	}

	// Equil and processing phase
	myclock::time_point start_proc = myclock::now();
	auto uncross_dur = duration_cast<seconds>(start_proc - start_uncross).count();
	cout << "Finished uncrossing phase (" << uncross_dur << " seconds)" << endl;
	
	cout << "Beginning equilibrium phase (" << simparams->nEquil << " sweeps)..." << endl;
	for (int i=1; i<simparams->nEquil; i++) {
		master.Sweep();
	}

	cout << "Done." << endl << "Beginning processing phase (" << simparams->nProc 
	     << " sweeps)..." << endl
			 << "Writing to " << argv[1] << endl;
	for (int i=1; i<=simparams->nProc; i++) {
		master.MCSweep(i);
		if (i%simparams->sweepEval==0) {
			master.WriteSweep(argv[1]);
			/////////////////////
			int resweep = 1;
			cout << "Resetting map.." << endl;
			master.ReInitializeSim();
			while (!master.noOverlap) {
				master.MCSweep(resweep);
				resweep++;
			}
			cout << "Equilibriating ("<< simparams->nEquil<< " sweeps)" << endl;
			for (int ii=0; ii <simparams->nEquil; ii++) master.Sweep();
			/////////////////////
			/*
			if (!simparams->shape.compare("T") || !simparams->shape.compare("X")) {
				double l = master.EvalOrder();
				cout << "Lambda " << l << endl;
				if (l>0.50) {
					cout << "Lambda exceeded 0.4, resetting map" << endl;
					master.ReInitializeSim();
					while (!master.noOverlap) master.Sweep();
					for (int ii=0; ii<simparams->nEquil; ii++) master.Sweep();
				}
			}
			*/
		}
	}


	auto proc_dur = duration_cast<seconds>(myclock::now() - start_proc).count();
	auto total_dur = duration_cast<seconds>(myclock::now() - start_uncross).count();
	cout << endl << "Processing time: " << proc_dur << " seconds" <<  endl;
	cout << endl << "Total time: " << total_dur << " seconds" <<  endl;

	return 0;
}
