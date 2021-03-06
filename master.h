#ifndef MASTER_H
#define MASTER_H
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <math.h>
#include <random>
#include <chrono>
#include <fstream>
#include <map>
#include <cstring>
#include "Vec2D.h"
#include "params.h"
#include "obj.h"

using namespace std;
using namespace chrono;

template<typename T>
class Master{

typedef typename multimap<int, Obj<T>*>::iterator cellmapIterator;
typedef pair<cellmapIterator, cellmapIterator> cellmapPair;

private:

	int cellNx, cellNy, nCell;
	int Nx, Ny, nObj;
	int sweepCount, crossEval;
	int sweepEval;
	int nFailure;
	double dr;
 	Vec<double> box;
	double cellWidth;
	double length;
	double molWidth;
	Obj<T> *ghost;
	double transMag, PtransMag;
	double angMag, PangMag;
	bool onefile;
	std::string shape;
	ofstream fout;

public:

	Master(Params *myparams):
		cellNx(myparams->cellNx),
		cellNy(myparams->cellNy),
		nCell(myparams->nCell),
		Nx(myparams->Nx),
		Ny(myparams->Ny),
		nObj(myparams->nObj),
		crossEval(myparams->crossEval),
		sweepEval(myparams->sweepEval),
		dr(myparams->dr),
		box(myparams->box),
		cellWidth(myparams->cellWidth),
		length(myparams->length),
		molWidth(myparams->molWidth),
		PtransMag(myparams->transMag),
		PangMag(myparams->angMag),
		onefile(myparams->onefile),
		shape(myparams->shape) {}
 
	bool noOverlap;
	double L;
	multimap<int, Obj<T>*> cellmap;

	void InitializeSim() {
		sweepCount = 0;
		noOverlap = false;
		ghost = new Obj<T>(-1);
		// Pre-Processing jiggle params
		//angMag = 0.2;
		//double transFactor = 0.1;
		angMag = 0.002;
		double transFactor = 0.02;
		transMag = transFactor*dr;

		InitMap();
	}


	void ReInitializeSim() {
		cellmap.clear();
		noOverlap = false;
		// Pre-Processing jiggle params
		//angMag = 0.2;
		//double transFactor = 0.1;
		angMag = 0.002;
		double transFactor = 0.02;
		transMag = transFactor*dr;

		InitMap();
	}


	void MCSweep(int sweepCount)
	{
		nFailure = 0;
		Sweep();
		if (!noOverlap) {
			if (sweepCount%crossEval == 0) {
				int o = TotalOverlap();
				cout << "Sweep " << sweepCount
						 << ", Overlaps: " << o 
						 << ", Failure rate: " << nFailure/double(nObj) << endl;
				if (o == 0) {
					noOverlap = true;
					angMag = PangMag;
					transMag = PtransMag;
				}
			}
		} else {
			if (sweepCount%sweepEval == 0) {
				cout << "Sweep " << sweepCount << ", failure rate: " 
						 << nFailure/double(nObj) << endl;
			}
		}
	}

	double normalizeAngle(double ang) {
		if (ang < 0.) ang += 2.*M_PI;
		if (ang > 2.*M_PI) ang -= 2.*M_PI;
		return ang;
	};

	void Sweep();
	void testfunc();
	void PrintMap();
	void WriteSweep(string fname);
	double EvalOrder();
	int TotalOverlap();
	
private:

	bool LineOverlap(Vec<double>[2], Vec<double>[2]);
	bool BoundaryClear(Obj<T> *);
	bool isOverlap(Obj<T> *, Obj<T> *);
	bool InitOverlapCheck(Obj<T> *);
	//void Sweep();
	void InitMap();
	void ReInitMap();
	void UpdateCellIdx(Obj<T> *);
	void UpdateCellNeighbors(Obj<T> *);
	void RandRotate(Obj<T> *);
	void GhostStep(const Obj<T> *);
	void WritePixelImg();
	int CountOverlap(Obj<T> *);

};

//
// DEFINITIONS
//
typedef high_resolution_clock myclock;

myclock::time_point beginning = myclock::now();
myclock::duration dur = myclock::now() - beginning;
unsigned seed = dur.count();
mt19937 generator(seed);
uniform_real_distribution<double> distribution(0,1.0);


template <typename T> void Master<T>::testfunc() {

}


template <typename T> int Master<T>::TotalOverlap()
{
	int nOverlap = 0;
	// Iterate over cellmap
	for (cellmapIterator it=cellmap.begin(); it!=cellmap.end(); it++) {
		nOverlap += (!BoundaryClear(it->second));
		int idx = it->second->cellIdx;
		for (cellmapIterator nbr=cellmap.equal_range(idx).first;
				 nbr!=cellmap.equal_range(idx).second; nbr++) {
			if (it->second->ID == nbr->second->ID) continue;
			nOverlap += isOverlap(it->second, nbr->second);
		}
		for (int n : it->second->neighborCells) {
			if (n!=-1) {
				for (cellmapIterator nbr=cellmap.equal_range(n).first;
						 nbr != cellmap.equal_range(n).second; nbr++) {
					nOverlap += isOverlap(it->second, nbr->second);
				}
			}
		}
	}
	return nOverlap;
}


template <typename T> double Master<T>::EvalOrder() 
{
	L = 0.;
	double s = 0., t = 0.; // When iterating, each rod will encounter itself
	int Nth = 0;
	// Iterate over cellmap
	if (0) {
		// if wanting to just do local comparisons
		for (cellmapIterator it=cellmap.begin(); it!=cellmap.end(); it++) {
			int idx = it->second->cellIdx;
			for (cellmapIterator nbr=cellmap.equal_range(idx).first;
					 nbr!=cellmap.equal_range(idx).second; nbr++) {
				if (nbr->second->ID == it->second->ID) continue;
				s += cos(2.*(it->second->angle - nbr->second->angle));
				t += sin(2.*(it->second->angle - nbr->second->angle));
				Nth++;
			}
			for (int n : it->second->neighborCells) {
				if (n!=-1) {
					for (cellmapIterator nbr=cellmap.equal_range(n).first;
							 nbr != cellmap.equal_range(n).second; nbr++) {
						if(0){
							s += cos(2.*(it->second->angle - nbr->second->angle));
							t += sin(2.*(it->second->angle - nbr->second->angle));
							Nth++;
						}
					}
				}
			}
		}
	}
	if (0) {
		s = 0., t = 0.; // When iterating, each rod will encounter itself
		Nth = nObj;
		for (cellmapIterator it=cellmap.begin(); it!=cellmap.end(); it++) {
			s += cos(2.*(it->second->angle));
			t += sin(2.*(it->second->angle));
		}
	}

	s /= Nth;
	t /= Nth;
	L = sqrt(s*s + t*t);

	if (!noOverlap) cout << "Order param L " << L << endl;
	return L;

}


template <typename T> void Master<T>::UpdateCellIdx(Obj<T> *p) 
{
	// Divide shifted coords x,y by cell width to locate
	int xCell = (int) (floor((0.5*box.x + p->rc.x) / cellWidth));
	int yCell = (int) (floor((0.5*box.y + p->rc.y) / cellWidth));
	p->cellIdx = xCell*cellNy + yCell;
} 



template <typename T> bool Master<T>::LineOverlap(Vec<double> r1[2], Vec<double> r2[2])
{
	double a1,b1,a2,b2; // slopes and intercepts
	a1 = (r1[0].y - r1[1].y) / (r1[0].x - r1[1].x);
	b1 = r1[0].y - a1*r1[0].x;
	a2 = (r2[0].y - r2[1].y) / (r2[0].x - r2[1].x);
	b2 = r2[0].y - a2*r2[0].x;

	double x = (b2-b1) / (a1-a2);

	if((x > min(r1[0].x,r1[1].x)) && (x < max(r1[0].x,r1[1].x)) &&
	   (x > min(r2[0].x,r2[1].x)) && (x < max(r2[0].x,r2[1].x)))
		return true;
	else
		return false;
}


template <> bool Master<Rod>::BoundaryClear(Obj<Rod> *p) {
	bool result = true;
	if (fabs(p->vert[0].x) > 0.5*box.x) result = false;
	if (fabs(p->vert[1].x) > 0.5*box.x) result = false;
	if (fabs(p->vert[0].y) > 0.5*box.y) result = false;
	if (fabs(p->vert[1].y) > 0.5*box.y) result = false;
	return result;
}


template <> bool Master<Mol>::BoundaryClear(Obj<Mol> *p) {
	bool result = true;
	if (abs(p->vert[0].x) > 0.5*box.x) result = false;
	if (abs(p->vert[1].x) > 0.5*box.x) result = false;
	if (abs(p->vert[2].x) > 0.5*box.x) result = false;
	if (abs(p->vert[3].x) > 0.5*box.x) result = false;
	if (abs(p->vert[0].y) > 0.5*box.y) result = false;
	if (abs(p->vert[1].y) > 0.5*box.y) result = false;
	if (abs(p->vert[2].y) > 0.5*box.y) result = false;
	if (abs(p->vert[3].y) > 0.5*box.y) result = false;
	return result;
}


template <> bool Master<Rod>::isOverlap(Obj<Rod> *A, Obj<Rod> *B)
{
	return LineOverlap(A->vert, B->vert);
}


template <> bool Master<Mol>::isOverlap(Obj<Mol> *A, Obj<Mol> *B)
{
	return false;
}


template <typename T> int Master<T>::CountOverlap(Obj<T> *p) {
	int nOverlap = 0;
	cellmapPair localCell = cellmap.equal_range(p->cellIdx); 
	cellmapIterator nbr = localCell.first; 
	while (nbr != localCell.second) {
		if (nbr->second->ID == p->ID) {nbr++; continue;}
		if (isOverlap(p, nbr->second)) nOverlap++;
		nbr++;
	}

	// Check neighbor cells
	for (int n : p->neighborCells) {
		if (n != -1) {
			cellmapIterator nbr = cellmap.equal_range(n).first;
			while(nbr != cellmap.equal_range(n).second) {
				if(isOverlap(p, nbr->second)) nOverlap++;
				nbr++;
			}
		}
	}
	return nOverlap;
};

template <typename T> void Master<T>::Sweep()
{
	nFailure = 0;
	for (int icell=0; icell < nCell; icell++) {

		cellmapIterator ref = cellmap.equal_range(icell).first;
		cellmapIterator end = cellmap.equal_range(icell).second;
		int count = 0;
		//int d = cellmap.count(icell);
		while (ref != end) {
			Obj<T> *copy = ref->second;
			// debugging
			if(0) {
				cout << "---------------------------" << endl;
				cout << "printing copy" << endl;
				copy->printObj();
			}
			int pre = CountOverlap(copy);
			GhostStep(copy);
			if (!BoundaryClear(ghost)) {
				nFailure++;
				ref++;
				count++;
				continue;
			}
			int post = CountOverlap(ghost);
			if (post <= pre) {
				// Commit ghost step
				if (ghost->cellIdx == copy->cellIdx) {
					ref->second->Copy(*ghost);
					ref++;
				} else {
					copy->Copy(*ghost);
					cellmap.insert(pair<int, Obj<Rod>*>(copy->cellIdx, copy)); 
					cellmap.erase(ref);
					ref = next(cellmap.equal_range(icell).first, count);
					if (ref == next(end)) break;
				}
			} else {
				nFailure++;
				ref++;
			}
			count++;
		}
	}
};


template <> void Master<Rod>::InitMap()
{
	Vec<double> c;
	Vec<double> shift(dr/2.0 - 0.5*box.x, dr/2.0 - 0.5*box.y);
	uniform_real_distribution<double> distribution(0,1.0);
	double th = distribution(generator);
	double dth = 0;
	double x = 0.,y = 0.;
		
	int n=0;
	cout << "start init" << endl;

	for (int nx=0; nx<Nx; nx++) {
		for (int ny=0; ny<Ny; ny++) {
			Obj<Rod> *r = new Obj<Rod>(n);
			c.set_values(nx*dr, ny*dr);
			c = c+shift;
			// Jiggle xy a bit
			c.y += 0.0001*(2.*distribution(generator)-1.);
			c.x += 0.0001*(2.*distribution(generator)-1.);

			x = c.x;
			y = c.y;
			dth = 0.1*(2.*distribution(generator) - 1.);

			if (!shape.compare("iso")) {
				// Random dist
				th = 2.*M_PI*distribution(generator);
			}
			if (!shape.compare("D")) {
				// D config
				th = -M_PI*0.25 + dth;

				if (x - y > 0.1*box.x) {
					th = (x+y)/box.x*M_PI*0.35 + -M_PI*0.25 + dth;
				}
				if (y - x > 0.1*box.x) {
					th = -(x+y)/box.x*M_PI*0.35 + -M_PI*0.25 + dth;
				}
				
				// The following adjusts the wall and corner rods "properly"
				if ((x>y+0.1*box.y) && (x>0.4*box.x))
					th = 0 + dth; // along right wall
				if ((y-0.1*box.y>x) && (x<-0.4*box.x))
					th = 0 + dth; // along left wall
				if ((y<x-0.1*box.x) && (y<-0.4*box.y))
					th = M_PI*0.5 + dth;
				if ((y>x+0.1*box.x) && (y>0.4*box.y))
					th = M_PI*0.5 + dth;
				// handle the corners
				if ((fabs(x+y) < 0.1*box.x) && (fabs(x) > 0.4*box.x))
					th = M_PI*0.25 + dth;
				if ((fabs(x-y) < 0.1*box.x) && (fabs(x) > 0.4*box.x))
					th = -M_PI*0.25 + dth;

			}
			if (!shape.compare("T")) {
				// Generating T configuration
				double thFactor;
				if ((x>y) && (x<-y)) {
					thFactor = 0.5*(y-x)/y; 
					th = -M_PI*0.25 - M_PI*0.5*thFactor + dth;
				}
				if ((x<y) && (x<-y)) {
					thFactor = (x-y)/x;
					th = -M_PI*0.25 + M_PI*0.25*thFactor + dth;
				}
				if (x>-y) {
					th = M_PI*0.25 + dth;
				}
				if ((nx + ny) == (Ny-1)) {
					th = M_PI*0.25 + dth;
				}
				if (nx>Nx-3) th = dth;
				if (nx<2) th = dth;
				if (ny>Ny-3) th = M_PI*0.5 + dth;
				if (ny<2) th = M_PI*0.5 + dth;
			}
			if (!shape.compare("X")) {
				// Generating X configuration
				double thFactor;
				if ((x>y) && (x>-y)) {
					thFactor = 1. - (x-y)/x;
					th = -M_PI*0.25*thFactor + dth;
				}
				if ((x>y) && (x<-y)) {
					thFactor = 0.5*(y-x)/y; 
					th = -M_PI*0.25 - M_PI*0.5*thFactor + dth;
				}
				if ((x<y) && (x>-y)) {
					thFactor = 0.5*(y-x)/y; 
					th = -M_PI*0.25 - M_PI*0.5*thFactor + dth;
				}
				if ((x<y) && (x<-y)) {
					thFactor = (x-y)/x;
					th = -M_PI*0.25 + M_PI*0.25*thFactor + dth;
				}
				if (nx == ny) {
					r->diag = true;
					th = M_PI*0.75 + dth;
				}
				if ((nx + ny) == (Ny-1)) {
					th = M_PI*0.25 + dth;
					r->diag = true;
				}
				if ((nx==0) && (ny==0)) {
					c.x += length*0.3;
					c.y += length*0.3;
				}
				if ((nx==Nx-1) && (ny==0)) {
					c.x -= length*0.3;
					c.y += length*0.3;
				}
				if ((nx==0) && (ny==Ny-1)) {
					c.x += length*0.3;
					c.y -= length*0.3;
				}
				if ((nx==Nx-1) && (ny==Ny-1)) {
					c.x -= length*0.3;
					c.y -= length*0.3;
				}
			}
			if (!shape.compare("L")) {
				// Generating L configuration
				// Like the X, but vertical in, say, the middle quarter 
				// Also make the curved regions "think" they are
				// approaching the center of the X shape
				double thFactor;
				double ythresh = box.y/8.;
				if (fabs(y) < ythresh) {
					th = dth;
				} else {
					if (y>0.) y -= ythresh;
					if (y<0.) y += ythresh;
					y *= 4./3.*0.5*box.y;

					if ((x>y) && (x>-y)) {
						thFactor = 1. - (x-y)/x;
						th = -M_PI*0.25*thFactor + dth;
					}
					if ((x>y) && (x<-y)) {
						thFactor = 0.5*(y-x)/y; 
						th = -M_PI*0.25 - M_PI*0.5*thFactor + dth;
					}
					if ((x<y) && (x>-y)) {
						thFactor = 0.5*(y-x)/y; 
						th = -M_PI*0.25 - M_PI*0.5*thFactor + dth;
					}
					if ((x<y) && (x<-y)) {
						thFactor = (x-y)/x;
						th = -M_PI*0.25 + M_PI*0.25*thFactor + dth;
					}
				}
				// Adjust corner rods
				if ((nx==0) && (ny==0)) {
					c.x += length*0.3;
					c.y += length*0.3;
				}
				if ((nx==Nx-1) && (ny==0)) {
					c.x -= length*0.3;
					c.y += length*0.3;
				}
				if ((nx==0) && (ny==Ny-1)) {
					c.x += length*0.3;
					c.y -= length*0.3;
				}
				if ((nx==Nx-1) && (ny==Ny-1)) {
					c.x -= length*0.3;
					c.y -= length*0.3;
				}
			}
			if (!shape.compare("U")) {
				// Generating U configuration
				double thFactor;
				double ythresh = box.y*0.5 - 2.2*dr;

				if (y > ythresh) {
					if (y>fabs(x)) th = M_PI*0.5 + dth;
					if (y<fabs(x)) th = dth;
					if (nx == ny || nx == ny+1) {
						r->diag = true;
						th = M_PI*0.75 + dth;
					}
					if ((nx+ny)==(Ny-1) || (nx+ny+1)==(Ny-1)) {
						th = M_PI*0.25 + dth;
						r->diag = true;
					}
					if (nx==0 or nx==Nx-1) th = dth;
				} else {
					double yy = y - ythresh; // normalize to area below ythresh
					double halfbox = box.x*0.5;
					yy *= 1./(halfbox+ythresh)*halfbox;

					if (x>0.25*halfbox) {
						thFactor = 1. - (0.8*x - 0.2*yy*yy/halfbox)/halfbox;
						th = M_PI*0.5*thFactor + dth;
					}
					else if (x<-0.25*halfbox) {
						thFactor = 1. - (-0.8*x - 0.2*yy*yy/halfbox)/halfbox;
						th = -M_PI*0.5*thFactor + dth;
					}
					else {
						th = M_PI*0.5 + dth;
					}

				}
				// Adjust middle rods
				if (fabs(x) < 0.1*box.x ) { 
					th = M_PI/2 + dth;
				}

				// Top and bottom walls
				if ((y<-0.4*box.y))
					th = M_PI*0.5 + dth;
				if ((y>0.4*box.y))
					th = M_PI*0.5 + dth;
				/*
				if ((x<-0.44*box.x))
					th = dth;
				if ((x>0.44*box.x))
					th = dth;
				*/
				// Adjust corner rods
				if ((nx==0) && (ny==0)) {
					th = -M_PI*0.25 + dth;
					c.x += length*0.3;
					c.y += length*0.3;
				}
				if ((nx==Nx-1) && (ny==0)) {
					th = M_PI*0.25 + dth;
					c.x -= length*0.3;
					c.y += length*0.3;
				}
				if ((nx==0) && (ny==Ny-1)) {
					th = M_PI*0.25 + dth;
					c.x += length*0.3;
					c.y -= length*0.3;
				}
				if ((nx==Nx-1) && (ny==Ny-1)) {
					th = -M_PI*0.25 + dth;
					c.x -= length*0.3;
					c.y -= length*0.3;
				}
			}
			//
			// The following adjusts the wall and corner rods "properly"
			// Taken from "D" config
			/*
			if ((x>y+0.1*box.y) && (x>0.4*box.x))
				th = 0 + dth; // along right wall
			if ((y-0.1*box.y>x) && (x<-0.4*box.x))
				th = 0 + dth; // along left wall
			if ((y<x-0.1*box.x) && (y<-0.4*box.y))
				th = M_PI*0.5 + dth;
			if ((y>x+0.1*box.x) && (y>0.4*box.y))
				th = M_PI*0.5 + dth;
			*/
			// handle the corners
			if ((fabs(x+y) < 0.1*box.x) && (fabs(x) > 0.4*box.x))
				th = M_PI*0.25 + dth;
			if ((fabs(x-y) < 0.1*box.x) && (fabs(x) > 0.4*box.x))
				th = -M_PI*0.25 + dth;

			r->rc = c;
			r->vert[0].set_values(c.x, c.y + length*0.5);
			r->vert[1].set_values(c.x, c.y - length*0.5);
			r->angle = 0.;
			UpdateCellIdx(r);
			UpdateCellNeighbors(r);
			th = normalizeAngle(th);
			r->RotateVerts(th);
			r->angle = th;
			ghost->Copy(*r);

			while (!BoundaryClear(ghost)) {
				ghost->Copy(*r);
				int initIdx = ghost->cellIdx;
				double minx = ghost->vert[0].x;
				double miny = ghost->vert[0].y;
				double maxx = ghost->vert[0].x;
				double maxy = ghost->vert[0].y;
				for (Vec<double> v : ghost->vert) {
					if (v.x < minx) minx = v.x;
					if (v.y < miny) miny = v.y;
					if (v.x > maxx) maxx = v.x;
					if (v.y > maxy) maxy = v.y;
				}
				double dx = 0., dy = 0.;
				if (minx < -0.5*box.x) dx = -0.5*box.x - minx + 0.001;
				if (miny < -0.5*box.y) dy = -0.5*box.y - miny + 0.001;
				if (maxx > 0.5*box.x) dx = 0.5*box.x - maxx - 0.001;
				if (maxy > 0.5*box.y) dy = 0.5*box.y - maxy - 0.001;

				ghost->rc.x += dx;
				ghost->rc.y += dy;
				int i=0;
				for (Vec<double> v : ghost->vert) {
					ghost->vert[i].x = v.x + dx;
					ghost->vert[i].y = v.y + dy;
					i++;
				}
				// Update ghost
				UpdateCellIdx(ghost);
				if (ghost->cellIdx != initIdx) UpdateCellNeighbors(ghost);
			}
			r->Copy(*ghost);
			cellmap.insert(pair<int, Obj<Rod>*>(r->cellIdx, r)); 
			n=n+1;
		}
	}
	cout << "done init" << endl;
}


template <> void Master<Mol>::InitMap()
{
	Vec<double> c;
	Vec<double> shift(dr/2.0 - 0.5*box.x, dr/2.0 - 0.5*box.y);
	uniform_real_distribution<double> distribution(0,1.0);
	double th = distribution(generator);
		
	int n=0;
	for (int nx=0; nx <Nx; nx++) {
		for (int ny=0; ny <Ny; ny++) {
			Obj<Mol> *m = new Obj<Mol>(n);
			c.set_values(nx*dr, ny*dr);
			c = c+shift;
			m->rc = c;
			m->vert[0].set_values(c.x - molWidth/2.0, c.y - cellWidth/2.0);
			m->vert[1].set_values(c.x - molWidth/2.0, c.y + cellWidth/2.0);
			m->vert[2].set_values(c.x + molWidth/2.0, c.y + cellWidth/2.0);
			m->vert[3].set_values(c.x + molWidth/2.0, c.y - cellWidth/2.0);
			m->angle = 0.;
			th = distribution(generator);
			th *= M_PI;
			m->RotateVerts(th);
			UpdateCellNeighbors(m);
			UpdateCellIdx(m);
			cellmap.insert(pair<int, Obj<Mol>*>(m->cellIdx, m)); 
			n=n+1;
		}
	}
}


template <typename T> void Master<T>::UpdateCellNeighbors(Obj<T> *p) {
	int xCell = (int) floor((0.5*box.x + p->rc.x) / cellWidth);
	int yCell = (int) floor((0.5*box.y + p->rc.y) / cellWidth);
	p->neighborCells[0] = (xCell-1)*cellNy + yCell-1;
	p->neighborCells[1] = (xCell-1)*cellNy + yCell;
	p->neighborCells[2] = (xCell-1)*cellNy + yCell+1;
	p->neighborCells[3] = xCell*cellNy + yCell-1;
	p->neighborCells[4] = xCell*cellNy + yCell+1;
	p->neighborCells[5] = (xCell+1)*cellNy + yCell-1;
	p->neighborCells[6] = (xCell+1)*cellNy + yCell;
	p->neighborCells[7] = (xCell+1)*cellNy + yCell+1;
	if(yCell == cellNy-1) p->neighborCells[2] = p->neighborCells[4] = p->neighborCells[7] = -1;
	if(yCell == 0) p->neighborCells[0] = p->neighborCells[3] = p->neighborCells[5] = -1;
	if(xCell == cellNx-1) p->neighborCells[5] = p->neighborCells[6] = p->neighborCells[7] = -1;
	if(xCell == 0) p->neighborCells[0] = p->neighborCells[1] = p->neighborCells[2] = -1;
}


template <typename T> void Master<T>::RandRotate(Obj<T> *p) {
	uniform_real_distribution<double> distribution(0,1.0);
	//normal_distribution<double> distribution(0,1.0);
	double dt = angMag*M_PI*(2.*distribution(generator) - 1.);
	p->RotateVerts(dt);
	p->angle = normalizeAngle(p->angle + dt);
}


template <typename T> void Master<T>::GhostStep(const Obj<T> *copy)
{
	ghost->Copy(*copy);
	int initIdx = ghost->cellIdx;
	
	// Rotate
	if (!noOverlap && ghost->diag) {}
	else RandRotate(ghost);
	
	// Translate
	double dx = transMag*(2.*distribution(generator) - 1);
	double dy = transMag*(2.*distribution(generator) - 1);
	ghost->rc.x = ghost->rc.x + dx;
	ghost->rc.y = ghost->rc.y + dy;
	int i=0;
	for (Vec<double> v : ghost->vert) {
		ghost->vert[i].x = v.x + dx;
		ghost->vert[i].y = v.y + dy;
		i++;
	}
	
	// Update ghost
	UpdateCellIdx(ghost);
	if (ghost->cellIdx != initIdx) UpdateCellNeighbors(ghost);
};


template <typename T> void Master<T>::PrintMap() {
	cout << "Printing Map" << endl;
	cout << "---------------------------------" << endl;
	for (cellmapIterator it = cellmap.begin(); it != cellmap.end(); it++) {
		cout << it->first << " ";
		it->second->printObjLine();
	}
	cout << endl;
};

template <typename T> void Master<T>::WriteSweep(string fname) {
	fout.open(fname, ios::out | ios::app | ios::binary);
	if (!fout.is_open()) {
		cout << "Could not open file for writing!" << endl;
		return;
	}
	for (cellmapIterator it = cellmap.begin(); it != cellmap.end(); it++) {
		fout << it->second->ID << " " 
				 << it->second->cellIdx << " " 
				 << it->second->rc.x << " " 
				 << it->second->rc.y << " " 
				 << it->second->angle << " ";
		for(Vec<double> v : it->second->vert) {
			fout << v.x << " " << v.y << " ";
		}
		for(int i=0; i<8; i++) {
			fout << it->second->neighborCells[i] << " ";
		}
		fout << endl;
	}
	fout << endl;
	fout.close();
};
	

template <typename T> void Master<T>::WritePixelImg() {
	// Write the pixelized image for input to CNN
	if ((Nx % cellNx != 0) || (Ny % cellNy != 0)) {
		// pixels will straddle cell borders
		cout << "Pixels straddle cell border, please revise for pixelwrite" << endl;
		return;
	}
	double avgTh[Nx][Ny] = {};
	double S[Nx][Ny] = {};

	// Allocate rods to pixels
	// Find the key that each pixel belongs to
	// Allocate nthetas equivalent to nrods in key
	int x,y,xCell,yCell;
	double dpx = dr;
	double dpy = dr;
	int key;
	int npxPerCell = (int) cellWidth / dpx;
	int npyPerCell = (int) cellWidth / dpy;
	int nthtmp[Nx][Ny] = {};
	for (int px=0; px<Nx; px++) {
		for (int py=0; py<Ny; py++) {
			x = (int) (floor(Nx * (px*dpx + dpx/2) / box.x));
			y = (int) (floor(Ny * (py*dpy + dpy/2) / box.y));
			xCell = (int) px / npxPerCell;
			yCell = (int) py / npyPerCell;
			key = xCell*cellNy + yCell;
			int nrod = cellmap.count(key);
			double thetas[nrod] = {};
			int nth = 0;
			// Iterate over the cell
			for (cellmapIterator it = cellmap.equal_range(key).first;
					 it != cellmap.equal_range(key).second; it++) {
				double rx = it->second->rc.x + 0.5*box.x;
				double ry = it->second->rc.y + 0.5*box.y;
				if ((fabs(rx - x) < dpx/2) &&
						(fabs(ry - y) < dpy/2)) {
					thetas[nth] = it->second->angle;
					nth++;
					avgTh[px][py] += it->second->angle;
				}
			}
			if (nth < 2) {
				S[px][py] = 1.;
			} else {
				avgTh[px][py] /= nth;

				// Calculate S
				int nS = 0;
				for (int i=nth-1; i>0; i--) {
					for (int j=0; j<i; j++) {
						nS++;
						S[px][py] += cos(2.*(thetas[i] - thetas[j]));
					}
				}
				S[px][py] /= nS;
			}
		}
	}


	fout.open("output/pixelImg", ios::out | ios::trunc | ios::binary);
	for (int x=0; x<Nx; x++) {
		for (int y=0; y<Ny; y++) {
			fout << x << " " << y << " " << avgTh[x][y] << " " << S[x][y] << " " << nthtmp[x][y] << endl;
		}
	}
		
}

#endif
