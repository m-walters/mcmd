#ifndef MYOBJ
#define MYOBJ
#include "Vec2D.h"
#include "params.h"

struct Rod {
	Vec<double> rc;
	Vec<double> vert[2];
	double angle;
};

struct Mol{
	Vec<double> rc;
	Vec<double> vert[4];
	double angle;
};

template<class Base> class Obj: public Base {
// Base will be either Rod or Mol
public:
  int ID;
	int cellIdx;
	int neighborCells[8];

	Obj() {};
	Obj(int id): ID(id) {};
	~Obj() {};
	
	void printObj();
	void printObjLine();
	void RotateVerts(double);
	void Copy(Obj<Base> p);
	
	Obj<Base> & operator=(const Obj<Base> &p) {
		if (this == &p) return *this;
		this->ID = p.ID;
		this->rc.x = p.rc.x;
		this->rc.y = p.rc.y;
		this->angle = p.angle;
		this->cellIdx = p.cellIdx;
		int i = 0;
		for (Vec<double> v : p.vert) {
			this->vert[i] = v;
			i++;
		}
		for (int i=0; i<8; i++) {
			this->neighborCells[i] = p.neighborCells[i];
		}
		return *this;
	}

};

//
// DEFINITIONS
//
template <typename Base> void Obj<Base>::Copy(Obj<Base> p) {
	if (this==&p) return;
	this->ID = p.ID;
	this->rc.x = p.rc.x;
	this->rc.y = p.rc.y;
	this->angle = p.angle;
	this->cellIdx = p.cellIdx;
	int i = 0;
	for (Vec<double> v : p.vert) {
		this->vert[i] = v;
		i++;
	}
	for (int i=0; i<8; i++) {
		this->neighborCells[i] = p.neighborCells[i];
	}
};

template <> void Obj<Rod>::printObj() {
	cout << "Obj ID " << ID << endl
			 << "cellIdx " << cellIdx << endl
			 << "Rx " << rc.x << ", Ry " << rc.y << endl
			 << "Theta " << angle << endl
			 << "Verts (x,y)" << endl
			 << vert[0].x << " " << vert[0].y << endl
			 << vert[1].x << " " << vert[1].y << endl
			 << "Neighbors ";
	for(int i=0; i<8; i++) {
		cout << neighborCells[i] << " ";
	}
	cout << endl;
};

template <> void Obj<Mol>::printObj() {
	cout << "Obj ID " << ID << endl
			 << "cellIdx " << cellIdx << endl
			 << "Rx " << rc.x << ", Ry " << rc.y << endl
			 << "Theta " << angle << endl
			 << "Verts (x,y)" << endl
			 << vert[0].x << " " << vert[0].y << endl
			 << vert[1].x << " " << vert[1].y << endl
			 << vert[2].x << " " << vert[2].y << endl
			 << vert[3].x << " " << vert[3].y << endl
			 << "Neighbors ";
	for(int i=0; i<8; i++) {
		cout << neighborCells[i] << " ";
	}
	cout << endl;
};


template <> void Obj<Rod>::printObjLine() {
	cout << ID << " " 
	     << cellIdx << " " 
	     << rc.x << " " 
			 << rc.y << " " 
			 << angle << " " 
			 << vert[0].x << " " 
			 << vert[0].y << " " 
			 << vert[1].x << " "
			 << vert[1].y << " ";
	for(int i=0; i<8; i++) {
		cout << neighborCells[i] << " ";
	}
	cout << endl;
};

template <> void Obj<Mol>::printObjLine() {
	cout << ID << " " 
	     << cellIdx << " " 
	     << rc.x << " " 
			 << rc.y << " " 
			 << angle << " ";
	for(int i=0; i<4; i++) {
		cout << vert[i].x << " " << vert[i].y << " ";
	}
	for(int i=0; i<8; i++) {
		cout << neighborCells[i] << " ";
	}
	cout << endl;
};


template <typename Base> void Obj<Base>::RotateVerts(double theta) {
	// Rotate object by theta from current angle
	int i=0;
	for (Vec<double> v : this->vert) {
		this->vert[i].x = this->rc.x + (v.x - this->rc.x)*cos(theta) - (v.y - this->rc.y)*sin(theta);
		this->vert[i].y = this->rc.y + (v.x - this->rc.x)*sin(theta) + (v.y - this->rc.y)*cos(theta);
		i++;
	}
};

#endif
