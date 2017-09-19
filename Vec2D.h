#ifndef VEC2D
#define VEC2D
#include <iostream>
#include <algorithm>
#include <math.h>
#include <random>
#include <chrono>

using namespace std;

template <class Type> class Vec {
// Data
public:
	Type x,y;

public:
	// Constructors
	Vec() { }

	Vec(const Type a, const Type b) {
		x=a;
		y=b;
	}

	void set_values(const Type a, const Type b) {
	  x=a;
		y=b;
	}

	void randomize(double s) {
		x=cos(s);
		y=sin(s);
	}

	Type VProd() {
		return (x*y);
	}

	Type VLen() {
		return sqrt(x*x+y*y);
	}

	Type VLenSQ() {
		return (x*x+y*y);
	}

	Type VCsum() {
		return (x+y);
	}

	Vec<Type> operator+(const Vec<Type> &v) {
		Type a = x+v.x;
		Type b = y+v.y;
		return Vec(a,b);
	}

	Vec<Type> operator-(const Vec<Type> &v) {
		Type a = x-v.x;
		Type b = y-v.y;
		return Vec(a,b);
	}

	double operator*(const Vec<Type> v) {
		 return (x*v.x+y*v.y);
	}

	Vec<Type> operator*(Type s) {
		 Type a = x*s;
		 Type b = y*s;
		 return Vec(a,b);
	}

	friend Vec<Type> operator*(Type s,Vec<Type> v) {
	return Vec<Type>(s*v.x,s*v.y);
	}
};
#endif
