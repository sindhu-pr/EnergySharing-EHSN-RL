#ifndef RANDGEN_H
#define RANDGEN_H

#include <cstdlib>
#include <math.h>
#include <ctime>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

class randgen{

public: randgen();
	double drand();
	double random_normal();
};

#endif


