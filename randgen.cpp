#include "randgen.h"

randgen::randgen()
{
}
double randgen::drand()   /* uniform distribution, (0..1] */
{
  //srand(time(0));
  return (rand()+1.0)/(RAND_MAX+1.0);
}
double randgen::random_normal()  /* normal distribution, centered on 0, std dev 1 */
{
  //srand(time(0));
  return sqrt(-2*log(drand())) * cos(2*M_PI*drand());
}



