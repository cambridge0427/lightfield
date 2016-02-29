#ifndef _BASIC
#define _BASIC

#include <math.h>
#include <float.h>

float max3f(const float& f1, const float& f2, const float& f3);

float min3f(const float& f1, const float& f2, const float& f3);

inline double round(double d){
  	return floor(d + 0.5);
}

inline bool IsNumber(double x) 
{
    // This looks like it should always be true, 
    // but it's false if x is a NaN.
    return (x == x); 
}
    
inline bool IsFiniteNumber(double x) 
{
	return (x <= DBL_MAX && x >= -DBL_MAX); 
}    
    

#endif
