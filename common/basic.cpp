#include "basic.h"

float max3f(const float& f1, const float& f2, const float& f3)
{
	float fMax = f1;
	if (fMax<f2) fMax = f2;
	if (fMax<f3) fMax = f3;
	return fMax;
}

float min3f(const float& f1, const float& f2, const float& f3)
{
	float fMin = f1;
	if (fMin>f2) fMin = f2;
	if (fMin>f3) fMin = f3;
	return fMin;
}