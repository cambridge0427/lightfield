
#ifndef STRUCTURE_TENSOR_VIGRA
#define STRUCTURE_TENSOR_VIGRA

#include <cv.h>
#include <highgui.h>
#include <vigra/convolution.hxx>

void STwithVigra(const cv::Mat imgIn, cv::Mat & imgDepth, cv::Mat & imgCoh,
	double fInner, double fOuter);

#endif