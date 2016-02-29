
#ifndef STRUCTURE_TENSOR_OPENCV
#define STRUCTURE_TENSOR_OPENCV

#include <cv.h>
#include <highgui.h>

#include "../common/imageOp.h"
#include "../common/basic.h"

// Difference of Gaussian
// dim: 1-x dim; 2-y dim; 3-both
void DOG(cv::Mat imgIn, cv::Mat & imgOut, int kernel1, int kernel2, int dim=3);

// get gradient  with Sobel
void getGradientWithSobel(const cv::Mat img, const bool bYdim, const bool bXdim, 
	cv::Mat & imgOut, const int nSize);

void StructureTensor(const cv::Mat imgIn, cv::Mat & imgDepth, cv::Mat & imgCoh,
	double fThreshold = 100000);

void reviseST(const cv::Mat & imgIn, const cv::Mat & imgDepth, const cv::Mat & imgCoh,
	cv::Mat & imgCohRevised, cv::Mat & imgTest);

void reviseSTNew(const cv::Mat & imgIn, cv::Mat & sliceDepth, cv::Mat & sliceCoh,
	cv::Mat & sliceCohRevised, cv::Mat & sliceCost);

#endif