#ifndef _IMAGE_OPERATION
#define _IMAGE_OPERATION

#include <cv.h>
#include <highgui.h>

// save an image
void saveImage(const std::string path, const cv::Mat img, bool bNormalize=true);
void saveStretchedImage(const std::string path, const cv::Mat img, const int nScale=10, bool bNormalize=true);

// show an image
void showImage(const cv::Mat img, const std::string strWinName, bool bNormalized=true);
// show image in a stretched way (vertical dimension)
void showStretchedImage(const cv::Mat img, const std::string strWinName,
	const int nScale=10, bool bNormalized=true);

// method: 2-L2Norm
void mergeChannel(cv::Mat imgIn, cv::Mat & imgOut, const int method = 2);

// normalize to 0-1 or 0-255
void normalize(cv::Mat & img);
//void normalize(cv::Mat imgIn, cv::Mat & imgOut);

// for a CV_32F image, change its value from 0-255 to 0-1
void unchar2Float(cv::Mat & img);

// get CV_32F image for CV_8U
cv::Mat getFloatImage(cv::Mat & img);

// get min and max of an 3-channal image
void minMaxLoc3D(cv::Mat img, double* min, double* max);

// clamp values in a certain section
void truncImage(cv::Mat & img, const float fUpBound, const float fLowBound);

// turn to abstract value
void getAbs(const cv::Mat imgIn, cv::Mat & imgOut);

// output histogram
void showHistogram(const cv::Mat img, const float fUpBound = 0, const float fLowBound = 0);

#endif
