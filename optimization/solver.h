#ifndef SOLVER
#define SOLVER

#include <cv.h>

class solver{

public:
	// constructor
	// destructor

protected:
	int mnWidth; // full image
	int mnHeight;// full image
	int mnWidthStep;// full image
	//int mnSize;
	int mnChannels;
	int mnPatchSize;
	
	// images 
	cv::Mat mImgResult, mImgView, mImgDepth, mImgCoh, mImgGuess;

	// ROI 
	bool mbUsingMask;
	int mnLeftX, mnRightX, mnUpY, mnDownY;

public:
	void loadImage(const cv::Mat& imgI, const cv::Mat& imgd, const cv::Mat& imgC);
	virtual void solve(cv::Mat& imgResult) = 0;
	void getResult(cv::Mat& imgResult);
};

#endif