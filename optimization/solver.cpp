#include "solver.h"

void solver::loadImage(const cv::Mat& imgI, const cv::Mat& imgD, const cv::Mat& imgC)
{
	mnHeight = imgI.rows;
	mnWidth = imgI.cols;
	mnWidthStep = mnWidth * imgI.channels();
	mnChannels = imgI.channels();
	mnLeftX = 0; mnRightX = mnWidth;
	mnUpY = 0; mnDownY = mnHeight;

	mImgView = imgI.clone();
	mImgDepth = imgD.clone();
	mImgCoh = imgC.clone();

	mImgResult.create(mnHeight, mnWidth, CV_32F);
	mImgResult.setTo(0);
}

void solver::getResult(cv::Mat& imgResult)
{
	imgResult = mImgResult.clone();
}