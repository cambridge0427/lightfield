#include <time.h>
#include <stdio.h>
#include <iostream>

//#include <cv.h>
#include <highgui.h>

#include "../common/imageOp.h"
#include "../common/basic.h"

#include "msImageProcessor.h"
#include "BgImage.h"
#include "BgEdgeDetect.h"


using namespace std;

int main()
{
	char strTmp[100];
	char strBaseDir[100];
	int nData = 0;
	sprintf(strBaseDir, "C:\\work\\data4\\data%02d", nData);

	// load an image
	//sprintf(strTmp, "%s\\imgView.jpg", strBaseDir);
	sprintf(strTmp, "%s\\imgDepthT.jpg", strBaseDir);
	cv::Mat imgView = cv::imread(strTmp, 1);
	int nWidth = imgView.cols;
	int nHeight = imgView.rows;
	
	//obtain image type (color or grayscale)
	imageType	gtype;
	if(imgView.channels()==3)
		gtype = COLOR;
	else
		gtype = GRAYSCALE;
	gtype = GRAYSCALE;

	BgImage* cbgImage_ = new BgImage();
	BgImage* filtImage_ = new BgImage();
	BgImage* segmImage_ = new BgImage();
	BgImage* whiteImage_ = new BgImage();

	//segmemtation parameters
	int		sigmaS(16), minRegion(400), kernelSize(2);
	float	sigmaR(8), aij(0.3), epsilon(0.3);
	float	*gradMap_(NULL), *confMap_(NULL), *weightMap_(NULL), *customMap_(NULL);
	SpeedUpLevel	speedUpLevel_ = MED_SPEEDUP /*NO_SPEEDUP*/ /*HIGH_SPEEDUP*/;
	float speedUpThreshold_(0.1);

	if (gtype==COLOR){
		cbgImage_->SetImageFromRGB(imgView.data, nWidth, nHeight, true);
	}else{
		cbgImage_->SetImageFromRGB(imgView.data, nWidth, nHeight, false);
	}

	//if gradient and confidence maps are not defined, 
	//and synergistic segmentation is requested, then compute them;
	//also compute them if the parameters have changed
	bool bUseWeightMap = true;
	if (bUseWeightMap){
		confMap_ = new float[nWidth*nHeight];
		gradMap_	= new float[nWidth*nHeight];
		
		//compute gradient and confidence maps
		BgEdgeDetect	edgeDetector(kernelSize);
		edgeDetector.ComputeEdgeInfo(cbgImage_, confMap_, gradMap_);
			
		//compute weight map...
		//allocate memory for weight map
		weightMap_ = new float[nWidth*nHeight];
			
		//compute weight map using gradient and confidence maps
		int i;
		for (i=0; i<nWidth*nHeight; i++)
		{
			if (gradMap_[i] > 0.02)
				weightMap_[i] = aij*gradMap_[i] + (1 - aij)*confMap_[i];
			else
				weightMap_[i] = 0;
		}	
	}

	//create instance of image processor class
	msImageProcessor *iProc = new msImageProcessor();
	iProc->DefineImage(cbgImage_->im_, gtype, nHeight, nWidth);
	iProc->SetWeightMap(weightMap_, epsilon);

	//check for errors in image definition or in the setting
	//of the confidence map...
	if (iProc->ErrorStatus)
	{
		sprintf(strTmp,"%s\n", iProc->ErrorMessage);
		cout<<strTmp<<endl;
		return -1;
	}

	//perform image segmentation or filtering....
	//timer_start(); //start the timer
    iProc->SetSpeedThreshold(speedUpThreshold_);

	//filter the image...
	iProc->Filter(sigmaS, sigmaR, speedUpLevel_);
	if (iProc->ErrorStatus == EL_ERROR)
	{
		sprintf(strTmp, "%s\n", iProc->ErrorMessage);
		cout<<strTmp<<endl;
		return -1;
	}

	int dim = imgView.channels();
	unsigned char *tempImage = new unsigned char [dim*nHeight*nWidth];
	iProc->GetResults(tempImage);
	if (iProc->ErrorStatus == EL_ERROR)
	{
		sprintf(strTmp, "%s\n", iProc->ErrorMessage);
		cout<<strTmp<<endl;
		delete [] tempImage;
		return -1;
	}
		
	//fuse regions...
	iProc->FuseRegions(sigmaR, minRegion);
	if (iProc->ErrorStatus == EL_ERROR)
	{
		sprintf(strTmp, "%s\n", iProc->ErrorMessage);
		cout<<strTmp<<endl;
		delete [] tempImage;
		return -1;
	}

	//obtain the segmented and filtered image...
	filtImage_->Resize(nWidth, nHeight, cbgImage_->colorIm_);
	if (gtype==COLOR){
		memcpy(filtImage_->im_, tempImage, dim*nHeight*nWidth*sizeof(unsigned char));
	}else{
		memcpy(filtImage_->im_, tempImage, nHeight*nWidth*sizeof(unsigned char));
	}
	delete [] tempImage;
	segmImage_->Resize(nWidth, nHeight, cbgImage_->colorIm_);
	iProc->GetResults(segmImage_->im_);
	if (iProc->ErrorStatus)
	{
		sprintf(strTmp, "%s\n", iProc->ErrorMessage);
		cout<<strTmp<<endl;
		return -1;
	}

	//save result
	cv::Mat imgResult(nHeight, nWidth, CV_32F);
	for (int iterH=0; iterH<nHeight; iterH++){
		for(int iterW=0; iterW<nWidth; iterW++){
			if (gtype==COLOR){
				imgResult.at<float>(iterH, iterW) = segmImage_->im_[iterH*nWidth*dim+iterW*dim];
			}else{
				imgResult.at<float>(iterH, iterW) = segmImage_->im_[iterH*nWidth+iterW];
			}
		}
	}
	sprintf(strTmp, "output\\d%d_%d_%2.1f_%d.jpg", nData, sigmaS, sigmaR, minRegion);
	saveImage(strTmp, imgResult);

	delete whiteImage_;
	delete segmImage_;
	delete filtImage_;
	delete cbgImage_;

	if (customMap_)	delete [] customMap_;
	if (confMap_)	delete [] confMap_;
	if (gradMap_)	delete [] gradMap_;
	if (weightMap_)	delete [] weightMap_;

	return 1;
}