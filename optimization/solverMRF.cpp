#include <stdio.h>
#include <iostream>
#include <fstream>
#include <math.h>
#include <string>

#include <cv.h>
#include <highgui.h>

#include "solverMRF.h"
#include "../common/imageOp.h"
#include "../MRF/ICM.h"
#include "../MRF/GCoptimization.h"

using namespace std;

// Edata = coh1 * abs (d-d_gt)
//MRF::CostVal solverMRF::dCost(int pix, int i)
//{
//	int nRows = pix/mnWidth;
//	int nCols = pix%mnWidth;
//	MRF::CostVal result = abs(mImgDepthRescale.at<float>(nRows, nCols) - i) * mImgCoh.at<float>(nRows, nCols);
//	return result;
//}

// Esmooth = 
//MRF::CostVal solverMRF::fnCost(int pix1, int pix2, int i, int j)
//{
//	const float lambda = mfLambda4SmoothTerm;
//	int nCols1 = pix1%mnWidth;
//	int nRows1 = pix1/mnWidth;
//	int nCols2 = pix2%mnWidth;
//	int nRows2 = pix2/mnWidth;
//	MRF::CostVal result = abs(i-j)^k;
//	result = result > maxVal ? maxVal : result;
//	return result;
//}

const float solverMRF::epsilon = 0.00001;
const float solverMRF::gamma = 30;
const int solverMRF::k = 2;
const MRF::CostVal solverMRF::maxVal = 100;
const int solverMRF::nIterMax = 100;
const float solverMRF::fTerminate = 0.00001;
const int solverMRF::mnNumOfLabels = 64;
const float solverMRF::mfLambda4SmoothTerm = 0.2;

void solverMRF::setDataArray()
{
	if (mpDataArray == NULL){
		mpDataArray = new MRF::CostVal[mnHeight*mnWidth*mnNumOfLabels];
	}
	for (int iLabel=0; iLabel<mnNumOfLabels; iLabel++){
		for (int iRow=0; iRow<mnHeight; iRow++){
			for (int iCol=0; iCol<mnWidth; iCol++){
				int pix = iRow*mnWidth+iCol;
				mpDataArray[pix*mnNumOfLabels+iLabel]
				= abs(mImgDepthRescale.at<float>(iRow, iCol) - iLabel) * mImgCoh.at<float>(iRow, iCol);
			//	cout<<mpDataArray[pix*mnNumOfLabels+iLabel]<<endl;
			}
		}
	}
}

void solverMRF::setCueArray()
{
	if (mpHCue == NULL){
		mpHCue = new MRF::CostVal[mnHeight*mnWidth];
	}
	if(mpVCue == NULL){
		mpVCue = new MRF::CostVal[mnHeight*mnWidth];
	}
	assert(mImgView.channels()== 3 && mImgView.depth() == CV_8U);

	// vertical (x,y) (x,y+1)
	for (int iRow=0; iRow<mnHeight-1; iRow++){
		for (int iCol=0; iCol<mnWidth; iCol++){
			int pix = iRow*mnWidth+iCol;	
			double dDis = 0;
			// compute dDis...
			cv::Vec3b v1 = mImgView.at<cv::Vec3b>(iRow, iCol);
			cv::Vec3b v2 = mImgView.at<cv::Vec3b>(iRow+1, iCol);
			dDis = cv::norm(v1-v2);
			dDis = exp(-dDis/gamma);
			if (dDis<epsilon)
				dDis = epsilon;
			mpVCue[pix] = dDis;
		}
	}

	// horizontal (x+1, y) (x,y)
	for (int iRow=0; iRow<mnHeight; iRow++){
		for (int iCol=0; iCol<mnWidth-1; iCol++){
			int pix = iRow*mnWidth+iCol;	
			double dDis = 0;
			// compute dDis...
			cv::Vec3b v1 = mImgView.at<cv::Vec3b>(iRow, iCol);
			cv::Vec3b v2 = mImgView.at<cv::Vec3b>(iRow, iCol+1);
			dDis = cv::norm(v1-v2);
			dDis = exp(-dDis/gamma);
			if (dDis<epsilon)
				dDis = epsilon;
			mpHCue[pix] = dDis;
		}
	}
}

solverMRF::solverMRF()
	: meOptMethod(SWAP_OPT)
	, mpDataArray(NULL)
	, mpVCue(NULL)
	, mpHCue(NULL)
{

}

//solverMRF::solverMRF(int numOfLable, int nIterTimes, float fLambda, MRFOptMethod method)
//	: mnNumOfLabels(numOfLable)
//	, mnIterTimes(nIterTimes)
//	, mfLambda4SmoothTerm(fLambda)
//	, meOptMethod(method)
//	, mpDataArray(NULL)
//	, mpVCue(NULL)
//	, mpHCue(NULL)
//{
//}

void solverMRF::solve(cv::Mat& imgResult)
{
	// rescale input depth map [dmin dmax] -> [0 NumOfLabels)
	mImgDepthRescale = cv::Mat(mImgDepth.size(), CV_32F);
	double dmax, dmin;
	cv::minMaxLoc(mImgDepth, &dmin, &dmax);
	double dInterval = (dmax-dmin)/(mnNumOfLabels-1);	
	for (int iRows=0; iRows<mnHeight; iRows++){
		for (int iCols=0; iCols<mnWidth; iCols++){
			mImgDepthRescale.at<float>(iRows, iCols) 
				= (mImgDepth.at<float>(iRows, iCols) - dmin)/dInterval;
		//	cout<<mImgDepthRescale.at<float>(iRows, iCols) <<endl;
		}
	}
	setDataArray();
	setCueArray();

	MRF* mrf;
    EnergyFunction *energy;
    MRF::EnergyVal ETotal;
	MRF::EnergyVal ESmooth;
	MRF::EnergyVal EData;
	MRF::EnergyVal ELastIter;

	DataCost *data = new DataCost(mpDataArray);
	SmoothnessCost *smooth;
	if (mpVCue==NULL || mpHCue==NULL) 
		smooth = new SmoothnessCost(k, maxVal, mfLambda4SmoothTerm);
	else
		smooth = new SmoothnessCost(k, maxVal, mfLambda4SmoothTerm, mpHCue, mpVCue);
	energy = new EnergyFunction(data,smooth);

	switch (meOptMethod){
	case ICM_OPT:
		printf("\n*******Started ICM *****\n");
		mrf = new ICM(mnWidth, mnHeight, mnNumOfLabels, energy);
		break;
	case EXPANSION:
		printf("\n*******Started graph-cuts expansion *****\n");
		mrf = new Expansion(mnWidth, mnHeight, mnNumOfLabels, energy);
		break;
	case SWAP_OPT:
		printf("\n*******Started graph-cuts swap *****\n");
		mrf = new Swap(mnWidth, mnHeight, mnNumOfLabels, energy);
		break;
	default:
		// how to deal with unexpected input??
		break;
	}

	mrf->initialize();
	mrf->clearAnswer();	
	ETotal = mrf->totalEnergy();
	ESmooth = mrf->smoothnessEnergy();
	EData = mrf->dataEnergy();
	cout<<"Smooth Energy:"<<ESmooth<<", Data Energy:"<<EData<<", Total Energy:"<<ETotal<<endl;

	float fTime(0);
	float fTotalTime(0);
	for (int iterIter = 0; iterIter<nIterMax; iterIter++){
		if (iterIter>0 && (ELastIter-ETotal)/ELastIter<fTerminate){
			break;
		}
		if (ETotal < 0 ) break;
						
		mrf->optimize(1,fTime);  
		ESmooth = mrf->smoothnessEnergy();
		EData   = mrf->dataEnergy();
		ELastIter = ETotal;
		ETotal = ESmooth + EData;
		fTotalTime += fTime;
		cout<<"Smooth Energy:"<<ESmooth<<", Data Energy:"<<EData<<", Total Energy:"<<ETotal<<endl;
	}// end iteration of optimization

	if (imgResult.empty())
		imgResult.create(mImgDepth.size(), CV_32F);
	if (mImgResult.empty())
		mImgResult.create(mImgDepth.size(), CV_32F);

	// get result [dmin dmax] <- [0 NumOfLabels)
	for (int iterCols =0; iterCols < mnWidth; iterCols++ ){
		for (int iterRows=0; iterRows < mnHeight; iterRows++){
			uchar tmp = mrf->getLabel(iterRows*mnWidth+iterCols);
			//cout<<(int)tmp<<endl;
			mImgResult.at<float>(iterRows, iterCols) 
			= imgResult.at<float>(iterRows, iterCols)
			= tmp * dInterval + dmin;
		}				
	}
}