
#include "structureTensorOpenCV.h"

#include <vector>
#include <math.h>
#include <cfloat>
#include <fstream>

using namespace std;

const float PIE = 3.1415926;
const double SQRT3 = 1.7320508;
const double lowCohThresh = 0.90;
const float nOcclusionCost = 50.0;
const float alpha(25);
const float beta(5); // if color information is in [0 1]

// get gradient with Sobel
void getGradientWithSobel(const cv::Mat img, const bool bXdim, const bool bYdim,
	cv::Mat & imgOut, const int nSize)
{
	if (img.empty()) return;	

	int nDepth = CV_32F;
	int nScale = 1;
	imgOut.create(img.size(), CV_32F);

	if (img.channels() == 1){
		cv::Mat imgBlur, imgGray, imgGrad, imgTmp;
		imgTmp = img;
		if (bYdim && !bXdim){ 	// y dim
			Sobel( imgTmp, imgGrad, CV_32F, 0, 1, nSize, nScale);
		}else if (bXdim && !bYdim){	// x dim
			Sobel( imgTmp, imgGrad, CV_32F, 1, 0, nSize, nScale);
		}else if (bXdim && bYdim){
			Sobel( imgTmp, imgGrad, CV_32F, 1, 1, nSize, nScale);
		}
		//getAbs( imgGrad, imgOut );
		imgOut = imgGrad;
	}else {
		if (img.channels() != 3){
			cout<<"Warning: not 3 channel in Sobel function!"<<endl;
		}
		cv::Mat mv[3];
		cv::Mat imgGrad[3];
		split(img, mv);
		for (int iterC=0; iterC<3; iterC++){
			if (bYdim && !bXdim){ 	// y dim
				Sobel( mv[iterC], imgGrad[iterC], CV_32F, 0, 1, nSize, nScale);
			}else if (bXdim && !bYdim){	// x dim
				Sobel( mv[iterC], imgGrad[iterC], CV_32F, 1, 0, nSize, nScale);
			}else if (bXdim && bYdim){
				Sobel( mv[iterC], imgGrad[iterC], CV_32F, 1, 1, nSize, nScale);
			}
		}
		for (int iterH=0; iterH<img.rows; iterH++){
			for (int iterW=0; iterW<img.cols; iterW++){
				float f1 = imgGrad[0].at<float>(iterH, iterW);
				float f2 = imgGrad[1].at<float>(iterH, iterW);
				float f3 = imgGrad[2].at<float>(iterH, iterW);
				imgOut.at<float>(iterH, iterW) = max3f(f1, f2, f3);
				//imgOut.at<float>(iterH, iterW) = sqrt(f1*f1+f2*f2+f3*f3);
			}
		}
	}

	return;
}

// Derivatives : Difference of Gaussian
void DOG(cv::Mat imgIn, cv::Mat& imgOut, int kernel1, int kernel2, int dim)
{
	imgOut.create(imgIn.size(),imgIn.type());

	cv::Mat dog_1(imgIn.size(), CV_32F);
	cv::Mat dog_2(imgIn.size(), CV_32F);

	if(dim == 1){  //x dim
		cv::GaussianBlur(imgIn, dog_2, cv::Size(kernel1, 1), 0);
		cv::GaussianBlur(imgIn, dog_1, cv::Size(kernel2, 1), 0);
	}else if(dim == 2){ 		// y dim
		cv::GaussianBlur(imgIn, dog_2, cv::Size(1,kernel1), 0); // Gaussian blur
		cv::GaussianBlur(imgIn, dog_1, cv::Size(1,kernel2), 0); 
	}else{
		cv::GaussianBlur(imgIn, dog_2, cv::Size(kernel1,kernel1), 0); // Gaussian blur
		cv::GaussianBlur(imgIn, dog_1, cv::Size(kernel2,kernel2), 0); 
	}

	cv::absdiff(dog_1, dog_2, imgOut);
}

void StructureTensor(const cv::Mat imgIn, cv::Mat & imgDepth, cv::Mat & imgCoh, double fThreshold)
{
	int H = imgIn.rows;
	int W = imgIn.cols;

	imgDepth.create(H, W, CV_32F);
	imgCoh.create(H, W, CV_32F);

	int nOuterParam(7), nInnerParam(7);
	cv::Mat imgGradY, imgGradX, imgJxx, imgJyy, imgJxy, imgJxxBlur, imgJyyBlur, imgJxyBlur;
	cv::Mat imgTest(H,W,CV_32F);

	cv::Mat imgUsing;
	if(imgIn.channels()==1){
		imgUsing = imgIn;
	}else{
		mergeChannel(imgIn, imgUsing);
	}

	getGradientWithSobel(imgIn, false, true, imgGradY, nInnerParam);
	getGradientWithSobel(imgIn, true, false, imgGradX, nInnerParam);

	//DOG(imgInMerged, imgGradY, 5, 3, 2);
	//DOG(imgInMerged, imgGradX, 5, 3, 1);
	showStretchedImage(imgGradY, "gradY", 10, true);
	showStretchedImage(imgGradX, "gradX", 10, true);

	cv::multiply(imgGradY, imgGradY, imgJyy);
	cv::multiply(imgGradX, imgGradX, imgJxx);
	cv::multiply(imgGradX, imgGradY, imgJxy);
	showStretchedImage(imgJxx, "Jxx", 10, true);
	showStretchedImage(imgJyy, "Jyy", 10, true);
	showStretchedImage(imgJxy, "Jxy", 10, true);

	GaussianBlur( imgJxx, imgJxxBlur, cv::Size(nOuterParam,nOuterParam), 0, 0, cv::BORDER_DEFAULT );
	GaussianBlur( imgJyy, imgJyyBlur, cv::Size(nOuterParam,nOuterParam), 0, 0, cv::BORDER_DEFAULT );
	GaussianBlur( imgJxy, imgJxyBlur, cv::Size(nOuterParam,nOuterParam), 0, 0, cv::BORDER_DEFAULT );
	showStretchedImage(imgJxxBlur, "JxxBlur", 10, true);
	showStretchedImage(imgJyyBlur, "JyyBlur", 10, true);
	showStretchedImage(imgJxyBlur, "JxyBlur", 10, true);

	int nDebugPt = 263;
	float ftmp = imgJxy.at<float>(5, nDebugPt);
	float ftmp2 = imgJxyBlur.at<float>(5, nDebugPt);
	float ftmp3 = imgGradX.at<float>(5, nDebugPt);
	float ftmp4 = imgGradY.at<float>(5, nDebugPt);

	//double minValJxx, minValJyy, maxValJxx, maxValJyy;
	//minMaxLoc3D(imgJxxBlur, &minValJxx, &maxValJxx);
	//minMaxLoc3D(imgJyyBlur, &minValJyy, &maxValJyy);

	//cout<<"threshold for Jxx+Jyy : "<<fThreshold<<endl;
	int nCntPts = 0;
	if (imgJxxBlur.depth() == CV_32F){
		for (int iterY=0; iterY<H; iterY++){
			for (int iterX=0; iterX<W; iterX++){
				if(iterY == 5 && iterX == nDebugPt){
					int a1=1;
				}
				imgDepth.at<float>(iterY, iterX) = 0.5*atan2((float)(2*imgJxyBlur.at<float>(iterY, iterX)),
					(float)(imgJyyBlur.at<float>(iterY, iterX)-imgJxxBlur.at<float>(iterY, iterX)));
				float ftmp1=imgDepth.at<float>(iterY, iterX);
				float ftmp2=imgJxyBlur.at<float>(iterY, iterX);
				float ftmp3=imgJyyBlur.at<float>(iterY, iterX);
				float ftmp4=imgJxxBlur.at<float>(iterY, iterX);
				if(abs(imgJxxBlur.at<float>(iterY, iterX)+imgJyyBlur.at<float>(iterY, iterX)) > fThreshold){
					imgCoh.at<float>(iterY, iterX) = sqrt(
						pow((imgJyyBlur.at<float>(iterY, iterX)-imgJxxBlur.at<float>(iterY, iterX)), 2)
						+ pow(imgJxyBlur.at<float>(iterY, iterX),2)*4)
						/abs(imgJxxBlur.at<float>(iterY, iterX)+imgJyyBlur.at<float>(iterY, iterX));
				}else{
					imgCoh.at<float>(iterY, iterX) = 0;
					nCntPts++;
				}
				imgTest.at<float>(iterY, iterX) = imgDepth.at<float>(iterY, iterX);
				// not sure I added pie  >.<
				// better to leave it there now
				if (imgDepth.at<float>(iterY, iterX)<0){
					imgDepth.at<float>(iterY, iterX) += PIE;
				}
			}
		}
	}
	// to deal with CV_8U cases,
	// it seems not useful anymore
	//else{
	//	for (int iterY=0; iterY<H; iterY++){
	//		for (int iterX=0; iterX<W; iterX++){
	//			imgDepth.at<float>(iterY, iterX) = 0.5*atan2((float)(2*imgJxyBlur.at<uchar>(iterY, iterX)),
	//						(float)(imgJyyBlur.at<uchar>(iterY, iterX)-imgJxxBlur.at<uchar>(iterY, iterX)));

	//			if(abs(imgJxxBlur.at<uchar>(iterY, iterX)+imgJyyBlur.at<uchar>(iterY, iterX)) > fThreshold){
	//				imgCoh.at<float>(iterY, iterX) = sqrt(pow((float)(imgJyyBlur.at<uchar>(iterY, iterX)-imgJxxBlur.at<uchar>(iterY, iterX)), 2)
	//					 	+ pow((float)imgJxyBlur.at<uchar>(iterY, iterX),2) *4)
	//						/abs(imgJxxBlur.at<uchar>(iterY, iterX)+imgJyyBlur.at<uchar>(iterY, iterX));
	//			}else{
	//				imgCoh.at<float>(iterY, iterX) = 0;	
	//			}
	//			imgTest.at<float>(iterY, iterX) = imgDepth.at<float>(iterY, iterX);
	//			//if (imgDepth.at<float>(iterY, iterX)<0){
	//			//	imgDepth.at<float>(iterY, iterX) += PIE;
	//			//}
	//		}
	//	}
	//}

	//cout<<"Jxx+Jyy==0: "<<nCntPts<<" out of "<<H*W<<endl;
	return;
}

void reviseST(const cv::Mat & imgIn, const cv::Mat & imgDepth, const cv::Mat & imgCoh,
	cv::Mat & imgCohRevised, cv::Mat & imgTest)
{
	int nViews = imgIn.rows;
	int W = imgIn.cols;
	int nOptView = nViews/2;
	int C = imgIn.channels();

	imgTest.create(1, W, CV_32F);
	//imgTest.create(1, W, CV_8UC3);

	// rescale depth to 0-pie
	// rescale to -pie/2 ~ pie/2
	// tan 
	cv::Mat imgDepthTan(1, W, CV_32F);
	for (int iterW=0; iterW<W; iterW++){
		imgDepthTan.at<float>(0, iterW) = imgDepth.at<float>(nOptView, iterW) /** PIE*/;
		imgDepthTan.at<float>(0, iterW) = PIE/2 - imgDepthTan.at<float>(0, iterW); 
		imgDepthTan.at<float>(0, iterW) = tan(imgDepthTan.at<float>(0, iterW));
	}

	// compute disparity
	cv::Mat disparity(nViews, W, CV_32F);
	for (int iterH=0; iterH<nViews; iterH++){
		for (int iterW=0; iterW<W; iterW++){
			disparity.at<float>(iterH, iterW) = imgDepthTan.at<float>(0, iterW)*(nOptView-iterH);
		}
	}

	cv::Mat imgCost = cv::Mat::zeros(1, W, CV_32F);
	int nWinRadius = 1;
	int nViewRadius = 4;
	if (nViews/2<nViewRadius)
		nViewRadius = nViews/2;
	for (int iterW=0; iterW<W; iterW++){
		cv::Vec3f v;
		float fv;
		if (C==3){
			v = imgIn.at<cv::Vec3f>(nOptView, iterW);
		}else{
			fv = imgIn.at<float>(nOptView, iterW);
		}
		float d = imgDepth.at<float>(nOptView, iterW);
		float coh = imgCoh.at<float>(nOptView, iterW);
		// if the trunctate coh step is moved before this step,
		// then by jump around coh=0, we can save lot of time
		// now we can do 0.95, since the truncate threshold is 0.95
		if (coh < lowCohThresh){
			imgTest.at<float>(0, iterW) = imgCost.at<float>(0, iterW) = 0;
			continue;
		}
		vector<float> costList;
		for (int iterV=nViews/2-nViewRadius; iterV<=nViews/2+nViewRadius; iterV++){
			int nWarpedX = round(disparity.at<float>(iterV, iterW)) + iterW;
			int nXLow = nWarpedX-nWinRadius;
			int nXUp = nWarpedX+nWinRadius;
			//int nXLow = floor(disparity.at<float>(iterV, iterW)) + iterW;
			//int nXUp = nXLow+1;
			float nCostMin(FLT_MAX), nCurrentCost(0.0), nOcclusionCost(50.0);
			int nCostMinIdx;

			for (int iterInnerW = nXLow; iterInnerW<=nXUp; iterInnerW++){
				if (iterInnerW<0 || iterInnerW>=W){
					nCurrentCost = nOcclusionCost;	
				}else if (C==3){
					cv::Vec3f vWarped = imgIn.at<cv::Vec3f>(iterV, iterInnerW);
					float dWarped = imgDepth.at<float>(iterV, iterInnerW);
					float cohWarped = imgCoh.at<float>(iterV, iterInnerW);
					nCurrentCost = sqrt(pow((float)v[0]-vWarped[0],2) 
						+ pow((float)v[1]-vWarped[1],2) 
						+ pow((float)v[2]-vWarped[2],2))
					    + (coh+cohWarped)/2.0*abs(d-dWarped));
				}else{
					float fWarped = imgIn.at<float>(iterV, iterInnerW);
					float dWarped = imgDepth.at<float>(iterV, iterInnerW);
					float cohWarped = imgCoh.at<float>(iterV, iterInnerW);
					nCurrentCost = SQRT3*(abs(fv-fWarped)+(coh+cohWarped)/2.0*abs(d-dWarped));
				}
				if (nCurrentCost<nCostMin){
					nCostMin = nCurrentCost;
					nCostMinIdx = iterInnerW;
				}
				if (nCostMinIdx<0 || nCostMinIdx>=W){
					nCostMinIdx = iterW;
				}
			}//end loop for the small win
			costList.push_back(nCostMin);
		}

		// choose the lowest from lower,middle, upper part of cost
		float fCostSum1(0), fCostSum2(0), fCostSum3(0);
		int nTakenViews = costList.size();
		for (int iterV=0; iterV<nTakenViews; iterV++){
			if (iterV<nTakenViews/2){
				fCostSum1 += costList.at(iterV);
			}
			if (iterV>nTakenViews/2){
				fCostSum3 += costList.at(iterV);
			}
			if (iterV>=nTakenViews/4 && iterV<=(nTakenViews/4+nTakenViews/2)){
				fCostSum2 += costList.at(iterV);
			}
		}
		imgCost.at<float>(0, iterW) = min3f(fCostSum1, fCostSum2, fCostSum3);
		imgCost.at<float>(0, iterW) /= (nTakenViews/2);

		// alternatively, sum all the cost
		/*float fCostSum = 0;
		for (int iterV=0; iterV<costList.size(); iterV++){
			fCostSum += costList.at(iterV);
		}
		imgCost.at<float>(0, iterW) = fCostSum/(costList.size());*/
		imgTest.at<float>(0, iterW) = imgCost.at<float>(0, iterW);
	}

	// cmpt penalty coef
	//float alpha(0.1),beta(0.05); // if color information is in [0 1]
	float alpha(5.0), beta(1);
	cv::Mat imgPenalty(1, W, CV_32F);
	for(int iterW=0; iterW<W; iterW++){
		//imgPenalty.at<float>(0, iterW) = fSigma/(fSigma+imgCost.at<float>(0, iterW));
		//imgPenalty.at<float>(0, iterW) =  1-1/(1+exp((alpha-imgCost.at<float>(0, iterW))/beta));
		//the following one is better!
		imgPenalty.at<float>(0, iterW) =  1/(1+exp((imgCost.at<float>(0, iterW)-alpha)/beta));
		if (imgPenalty.at<float>(0, iterW) <0 ){
			imgPenalty.at<float>(0, iterW) = 0;
		}
	}
	
	// revise coherence
	imgCohRevised.create(1, W, CV_32F);
	for(int iterW=0; iterW<W; iterW++){
		imgCohRevised.at<float>(0, iterW) =	imgCoh.at<float>(nOptView, iterW)*imgPenalty.at<float>(0, iterW);
	}

	//cout<<imgCost.at<float>(0, 88)<<" "<<imgPenalty.at<float>(0, 88)<<endl;

	return;
}

// use the computed disparity to do shift, which will save some time
// sliceDepth, sliceCoh, sliceCohRevised, sliceCost are 1*W image
// written on May 24th, 2013
void reviseSTNew(const cv::Mat & imgIn, cv::Mat & sliceDepth, cv::Mat & sliceCoh,
	cv::Mat & sliceCohRevised, cv::Mat & sliceCost){
	int nViews = imgIn.rows;
	int W = imgIn.cols;
	int nOptView = nViews/2;
	int C = imgIn.channels();
	sliceCohRevised.create(1, W, CV_32F);
	sliceCost.create(1, W, CV_32F);

	cv::Mat imgDepthTan = -1 * sliceDepth;
	// compute disparity
	cv::Mat disparity = cv::Mat::zeros(nViews, W, CV_32F);
	for (int iterH=0; iterH<nViews; iterH++){
		for (int iterW=0; iterW<W; iterW++){
			disparity.at<float>(iterH, iterW) = imgDepthTan.at<float>(0, iterW)*(nOptView-iterH);
		}
	}
			
	// for debugging
	//ofstream outputDisparity("disparity.txt");
	//for (int iterW=0; iterW<W; iterW++){
	//	outputDisparity<<iterW<<":";
	//	for (int iterH=0; iterH<nViews; iterH++){
	//		outputDisparity<<disparity.at<float>(iterH, iterW)<<" ";
	//	}
	//	outputDisparity<<endl;
	//}

	int nWinRadius = 1;
	int nViewRadius = 4;
	if (nViews/2<nViewRadius)
		nViewRadius = nViews/2;

	for (int iterW=0; iterW<W; iterW++){
		cv::Vec3f v;
		float fv;
		if (C==3){
			v = imgIn.at<cv::Vec3f>(nOptView, iterW);
		}else{
			fv = imgIn.at<float>(nOptView, iterW);
		}
		float d = sliceDepth.at<float>(0, iterW);
		float coh = sliceCoh.at<float>(0, iterW);
		// if the trunctate coh step is moved before this step,
		// then by jump around coh=0, we can save lot of time
		// now we can do 0.95, since the truncate threshold is 0.95
		//if (coh < lowCohThresh){
		//	sliceCost.at<float>(0, iterW) = 0;
		//	continue;
		//}
		vector<float> costList;
		for (int iterV=nViews/2-nViewRadius; iterV<=nViews/2+nViewRadius; iterV++){
			float fDisparity = disparity.at<float>(iterV, iterW);
			int nXLow = floor(fDisparity) + iterW;
			int nXUp = ceil(fDisparity) + iterW;
			int nCostLow(0), nCostUp(0), nCost(0);
			if (nXLow<0 || nXLow>=W){
				nCostLow = nOcclusionCost;	
			}else if (C==3){
				cv::Vec3f vWarped = imgIn.at<cv::Vec3f>(iterV, nXLow);
				nCostLow = sqrt(pow((float)v[0]-vWarped[0],2) 
					+ pow((float)v[1]-vWarped[1],2) 
					+ pow((float)v[2]-vWarped[2],2));
			}else{
				float fWarped = imgIn.at<float>(iterV, nXLow);
				nCostLow = SQRT3*abs(fv-fWarped);
			}
			if (nXUp<0 || nXUp>=W){
				nCostUp = nOcclusionCost;	
			}else if (C==3){
				cv::Vec3f vWarped = imgIn.at<cv::Vec3f>(iterV, nXUp);
				nCostUp = sqrt(pow((float)v[0]-vWarped[0],2) 
					+ pow((float)v[1]-vWarped[1],2) 
					+ pow((float)v[2]-vWarped[2],2));
			}else{
				float fWarped = imgIn.at<float>(iterV, nXUp);
				nCostUp = SQRT3*abs(fv-fWarped);
			}
			if (floor(fDisparity) == ceil(fDisparity))
				nCost = nCostLow;
			else
				nCost = nCostLow*abs(fDisparity-floor(fDisparity))+nCostUp*abs(fDisparity-ceil(fDisparity));
			costList.push_back(nCost);
		}// end loop of current pixel
		//temporal selection !!! ---------- can (and should) be improved 
		// choose the lowest from lower,middle, upper part of cost
		//float fCostSum1(0), fCostSum2(0), fCostSum3(0);
		//int nTakenViews = costList.size();
		//for (int iterV=0; iterV<nTakenViews; iterV++){
		//	if (iterV<nTakenViews/2){
		//		fCostSum1 += costList.at(iterV);
		//	}
		//	if (iterV>nTakenViews/2){
		//		fCostSum3 += costList.at(iterV);
		//	}
		//	if (iterV>=nTakenViews/4 && iterV<=(nTakenViews/4+nTakenViews/2)){
		//		fCostSum2 += costList.at(iterV);
		//	}
		//}
		//sliceCost.at<float>(0, iterW) = min3f(fCostSum1, fCostSum2, fCostSum3);
		//sliceCost.at<float>(0, iterW) /= (nTakenViews/2);
		sliceCost.at<float>(0, iterW) = 0;
		int nTakenViews = costList.size();
		for (int iterV=0; iterV<nTakenViews; iterV++){
			sliceCost.at<float>(0, iterW) += costList.at(iterV);
		}
		sliceCost.at<float>(0, iterW) /= nTakenViews;
	}// end loop for different views

	// cmpt penalty coef
	//float alpha(50.0), beta(1);
	cv::Mat imgPenalty(1, W, CV_32F);
	for(int iterW=0; iterW<W; iterW++){
		//imgPenalty.at<float>(0, iterW) = fSigma/(fSigma+imgCost.at<float>(0, iterW));
		//imgPenalty.at<float>(0, iterW) =  1-1/(1+exp((alpha-imgCost.at<float>(0, iterW))/beta));
		//the following one is better!
		imgPenalty.at<float>(0, iterW) =  1/(1+exp((sliceCost.at<float>(0, iterW)-alpha)/beta));
		if (imgPenalty.at<float>(0, iterW) <0 ){
			imgPenalty.at<float>(0, iterW) = 0;
		}
	}
	
	// revise coherence
	for(int iterW=0; iterW<W; iterW++){ 
		sliceCohRevised.at<float>(0, iterW) = sliceCoh.at<float>(0, iterW)*imgPenalty.at<float>(0, iterW);
	}

	return;
}