
#include <iostream>
#include <math.h>
#include <list>
#include <vector>
#include <fstream>

#include "imageOp.h"
#include "basic.h"

using namespace std;

const float PIE = 3.1415926;

// save an image
void saveImage(const std::string path, const cv::Mat img, bool bNormalize)
{
	if (img.empty()){
		return;
	}
	cv::Mat imgUsing = img.clone();

  	if (bNormalize && img.depth()==CV_32F && img.channels()==1){
		cv::normalize(img, imgUsing, 0, 255, cv::NORM_MINMAX);
  	}else if(bNormalize && img.depth()==CV_8U && img.channels()==1){
		cv::normalize(img, imgUsing, 0, 255, cv::NORM_MINMAX);
	}

  	imwrite(path, imgUsing);
}

void saveStretchedImage(const std::string path, const cv::Mat img, const int nScale, bool bNormalize)
{
	if(img.empty() || nScale==0)
		return;

	cv::Mat imgBigger;
	if (img.depth()==CV_32F && img.channels()==1){
		imgBigger = cv::Mat(img.rows*nScale, img.cols, CV_32F);
		for (int iterR=0; iterR<imgBigger.rows; iterR++){
			for (int iterC=0; iterC<imgBigger.cols; iterC++){
				imgBigger.at<float>(iterR, iterC) = 
					img.at<float>(iterR/nScale, iterC);
			}
		}
		saveImage(path, imgBigger, bNormalize);
	}else if(img.depth()==CV_32F && img.channels()==3){
		imgBigger = cv::Mat(img.rows*nScale, img.cols, CV_32FC3);
		for (int iterR=0; iterR<imgBigger.rows; iterR++){
			for (int iterC=0; iterC<imgBigger.cols; iterC++){
				imgBigger.at<cv::Vec3f>(iterR, iterC) = 
					img.at<cv::Vec3f>(iterR/nScale, iterC);
			}
		}
		saveImage(path, imgBigger, bNormalize);
	}else if (img.depth()==CV_8U && img.channels()==1){
		imgBigger = cv::Mat(img.rows*nScale, img.cols, CV_8U);
		for (int iterR=0; iterR<imgBigger.rows; iterR++){
			for (int iterC=0; iterC<imgBigger.cols; iterC++){
				imgBigger.at<uchar>(iterR, iterC) = 
					img.at<uchar>(iterR/nScale, iterC);
			}
		}
		saveImage(path, imgBigger, bNormalize);
	}else if (img.depth()==CV_8U && img.channels()==3){
		imgBigger = cv::Mat(img.rows*nScale, img.cols, CV_8UC3);
		for (int iterR=0; iterR<imgBigger.rows; iterR++){
			for (int iterC=0; iterC<imgBigger.cols; iterC++){
				cv::Vec3b v = img.at<cv::Vec3b>(iterR/nScale, iterC);
				imgBigger.at<cv::Vec3b>(iterR, iterC) = v;
			}
		}
		saveImage(path, imgBigger, bNormalize);
	}
}

// show an image
void showImage(const cv::Mat img, const std::string strWinName, bool bNormalized)
{
	if(img.empty())
		return;

	cv::Mat imgUsing = img.clone();

  	if (bNormalized && img.depth()==CV_32F && img.channels() ==1){
		cv::normalize(img, imgUsing, 0, 1, cv::NORM_MINMAX);
  	}else if(bNormalized && img.depth()==CV_8U && img.channels() == 1){
		cv::normalize(img, imgUsing, 0, 255, cv::NORM_MINMAX);
	}

    cv::namedWindow(strWinName, CV_WINDOW_AUTOSIZE);
    cv::imshow(strWinName, imgUsing);
//	cv::waitKey();	
	return;
}

// show image in a stretched way (vertical dimension)
void showStretchedImage(const cv::Mat img, const std::string strWinName, 
	const int nScale, bool bNormalized)
{
	if(img.empty() || nScale==0)
		return;

	cv::Mat imgBigger;

	if (img.depth()==CV_32F && img.channels()==1){
		imgBigger = cv::Mat(img.rows*nScale, img.cols, CV_32F);
		for (int iterR=0; iterR<imgBigger.rows; iterR++){
			for (int iterC=0; iterC<imgBigger.cols; iterC++){
				imgBigger.at<float>(iterR, iterC) = 
					img.at<float>(iterR/nScale, iterC);
			}
		}
		showImage(imgBigger, strWinName, bNormalized);
	}else if(img.depth()==CV_32F && img.channels()==3){
		imgBigger = cv::Mat(img.rows*nScale, img.cols, CV_32FC3);
		for (int iterR=0; iterR<imgBigger.rows; iterR++){
			for (int iterC=0; iterC<imgBigger.cols; iterC++){
				imgBigger.at<cv::Vec3f>(iterR, iterC) = 
					img.at<cv::Vec3f>(iterR/nScale, iterC);
			}
		}
		showImage(imgBigger, strWinName, bNormalized);
	}else if (img.depth()==CV_8U && img.channels()==1){
		imgBigger = cv::Mat(img.rows*nScale, img.cols, CV_8U);
		for (int iterR=0; iterR<imgBigger.rows; iterR++){
			for (int iterC=0; iterC<imgBigger.cols; iterC++){
				imgBigger.at<uchar>(iterR, iterC) = 
					img.at<uchar>(iterR/nScale, iterC);
			}
		}
		showImage(imgBigger, strWinName, bNormalized);
	}else if (img.depth()==CV_8U && img.channels()==3){
		imgBigger = cv::Mat(img.rows*nScale, img.cols, CV_8UC3);
		for (int iterR=0; iterR<imgBigger.rows; iterR++){
			for (int iterC=0; iterC<imgBigger.cols; iterC++){
				cv::Vec3b v = img.at<cv::Vec3b>(iterR/nScale, iterC);
				imgBigger.at<cv::Vec3b>(iterR, iterC) = v;
			}
		}
		showImage(imgBigger, strWinName, bNormalized);
	}
	
}

// for a CV_32F image, change its value from 0-255 to 0-1
void unchar2Float(cv::Mat & img)
{
	assert (img.depth() == CV_32F);
	for (int iterR=0; iterR<img.rows; iterR++){
		for (int iterC=0; iterC<img.cols; iterC++){
			if (img.channels() == 1) {
				img.at<float>(iterR, iterC) = img.at<float>(iterR, iterC)/255.0;
			}else if (img.channels() == 3){
				cv::Vec3f v;
				v[0] = img.at<cv::Vec3f>(iterR, iterC)[0]/255.0;
				v[1] = img.at<cv::Vec3f>(iterR, iterC)[1]/255.0;
				v[2] = img.at<cv::Vec3f>(iterR, iterC)[2]/255.0;
				img.at<cv::Vec3f>(iterR, iterC) = v;
			}
		}
	}
}

cv::Mat getFloatImage(cv::Mat & img)
{
	assert (img.depth() == CV_8U && img.channels()==1);
	cv::Mat imgOut(img.size(), CV_32F);
	for (int iterR=0; iterR<img.rows; iterR++){
		for (int iterC=0; iterC<img.cols; iterC++){
			if (img.channels() == 1) {
				imgOut.at<float>(iterR, iterC) = img.at<uchar>(iterR, iterC)/255.0;
			}else if (img.channels() == 3){
				cv::Vec3f v;
				v[0] = img.at<cv::Vec3f>(iterR, iterC)[0]/255.0;
				v[1] = img.at<cv::Vec3f>(iterR, iterC)[1]/255.0;
				v[2] = img.at<cv::Vec3f>(iterR, iterC)[2]/255.0;
				img.at<cv::Vec3f>(iterR, iterC) = v;
			}
		}
	}
	return imgOut;
}

// image merge
void mergeChannel(cv::Mat imgIn, cv::Mat & imgOut, const int method)
{
	
	if (imgIn.channels() == 1){
		imgOut= imgIn.clone();
	}else if(imgIn.channels() == 3){
		imgOut.create(imgIn.size(), imgIn.depth());
		for (int iterR = 0; iterR<imgIn.rows; iterR++){
			for (int iterC=0; iterC<imgIn.cols; iterC++){
				if (imgIn.depth() == CV_8U){
					cv::Vec3b v = imgIn.at<cv::Vec3b>(iterR,iterC);
					imgOut.at<uchar>(iterR, iterC) = sqrt((float)(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]));
				}else if(imgIn.depth() == CV_32F){
					cv::Vec3f v = imgIn.at<cv::Vec3f>(iterR,iterC);
					imgOut.at<float>(iterR, iterC) = sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);
				}
			}
		}
	}
}

// normalize to 0-1 or 0-255
void normalize(cv::Mat & img)
{
	double minVal, maxVal;
	minMaxLoc3D(img, &minVal, &maxVal);
	for (int iterR=0; iterR<img.rows; iterR++){
		for (int iterC=0; iterC<img.cols; iterC++){
			if (img.depth() == CV_8U){
				img.at<uchar>(iterR, iterC) = (img.at<uchar>(iterR,iterC)-minVal)/(float)(maxVal-minVal)*255;
			}else if(img.depth() == CV_32F){
				img.at<float>(iterR, iterC) = (img.at<float>(iterR,iterC)-minVal)/(float)(maxVal-minVal);
			}
		}
	}
}

/*
void normalize(cv::Mat imgIn, cv::Mat & imgOut)
{
	double minVal, maxVal;
	minMaxLoc3D(imgIn, &minVal, &maxVal);
	imgOut.create(imgIn.size(), imgIn.type());	

	for (int iterR=0; iterR<imgIn.rows; iterR++){
		for (int iterC=0; iterC<imgIn.cols; iterC++){
			if (imgIn.depth() == CV_8U){
				imgOut.at<uchar>(iterR, iterC) = (imgIn.at<uchar>(iterR,iterC)-minVal)/(float)(maxVal-minVal)*255;
			}else if(imgIn.depth() == CV_32F){
				imgOut.at<float>(iterR, iterC) = (imgIn.at<float>(iterR,iterC)-minVal)/(float)(maxVal-minVal);
			}
		}
	}
}*/


// get min and max of an image
void minMaxLoc3D(cv::Mat img, double* min, double* max)
{
	if(img.depth()==CV_8U){
		uchar* dataPtr = img.data;
		uchar dataFirst = *dataPtr;
		*min = *max = dataFirst;
		for (int iterDataPtr=0; iterDataPtr<img.rows*img.cols*img.channels(); iterDataPtr++){
			uchar data = *(dataPtr+iterDataPtr);
			if (data<=*min){
				*min = data;	
			}else if(data>=*max){
				*max = data;
			}
		}
	}else if(img.depth()==CV_32F){
		float* dataPtr = (float*)img.data;
		float dataFirst = *dataPtr;
		*min = *max = dataFirst;
		for (int iterDataPtr=0; iterDataPtr<img.rows*img.cols*img.channels(); iterDataPtr++){
			float data = *(dataPtr+iterDataPtr);
			if (data<=*min){
				*min = data;	
			}else if(data>=*max){
				*max = data;
			}
		}
	}
	return;
}


// clamp values in a certain section
void truncImage(cv::Mat & img, const float fUpBound, const float fLowBound)
{
	assert(img.channels()==1);
	if (img.depth() == CV_32F){
		for (int iterR=0; iterR<img.rows; iterR++){
			for (int iterC=0; iterC<img.cols; iterC++){
				if (img.at<float>(iterR,iterC) > fUpBound){
					img.at<float>(iterR,iterC) = fUpBound;
				}else if(img.at<float>(iterR,iterC) < fLowBound){
					img.at<float>(iterR, iterC) = fLowBound;
				}
			}
		}
	}else if(img.depth() == CV_8U){
		for (int iterR=0; iterR<img.rows; iterR++){
			for (int iterC=0; iterC<img.cols; iterC++){
				if (img.at<uchar>(iterR,iterC) > fUpBound){
					img.at<uchar>(iterR,iterC) = (uchar)fUpBound;
				}else if(img.at<uchar>(iterR,iterC) < fLowBound){
					img.at<uchar>(iterR, iterC) = (uchar)fLowBound;
				}
			}
		}
	}
}


// output histogram
void showHistogram(const cv::Mat img, const float fUpBound, const float fLowBound)
{
	float fUpB = fUpBound;
	float fLowB = fLowBound;
 	if (fUpB == fLowB){
		double minVal, maxVal;
		minMaxLoc3D(img, &minVal, &maxVal);
		cout<<"min Val: "<<minVal<<"; max Val: "<<maxVal<<endl;
		fUpB = maxVal;
		fLowB = minVal;
	}

	
	int nbins = 10;
 	int histSize[] = {nbins};
	float hranges[] = { fLowB-1, fUpB+1 };
	const float* ranges[] = { hranges };
	cv::MatND hist;
    	int channels[] = {0};

    	cv::calcHist( &img, 1, channels, cv::Mat(), // do not use mask
       		hist, 1, histSize, ranges,
        	true, // the histogram is uniform
        	false );
	
	for (int i=0; i<nbins; i++){
		cout<<"bin "<<i<<" :"<< hist.at<float>(i)<<endl;
	}
}

// turn to abstract value
void getAbs(cv::Mat imgIn, cv::Mat & imgOut)
{
	imgOut = imgIn.clone();
	if(imgIn.depth()==CV_8U){
		// to be implemented
	}else if(imgIn.depth()==CV_32F){
		float* dataPtr = (float*)imgOut.data;
		for (int iterDataPtr=0; iterDataPtr<imgIn.rows*imgIn.cols*imgIn.channels(); iterDataPtr++){
			float data = *(dataPtr+iterDataPtr);
			if (data<=0){
				*(dataPtr+iterDataPtr) = data*-1;
			}
		}
	}
}

