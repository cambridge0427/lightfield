
#include "structureTensorVigra.h"

using namespace std;
using namespace vigra;

#include <fstream>

const float PIE = 3.1415926;

// my own implementation of bilateral filtering
template <class SrcIterator, class SrcAccessor,
class DestIterator, class DestAccessor>
	void bilateralSmoothing(triple<SrcIterator, SrcIterator, SrcAccessor> src,
	pair<DestIterator, DestAccessor> dest,
	double scaleForGauss=5, double scaleForSimilarity=0.005)
{
	// for debugging
	ofstream fOutput("test.txt");

	typedef typename SrcIterator::value_type SrcType;
	typedef typename NumericTraits<SrcType>::Promote SumType;

	int nWindowSize = 7; // ad-hoc, should be defined in a better way

	// calculate gaussian coefficient
	// in this case probably a larger sigma is better
	SrcIterator iSrcStart = src.first;
	SrcIterator iSrcEnd = src.second;
	DestAccessor iDestAccessor = dest.second;

	DestIterator iDestPixY, iDestPixX;
	SrcIterator iSrcPixY, iSrcPixX;
	
	iDestPixY = dest.first;
	iSrcPixY = iSrcStart;
	int x, y = 0; 
	for(; iSrcPixY.y < iSrcEnd.y;  ++iSrcPixY.y, ++iDestPixY.y, y++) { 
		iDestPixX = iDestPixY;
		iSrcPixX = iSrcPixY;
		x = 0;
		for(; iSrcPixX.x < iSrcEnd.x;  ++iSrcPixX.x, ++iDestPixX.x, x++)  { 
			SumType lSummation = 0;		//for normalization
			float fSrcOrigValue = *iSrcPixX;
			
			// inner loop : local neighborhood
			for(int iterH=-nWindowSize/2; iterH<+nWindowSize/2; iterH++){
				if (iSrcPixY.y+iterH<iSrcStart.y || iSrcPixY.y+iterH>=iSrcEnd.y){
					continue;
				}
				for (int iterW=-nWindowSize/2; iterW<nWindowSize/2; iterW++){
					if (iSrcPixX.x+iterW<iSrcStart.x || iSrcPixX.x+iterW>=iSrcEnd.x){
						continue;
					}
					float fCoefGaussian = exp(-0.5*(iterH*iterH+iterW*iterW)/(scaleForGauss*scaleForGauss)) ;
					SrcIterator iterCmpPix = iSrcPixX + Dist2D(iterW, iterH);
					float fCompIntensity = *iterCmpPix;
					float fRefIntensity =  *iSrcPixX; 
					float fCoefSimilarity = 
						exp(-0.5*(fCompIntensity-fRefIntensity)*(fCompIntensity-fRefIntensity)
						/(scaleForSimilarity*scaleForSimilarity)) ;
					//fCoefSimilarity = 1.0;
					lSummation += fCoefGaussian*fCoefSimilarity;
				}
			}	//end of inner loop


			if (y == (iSrcEnd.y-iSrcStart.y)/2){
				fOutput<<"("<<x<<","<<y<<") summation:"<<lSummation<<endl;
			}
				
			float fFilteredResult = 0.0;
			// inner loop again
			for(int iterH=-nWindowSize/2; iterH<+nWindowSize/2; iterH++){
				if (iSrcPixY.y+iterH<iSrcStart.y || iSrcPixY.y+iterH>=iSrcEnd.y){
					continue;
				}
				for (int iterW=-nWindowSize/2; iterW<nWindowSize/2; iterW++){
					if (iSrcPixX.x+iterW<iSrcStart.x || iSrcPixX.x+iterW>=iSrcEnd.x){
						continue;
					}
					float fCoefGaussian = exp(-0.5*(iterH*iterH+iterW*iterW)/(scaleForGauss*scaleForGauss));
					SrcIterator iterCmpPix = iSrcPixX + Dist2D(iterW, iterH);
					float fCompIntensity = *iterCmpPix;
					float fRefIntensity =  *iSrcPixX; 
					float fCoefSimilarity = 
						exp(-0.5*(fCompIntensity-fRefIntensity)*(fCompIntensity-fRefIntensity)
						/(scaleForSimilarity*scaleForSimilarity)) ;
					
					//fCoefSimilarity = 1.0;
					float fCoef = fCoefGaussian*fCoefSimilarity/lSummation;
					fFilteredResult += fCoef*fCompIntensity;

					if (y == (iSrcEnd.y-iSrcStart.y)/2){
						fOutput<<"Gaussian:"<<fCoefGaussian<<" Similarity:"<<fCoefSimilarity
							<<" ColorDist:"<<abs(fCompIntensity-fRefIntensity)<<"coef:"<<fCoef<<endl;
					}
				}
			}
			iDestAccessor.set(fSrcOrigValue, iDestPixX);
			//*iDestPixX = fFilteredResult;
        }
	}
	fOutput.close();
	//dest.second.set(float, dest.first);
}

void STwithVigra(const cv::Mat imgIn, cv::Mat & imgDepth, cv::Mat & imgCoh,
	double fInner, double fOuter)
{
	int nWidth = imgIn.cols;
	int nHeight = imgIn.rows;
	FImage src(nWidth,nHeight), stxx(nWidth,nHeight), stxy(nWidth,nHeight), styy(nWidth,nHeight);

	FImage::traverser end = src.lowerRight();
	//cout<<end.x<<" "<<end.y<<endl;
	int nCntW(0), nCntH(0);
	for(FImage::traverser iPixY = src.upperLeft();  iPixY.y < end.y;  ++iPixY.y, nCntH++) 
    { 
		nCntW = 0;
		for(FImage::traverser iPixX = iPixY;  iPixX.x < end.x;  ++iPixX.x, nCntW++)  
        { 
			if (imgIn.channels() == 1){
				*iPixX = (float)imgIn.at<float>(nCntH, nCntW);
			}else{
				cv::Vec3f v = imgIn.at<cv::Vec3f>(nCntH, nCntW);
				*iPixX = (float)(v[0]+v[1]+v[2]);
				//*iPixX = (float)v[0];
			}
        } 
	}

	//===============================
	// use the code instead of calling "structure tensor"
	float fOuterG(1), fOuterS(0.005);
	bool bBilateral = false;
	FImage tmpx(nWidth, nHeight), tmpy(nWidth, nHeight), tmp(nWidth, nHeight);
	gaussianGradient(srcImageRange(src), destImage(tmpx), destImage(tmpy), fInner);
	combineTwoImages(srcImageRange(tmpx), srcImage(tmpx),
		destImage(tmp), std::multiplies<float>());
	if (!bBilateral){
		gaussianSmoothing(srcImageRange(tmp),
			destImage(stxx), fOuter);
	}else{
		bilateralSmoothing(srcImageRange(tmp), destImage(stxx));
	}
	
	combineTwoImages(srcImageRange(tmpy), srcImage(tmpy),
		destImage(tmp), std::multiplies<float>());
	if (!bBilateral){
		gaussianSmoothing(srcImageRange(tmp),
			destImage(styy), fOuter);
	}else{
		bilateralSmoothing(srcImageRange(tmp), destImage(styy));
	}

	combineTwoImages(srcImageRange(tmpx), srcImage(tmpy),
		destImage(tmp), std::multiplies<float>());
	if (!bBilateral){
		gaussianSmoothing(srcImageRange(tmp),
			destImage(stxy), fOuter);
	}else{
		bilateralSmoothing(srcImageRange(tmp), destImage(stxy));
	}
	// ================================

	end = stxx.lowerRight();
	vigra::FImage::traverser istxx_x, istxy_x, istyy_x;
	vigra::FImage::traverser istxx_y, istxy_y, istyy_y;
	nCntH = 0; nCntW = 0;
	float fThreshold = 0;
	imgDepth.create(nHeight, nWidth, CV_32F);
	imgCoh.create(nHeight, nWidth, CV_32F);
	for (istxx_y=stxx.upperLeft(), istxy_y=stxy.upperLeft(), istyy_y=styy.upperLeft();
		istxx_y.y<end.y; 
		istxx_y.y++, istxy_y.y++, istyy_y.y++, nCntH++){
		nCntW = 0;
		for (istxx_x = istxx_y, istxy_x = istxy_y, istyy_x = istyy_y;
			istxx_x.x<end.x;
			istxx_x.x++, istxy_x.x++, istyy_x.x++, nCntW++){
				imgDepth.at<float>(nCntH, nCntW) = 
					0.5*atan2((float)(2*(*istxy_x)), (float)(*istyy_x-*istxx_x));
			if(abs(*istxx_x+*istyy_x) > fThreshold){
				imgCoh.at<float>(nCntH, nCntW) = 
					sqrt(pow(*istyy_x-*istxx_x,2)+ pow(*istxy_x,2)*4)/abs(*istxx_x+*istyy_x);
			}else{
				imgCoh.at<float>(nCntH, nCntW) = 0;
			//	nCntPts++;
			}
			//imgTest.at<float>(iterY, iterX) = imgDepth.at<float>(iterY, iterX);
			
			// the angle detected is between the line and the horizontal line (to right)
			// the original range is (-pie/2, pie/2)
			// now it is shifted to (0, pie)
			// however, the value of tan does not chage
			//if (imgDepth.at<float>(nCntH, nCntW)<0){
			//	imgDepth.at<float>(nCntH, nCntW) += PIE;
			//}
		}
	}
}


// old version that called "structure tensor"
//void STwithVigra(const cv::Mat imgIn, cv::Mat & imgDepth, cv::Mat & imgCoh,
//	double fInner, double fOuter)
//{
//	int nWidth = imgIn.cols;
//	int nHeight = imgIn.rows;
//	vigra::FImage src(nWidth,nHeight), stxx(nWidth,nHeight), stxy(nWidth,nHeight), styy(nWidth,nHeight);
//
//	vigra::FImage::traverser end = src.lowerRight();
//	//cout<<end.x<<" "<<end.y<<endl;
//	int nCntW(0), nCntH(0);
//	for(vigra::FImage::traverser iPixY = src.upperLeft();  iPixY.y < end.y;  ++iPixY.y, nCntH++) 
//    { 
//		nCntW = 0;
//		for(vigra::FImage::traverser iPixX = iPixY;  iPixX.x < end.x;  ++iPixX.x, nCntW++)  
//        { 
//			if (imgIn.channels() == 1){
//				*iPixX = (float)imgIn.at<float>(nCntH, nCntW);
//			}else{
//				cv::Vec3f v = imgIn.at<cv::Vec3f>(nCntH, nCntW);
//				*iPixX = (float)(v[0]+v[1]+v[2]);
//				//*iPixX = (float)v[0];
//			}
//        } 
//	}
//
//	//exportImage(srcImageRange(src), vigra::ImageExportInfo("try.gif"));
//    // calculate Structure Tensor at inner scale = 1.0 and outer scale = 3.0
//    vigra::structureTensor(srcImageRange(src),
//        destImage(stxx), destImage(stxy), destImage(styy), fInner, fOuter);
//	
//	end = stxx.lowerRight();
//	vigra::FImage::traverser istxx_x, istxy_x, istyy_x;
//	vigra::FImage::traverser istxx_y, istxy_y, istyy_y;
//	nCntH = 0; nCntW = 0;
//	float fThreshold = 0;
//	imgDepth.create(nHeight, nWidth, CV_32F);
//	imgCoh.create(nHeight, nWidth, CV_32F);
//	for (istxx_y=stxx.upperLeft(), istxy_y=stxy.upperLeft(), istyy_y=styy.upperLeft();
//		istxx_y.y<end.y; 
//		istxx_y.y++, istxy_y.y++, istyy_y.y++, nCntH++){
//		nCntW = 0;
//		for (istxx_x = istxx_y, istxy_x = istxy_y, istyy_x = istyy_y;
//			istxx_x.x<end.x;
//			istxx_x.x++, istxy_x.x++, istyy_x.x++, nCntW++){
//				imgDepth.at<float>(nCntH, nCntW) = 
//					0.5*atan2((float)(2*(*istxy_x)), (float)(*istyy_x-*istxx_x));
//			if(abs(*istxx_x+*istyy_x) > fThreshold){
//				imgCoh.at<float>(nCntH, nCntW) = 
//					sqrt(pow(*istyy_x-*istxx_x,2)+ pow(*istxy_x,2)*4)/abs(*istxx_x+*istyy_x);
//			}else{
//				imgCoh.at<float>(nCntH, nCntW) = 0;
//			//	nCntPts++;
//			}
//			//imgTest.at<float>(iterY, iterX) = imgDepth.at<float>(iterY, iterX);
//			if (imgDepth.at<float>(nCntH, nCntW)<0){
//				imgDepth.at<float>(nCntH, nCntW) += PIE;
//			}
//		}
//	}
//}