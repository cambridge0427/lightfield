
#include <iostream>
#include <ctime>

#include "opencv2/imgproc/imgproc.hpp"

#include "../common/fileOp.h"
#include "../common/imageOp.h"

#include "lightfield.h"
#include "structureTensorOpenCV.h"
#include "structureTensorVigra.h"

using namespace std;

const string lightfield::LFDatasetName = "LF";
const string lightfield::GTDatasetName = "GT_DEPTH";
const string lightfield::DepthDatasetName = "Depth";
const string lightfield::CenterViewDatasetName = "CenterView";

const double lightfield::dInner = 0.8;
const double lightfield::dOuter = 0.8;
const double lightfield::dLowCohThresh = 0.95;

const bool lightfield::bCohImgBlur = true;

#define PIE 3.1415926

// constructor
lightfield::lightfield(const string & H5Filename, bool bSwapCh):
	bEmpty(true),
	bSwapChannel(bSwapCh),
	bFlip(false),
	LFData(NULL),
	fMinGT(-3.0),
	fMaxGT(3.0)
{
	if (openH5File(H5Filename)){
		bEmpty = false;
		loadAttributes();
	}else {
		bEmpty = true;
	}
	return;
}

lightfield::lightfield():
	bEmpty(true),
	bSwapChannel(false),
	bFlip(false),
	LFData(NULL)
{
}

// destructor
lightfield::~lightfield()
{
	if(bOpen){
		H5Fclose(fileID);	
	}
	if(LFData!=NULL){
		delete[] LFData;
	}
}

void lightfield::releaseLFData()
{
	if(LFData!=NULL){
		delete[] LFData;
		LFData = NULL;
	}
}


//--------------- fetch data------------------------------//
cv::Mat lightfield::getViewImage()
{
	size_t S(sRes), T(tRes), H(yRes), W(xRes), C(nChannels);	
	// Open the light field data sets
	hid_t dset = H5Dopen (fileID, CenterViewDatasetName.c_str(), H5P_DEFAULT);
	if ( dset <= 0 ) {
		cout<< "could not open light field data set '" << CenterViewDatasetName << "'" << endl;
		cout<<"fetch from light field data!"<<endl;	
		if (LFData == NULL && loadLFData()==false){
			cout<<"LF data loading failed!"<<endl;
			return cv::Mat();
		}

		// get center view
		uchar *data3d = new uchar[W*H*C];
		hsize_t nCenterS = S/2;
		hsize_t nCenterT = T/2;
		hsize_t nOffSet = nCenterS*T*W*H*C + nCenterT*W*H*C;
		memcpy(data3d, LFData + nOffSet, W*H*C);

		// generate cv image
		if (C == 3){
			cv::Mat viewImg = cv::Mat(H, W, CV_8UC3, data3d);
			//viewImg = viewImg/255;
			return viewImg;
		}else if (C==1){
			cv::Mat viewImg = cv::Mat(H, W, CV_8U, data3d);
			//viewImg = viewImg/255;
			return viewImg;
		}else{
			return cv::Mat();
		}
	}else{
		uchar *data3d = new uchar[W*H*C];
		hid_t dset = H5Dopen (fileID, CenterViewDatasetName.c_str(), H5P_DEFAULT);
		herr_t status = H5Dread(dset, H5T_NATIVE_UCHAR, H5S_ALL, H5S_ALL, H5P_DEFAULT, data3d);
		cout<<"finish loading data!"<<endl;
		H5Dclose( dset );
		cv::Mat centerViewImage = cv::Mat(H, W, CV_8UC3, data3d);
		return centerViewImage.clone();
	}
}

cv::Mat lightfield::getGroundTruth()
{
	if (!m_imgGT.empty())
		return m_imgGT;

	string dset_name = GTDatasetName;
	// Open the light field data sets
	hid_t dset = H5Dopen (fileID, dset_name.c_str(), H5P_DEFAULT);
	if ( dset <= 0 ) {
		cout<< "could not open light field data set '" << dset_name << "'" << endl;
		return cv::Mat();
	}
	cout<< "  data set '" << dset_name << "' opened." << endl;

	// Retrieve dimension attributes
	float *data2d = NULL;
	int ndims;
	size_t W, H;
	H5LTget_dataset_ndims(fileID, dset_name.c_str(), &ndims );
	if(ndims==2){
		hsize_t dims[2];
		H5LTget_dataset_info(fileID, dset_name.c_str(), dims, NULL, NULL);
		W = dims[1];
		H = dims[0];
		cout<< "  data set '" << dset_name <<  " views, " << W << " x " << H << endl;
		if (W!=xRes || H!=yRes){
			cout<<"resolution does not match!"<<endl;
			return cv::Mat();
		}
		// create buffer for light field data
		data2d = new float[ W*H];
		H5LTread_dataset_float( fileID, dset_name.c_str(), data2d );
	}else if (ndims==4){
		hsize_t dims[4];
		H5LTget_dataset_info(fileID, dset_name.c_str(), dims, NULL, NULL);
		W = dims[3];
		H = dims[2];
		size_t S = dims[1];
		size_t T = dims[0];
		if (W!=xRes || H!=yRes){
			cout<<"resolution does not match!"<<endl;
			return cv::Mat();
		}
		// create buffer for light field data
		data2d = new float[ W*H];
		float *data4d = new float[S*T*W*H];
		if (data4d == NULL){
			cout<<"failed to allocate memory!"<<endl;
			return cv::Mat();
		}
		H5LTread_dataset_float( fileID, dset_name.c_str(), data4d );
		int nOffset = T/2*S*H*W + S/2*H*W;
		memcpy(data2d, data4d+nOffset,sizeof(float)*H*W );
		delete[] data4d;
	}else {
		cout<< "light field data set '" << dset_name << "' is not two or four dimensional." << endl;
		return cv::Mat();
	}
	H5Dclose( dset );

	if (data2d!=NULL){
		//generate cv image
		cv::Mat groundTruthImg = cv::Mat(H, W, CV_32F, data2d);
		cv::Mat groundTruthImg2 = groundTruthImg.clone();

		// disparity ->depth
		for (int iterH=0; iterH<H; iterH++){
			for (int iterW=0; iterW<W; iterW++){
				groundTruthImg2.at<float>(iterH, iterW) = 
				(dH*xRes)/(2*tan(fFocalLength/2))
				*(1/(groundTruthImg.at<float>(iterH, iterW))-1/fCamDistance);
			}
		}

		//showHistogram(groundTruthImg);
		// see range
		double minVal, maxVal;
		cv::minMaxLoc(groundTruthImg2, &minVal, &maxVal);
		fMinGT = minVal;
		fMaxGT = maxVal;
		
		delete[] data2d;
		m_imgGT = groundTruthImg2;
		return groundTruthImg2;
	}else
		return cv::Mat();
}

cv::Mat lightfield::getDepthMap()
{
	cv::Mat depthImg, depthImg2;
	
	string dset_name = DepthDatasetName;
	// Open the light field data sets
	hid_t dset = H5Dopen (fileID, dset_name.c_str(), H5P_DEFAULT);
	if ( dset <= 0 ) {
		cout<< "could not open light field data set '" << dset_name << "'"<<endl;
		return cv::Mat();
	}
	cout<< "  data set '" << dset_name << "' opened." << endl;

	// Retrieve dimension attributes
	int ndims;
	H5LTget_dataset_ndims(fileID, dset_name.c_str(), &ndims );
	if ( ndims != 2 ) {
		cout<< "light field data set '" << dset_name << "' is not two-dimensional." << endl;
		return cv::Mat();
	}

	hsize_t dims[2];
	H5LTget_dataset_info(fileID, dset_name.c_str(), dims, NULL,NULL);
	size_t W = dims[1];
	size_t H = dims[0];
	cout<< "  data set '" << dset_name <<  " views, " << W << " x " << H << endl;
	if (W!=xRes || H!=yRes){
		cout<<"resolution does not match!"<<endl;
		return cv::Mat();
	}

	// create buffer for light field data
	float *data2d = new float[ W*H];
	H5LTread_dataset_float( fileID, dset_name.c_str(), data2d );
	H5Dclose( dset );

	depthImg = cv::Mat(H, W, CV_32F, data2d);
	// see range
	double minVal, maxVal;
	cv::minMaxLoc(depthImg, &minVal, &maxVal);
	cout<<"range of depth image:"<<minVal<<"to"<<maxVal<<endl;
	// rescale to 0-255
	//depthImg = (depthImg-minVal)/(maxVal-minVal)*255;
	for (int iterC=0; iterC<depthImg.cols; iterC++){
		for (int iterR=0; iterR<depthImg.rows; iterR++){
			depthImg.at<float>(iterR, iterC) = (depthImg.at<float>(iterR, iterC) - minVal)/(maxVal-minVal)*255;
		}
	}
	depthImg.convertTo(depthImg2, CV_8U);

	delete[] data2d;
	return depthImg2;
}

void lightfield::getViewImage(int nt, int ns, cv::Mat & img)
{
	size_t H(yRes), W(xRes), T(tRes), S(sRes), C(nChannels);

	if (LFData == NULL && loadLFData()==false){
		cout<<"LF data loading failed!"<<endl;
		return;
	}

	if (C==1){
		img.create(H, W, CV_8U);
		int nOffset = nt*S*H*W*C + ns*H*W*C;
		for (int iterH=0; iterH<H; iterH++){
			for (int iterW=0; iterW<W; iterW++){
				img.at<uchar>(iterH, iterW) = LFData[nOffset++];
			}
		}
	}else{
		img.create(H, W, CV_8UC3);
		int nOffset = nt*S*H*W*C + ns*H*W*C;
		for (int iterH=0; iterH<H; iterH++){
			for (int iterW=0; iterW<W; iterW++){
				cv::Vec3b v;
				v[0] = LFData[nOffset++];
				v[1] = LFData[nOffset++];
				v[2] = LFData[nOffset++];
				img.at<cv::Vec3b>(iterH, iterW) = v;
			}
		}
	}
	return;
}

bool lightfield::getGTMinFromH5(float & minVal){
	herr_t  bGot = H5LTget_attribute_float (fileID, "/", "minGT", &minVal);
	if (bGot>=0){
		fMinGT = (dH*xRes)/(2*tan(fFocalLength/2))*(1/minVal-1/fCamDistance);
		return true;
	}else
		return false;
}

bool lightfield::getGTMaxFromH5(float & maxVal){
	herr_t  bGot = H5LTget_attribute_float (fileID, "/", "maxGT", &maxVal);
	if (bGot>=0){
		fMaxGT = (dH*xRes)/(2*tan(fFocalLength/2))*(1/maxVal-1/fCamDistance);
		return true;
	}else
		return false;
}

void lightfield::getYTSlice(const int nx, const int ns, cv::Mat & imgSlice)
{
	size_t S(sRes), T(tRes), H(yRes), W(xRes), C(nChannels);	
	if (LFData == NULL && loadLFData()==false){
		cout<<"LF data loading failed!"<<endl;
		return;
	}
	uchar* FiveDData = LFData;
	if (FiveDData==NULL) return;

	if (C==1){
		imgSlice.create(T, H, CV_32F);
		int nOffSet = 0;
		for (int iterT=0; iterT<T; iterT++){
			nOffSet = iterT*S*H*W*C + ns*H*W*C + nx*C;
			for (int iterY=0; iterY<H; iterY++){
				float tmp = 0;
				tmp = *(FiveDData+nOffSet);
				imgSlice.at<float>(iterT,iterY) = tmp;
				nOffSet+=W*C;
			}
			nOffSet += W*H*C;
		}
	}else {
		imgSlice.create(T, H, CV_32FC3);
		int nOffSet = 0;
		for (int iterT=0; iterT<T; iterT++){
			nOffSet = iterT*S*H*W*C + ns*H*W*C + nx*C;
			for (int iterY=0; iterY<H; iterY++){
				cv::Vec3f v;
				v[0] = *(FiveDData+nOffSet);
				v[1] = *(FiveDData+nOffSet+1);
				v[2] = *(FiveDData+nOffSet+2);
				imgSlice.at<cv::Vec3f>(iterT,iterY) = v;
				nOffSet+=W*C;
			}
			nOffSet += W*H*C;
		}
	}
}
	
void lightfield::getXSSlice(const int ny, const int nt, cv::Mat & imgSlice)
{
	size_t S(sRes), T(tRes), H(yRes), W(xRes), C(nChannels);	
	if (LFData == NULL && loadLFData()==false){
		cout<<"LF data loading failed!"<<endl;
		return;
	}
	uchar* FiveDData = LFData;
	if (FiveDData==NULL) return;

if (C==1){
	imgSlice.create(S, W, CV_32F);
	int nOffSet = 0;
	for (int iterS=0; iterS<S; iterS++){
		nOffSet = nt*S*W*H*C + ny*W*C + iterS*W*H*C;
		for (int iterX=0; iterX<W; iterX++){
			float tmp = 0;
			tmp = *(FiveDData+nOffSet);
			imgSlice.at<float>(iterS,iterX) = tmp;
			nOffSet+=C;
		}
	}
}else{
	imgSlice.create(S, W, CV_32FC3);
	int nOffSet = 0;
	for (int iterS=0; iterS<S; iterS++){
		nOffSet = nt*S*W*H*C + ny*W*C + iterS*W*H*C;
		for (int iterX=0; iterX<W; iterX++){
			cv::Vec3f v;
			v[0] = *(FiveDData+nOffSet);
			v[1] = *(FiveDData+nOffSet+1);
			v[2] = *(FiveDData+nOffSet+2);
			imgSlice.at<cv::Vec3f>(iterS,iterX) = v;
			nOffSet+=C;
		}
	}
}
	return;
}


//---------------depth estimation-------------------------//
void lightfield::getDepthEstimation(cv::Mat & depthS, cv::Mat & cohS, 
		cv::Mat & depthT, cv::Mat & cohT, bool bInverseYT)
{
	clock_t tStart;
	tStart = clock();
	size_t H(yRes), W(xRes), T(tRes), S(sRes), C(nChannels);
	if (LFData == NULL && loadLFData()==false){
		cout<<"LF data loading failed!  "<<(clock()-tStart)/CLOCKS_PER_SEC<<endl;
		return;
	}
	tStart = clock();

	cv::Mat imgSlice, sliceDepth, sliceCoh;
	// ---------- FOR FIXED T ----------------//
	m_imgDepthT.create(H, W, CV_32F);
	m_imgCohT.create(H, W, CV_32F);
	for (int iterY = 0; iterY<H; iterY++){
		// first fix y and t, get x-s slice 
		int nt = T/2;
		int ny = iterY;
		getXSSlice(ny, nt, imgSlice);
		STwithVigra(imgSlice, sliceDepth, sliceCoh, dInner, dOuter);
		for (int iterX=0; iterX<W; iterX++){
			m_imgDepthT.at<float>(ny, iterX) = -1/tan(sliceDepth.at<float>(S/2, iterX));
			m_imgCohT.at<float>(ny, iterX) = sliceCoh.at<float>(S/2, iterX);
		}
	}

	// ------- FOR FIXED S----------------//
	m_imgDepthS.create(H, W, CV_32F);
	m_imgCohS.create(H, W, CV_32F);	
	for (int iterX = 0; iterX<W; iterX++){
		// first fix x and s, get y-t slice 
		int ns = S/2;
		int nx = iterX;
		getYTSlice(nx, ns, imgSlice);
		STwithVigra(imgSlice, sliceDepth, sliceCoh, dInner, dOuter);
		for (int iterY=0; iterY<H; iterY++){
			if(bInverseYT){
				sliceDepth.at<float>(T/2, iterY) = PIE - sliceDepth.at<float>(T/2, iterY);
			}
			m_imgDepthS.at<float>(iterY, nx) =  -1/tan(sliceDepth.at<float>(T/2, iterY));
			m_imgCohS.at<float>(iterY, nx) = sliceCoh.at<float>(T/2, iterY);
		}
	}

	cout<<"depth estimation done!   "<<(clock()-tStart)/CLOCKS_PER_SEC<<endl;
	if (bCohImgBlur){
		cv::Mat cohTBlur, cohSBlur;
		GaussianBlur( m_imgCohT, cohTBlur, cv::Size(5,5), 0, 0, cv::BORDER_DEFAULT );
		m_imgCohT = cohTBlur;
		GaussianBlur( m_imgCohS, cohSBlur, cv::Size(5,5), 0, 0, cv::BORDER_DEFAULT );
		m_imgCohS = cohSBlur;
		cout<<"blur coherence maps done!"<<endl;
	}

	// truncate results
	truncateResults();

	// return results
	depthS = m_imgDepthS;
	depthT = m_imgDepthT;
	cohS = m_imgCohS;
	cohT = m_imgCohT;

	return;
}

// apply on the whole image when depth and coh images are already available.
void lightfield::reviseCoh(cv::Mat & cohRS, cv::Mat & cohRT, cv::Mat & imgCost)
{
	assert((!m_imgDepthS.empty()) && (!m_imgDepthT.empty()) && (!m_imgCohS.empty()) && (!m_imgCohT.empty()));
	size_t H(yRes), W(xRes), T(tRes), S(sRes), C(nChannels);
	cv::Mat imgSlice;
	cv::Mat sliceDepth, sliceCoh, sliceCohRevised, sliceCost;
	imgCost = cv::Mat::zeros(H, W, CV_32F);
	m_imgCohTR.create(H, W, CV_32F);
	m_imgCohSR.create(H, W, CV_32F);
	
	clock_t tStart;
	tStart = clock();

	// ---------- FOR FIXED T ----------------//	
	//sliceCohRevised.create(1, W, CV_32F);
	//sliceCost.create(1, W, CV_32F);
	for (int iterY = 0; iterY<H; iterY++){
		// first fix y and t, get x-s slice 
		int nt = T/2;
		int ny = iterY;
		m_imgDepthT.row(ny).copyTo(sliceDepth);
		m_imgCohT.row(ny).copyTo(sliceCoh);
		getXSSlice(ny, nt, imgSlice);
		reviseSTNew(imgSlice, sliceDepth, sliceCoh, sliceCohRevised, sliceCost);
		for (int i=0; i<W; i++){
		//	imgCost.at<float>(ny, i) += sliceCost.at<float>(0, i);
			m_imgCohTR.at<float>(ny, i) = sliceCohRevised.at<float>(0, i);
		}
	}
	// ------- FOR FIXED S----------------//
	//sliceCohRevised.create(1, H, CV_32F);
	//sliceCost.create(1, H, CV_32F);
	for (int iterX = 0; iterX<W; iterX++){
		// first fix x and s, get y-t slice 
		int ns = S/2;
		int nx = iterX;
		m_imgDepthS.col(nx).copyTo(sliceDepth);
		m_imgCohS.col(nx).copyTo(sliceCoh);
		sliceDepth = sliceDepth.reshape(0,1);
		sliceCoh = sliceCoh.reshape(0,1);
		getYTSlice(nx, ns, imgSlice);
		reviseSTNew(imgSlice, sliceDepth, sliceCoh, sliceCohRevised, sliceCost);
		for (int i=0; i<H; i++){
			imgCost.at<float>(i, iterX) += sliceCost.at<float>(0, i);
			m_imgCohSR.at<float>(i, iterX) = sliceCohRevised.at<float>(0, i);
		}
	}
	cohRS = m_imgCohSR;
	cohRT = m_imgCohTR;
	cout<<"Refinement done!   "<<(clock()-tStart)/CLOCKS_PER_SEC<<endl;
	return;
}

// set the result from outside, for debugging purpose
void lightfield::cheatDepthEstimation(cv::Mat & depthS, cv::Mat & cohS, 
		cv::Mat & depthT, cv::Mat & cohT)
{
	if (depthS.depth()!=CV_32F)
		m_imgDepthS = getFloatImage(depthS);
	if (depthT.depth()!=CV_32F)
		m_imgDepthT = getFloatImage(depthT);
	if (cohS.depth()!=CV_32F)
		m_imgCohS = getFloatImage(cohS);
	if (cohT.depth()!=CV_32F)
		m_imgCohT = getFloatImage(cohT);
}

// with Revision
void lightfield::getDepthEstimation2(cv::Mat & depthS,  cv::Mat & cohS,
		cv::Mat & depthT, cv::Mat & cohT, bool bInverseYT, float fThreshold)
{
	//cv::Mat depthS, depthT;
	size_t H(yRes), W(xRes), T(tRes), S(sRes), C(nChannels);

	if (LFData == NULL && loadLFData()==false){
		cout<<"LF data loading failed!"<<endl;
		return;
	}

	cv::Mat imgTest(H, W, CV_32F);
	
	// ---------- FOR FIXED T ------------------//		
	depthT.create(H, W, CV_32F);
	cohT.create(H, W, CV_32F);

	for (int iterY = 0; iterY<H; iterY++){
		// first fix y and t, get x-s slice 
		int nt = T/2;
		int ny = iterY;
		cv::Mat imgSlice, sliceDepth, sliceCoh, sliceCohRevised, imgCost;
		getXSSlice(ny, nt, imgSlice);
		STwithVigra(imgSlice, sliceDepth, sliceCoh, dInner, dOuter);
		reviseST(imgSlice, sliceDepth, sliceCoh, sliceCohRevised, imgCost);
		for (int iterX=0; iterX<W; iterX++){
			depthT.at<float>(ny, iterX) = tan(sliceDepth.at<float>(S/2, iterX)-PIE/2);
			cohT.at<float>(ny, iterX) = sliceCohRevised.at<float>(0, iterX);
			imgTest.at<float>(ny, iterX) = imgCost.at<float>(0, iterX);
			//imgTest.at<cv::Vec3b>(ny, iterX) = imgCost.at<cv::Vec3b>(0, iterX);
		}
	}
	showHistogram(imgTest);
	saveImage("imgCost.jpg", imgTest, false);

	// ------- FOR FIXED S----------------//
	depthS.create(H, W, CV_32F);
	cohS.create(H, W, CV_32F);
	
	for (int iterX = 0; iterX<W; iterX++){
		// first fix x and s, get y-t slice 
		int ns = S/2;
		int nx = iterX;
		cv::Mat imgSlice, sliceDepth, sliceCoh, sliceCohRevised, imgCost;
		getYTSlice(nx, ns, imgSlice);
		STwithVigra(imgSlice, sliceDepth, sliceCoh, dInner, dOuter);
		reviseST(imgSlice, sliceDepth, sliceCoh, sliceCohRevised, imgCost);
		for (int iterY=0; iterY<H; iterY++){
			if(bInverseYT){
				sliceDepth.at<float>(T/2, iterY) = PIE - sliceDepth.at<float>(T/2, iterY);
			}
			depthS.at<float>(iterY, nx) = tan(sliceDepth.at<float>(T/2, iterY)-PIE/2);
			cohS.at<float>(iterY, nx) = sliceCohRevised.at<float>(0, iterY);
			imgTest.at<float>(iterY, nx) = imgCost.at<float>(0, iterY);
		}
	}
	showHistogram(imgTest);
	saveImage("imgCost2.jpg", imgTest);

	bool bCohImgBlur = false;
	if (bCohImgBlur){
		cv::Mat cohTBlur, cohSBlur;
		GaussianBlur( cohT, cohTBlur, cv::Size(5,5), 0, 0, cv::BORDER_DEFAULT );	cohT = cohTBlur;
		GaussianBlur( cohS, cohSBlur, cv::Size(5,5), 0, 0, cv::BORDER_DEFAULT );	cohS = cohSBlur;
	}

	cout<<"depth estimation with refinement done!"<<endl;
	return;
}


//--------------operation on H5 file---------------------//	
bool lightfield::setGTMinMax(){
	// Open the light field data sets
	hid_t dset = H5Dopen (fileID, GTDatasetName.c_str(), H5P_DEFAULT);
	if ( dset <= 0 ) {
		return false;
	}
	cout<< "  data set '" << GTDatasetName << "' opened." << endl;
	
	int nDims;
	hsize_t dims2D[2];
	// Retrieve dimension attributes
	H5LTget_dataset_ndims(fileID, GTDatasetName.c_str(), &nDims);
	if ( nDims != 2 ) {
		cout<< "light field data set '" << GTDatasetName.c_str() << "' is not two-dimensional." << endl;
		return -1;
	}

	H5LTget_dataset_info(fileID, GTDatasetName.c_str(), dims2D, NULL, NULL);
	size_t W = dims2D[1];
	size_t H = dims2D[0];
	cout<< "  data set '" << GTDatasetName.c_str() <<  " views, " << W << " x " << H << endl;

	// create buffer for light field data
	float *data2d = new float[ W*H];
	H5LTread_dataset_float( fileID, GTDatasetName.c_str(), data2d );

	float minVal(1000000), maxVal(-1000000);
	for (int iterPix=0; iterPix<H*W; iterPix++){
		if (data2d[iterPix]<minVal){
			minVal = data2d[iterPix];
		}else if(data2d[iterPix]>maxVal){
			maxVal = data2d[iterPix];
		}
	}
	cout<<minVal<<" "<<maxVal<<endl;
	herr_t bSuccess = H5LTset_attribute_float (fileID, "/", "minGT", &minVal, 1);
	bSuccess = H5LTset_attribute_float (fileID, "/", "maxGT", &maxVal, 1);

	fMinGT = minVal;
	fMaxGT = maxVal;

	H5Dclose( dset );
	return true;
}
	
bool lightfield::loadAttributes(){
	hsize_t dims;
	hsize_t dims2D[2];
	H5T_class_t typeClass;
	size_t typeSize;
	int nDims;

	herr_t bFound = H5LTfind_attribute (fileID, "focalLength");
	if (bFound<0)
		return false;

	herr_t bGotInfo = H5LTget_attribute_ndims(fileID, "/", "focalLength",  &nDims );
	bGotInfo = H5LTget_attribute_info(fileID, "/", "focalLength", &dims, &typeClass, &typeSize );
	bGotInfo = H5LTget_attribute_float(fileID, "/", "focalLength",  &fFocalLength );

	bGotInfo = H5LTget_attribute_ndims(fileID, "/", "camDistance",  &nDims );
	bGotInfo = H5LTget_attribute_info(fileID, "/", "camDistance", &dims, &typeClass, &typeSize );
	bGotInfo = H5LTget_attribute_float(fileID, "/", "camDistance",  &fCamDistance );

	//bGotInfo = H5LTget_attribute_ndims(fileID, "/", "xRes",  &nDims );
	//bGotInfo = H5LTget_attribute_info(fileID, "/", "xRes", &dims, &typeClass, &typeSize );
	//bGotInfo = H5LTget_attribute_int(fileID, "/", "xRes",  &xRes );

	bGotInfo = H5LTget_attribute_ndims(fileID, "/", "dH",  &nDims );
	bGotInfo = H5LTget_attribute_info(fileID, "/", "dH", &dims, &typeClass, &typeSize );
	bGotInfo = H5LTget_attribute_float(fileID, "/", "dH",  &dH );

	return true;
}

bool lightfield::setViewImg(){
	size_t S(sRes), T(tRes), H(yRes), W(xRes), C(nChannels);	

	if (LFData == NULL && loadLFData()==false){
		cout<<"LF data loading failed!"<<endl;
		return false;
	}

	// get center view
	uchar *data3d = new uchar[W*H*C];
	hsize_t nCenterS = S/2;
	hsize_t nCenterT = T/2;
	hsize_t nOffSet = nCenterS*T*W*H*C + nCenterT*W*H*C;
	memcpy(data3d, LFData + nOffSet, W*H*C);

	// write the info back
	hsize_t dim3d[3];
	dim3d[0] = W; dim3d[1] = H; dim3d[2] = C; 
	hid_t dataspace = H5Screate_simple(3, dim3d, NULL); 
	hid_t dataset = H5Dcreate(fileID, "CenterView", H5T_NATIVE_UCHAR, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	herr_t bErr = H5Dwrite(dataset, H5T_NATIVE_UCHAR, H5S_ALL, H5S_ALL, H5P_DEFAULT, data3d);
 
	H5Dclose(dataset);
	H5Sclose(dataspace);

	if (bErr<0)
		return false;
	else
		return true;
}

// old function
bool lightfield::getDepthEstimationAll(float* output, bool bInverseYT)
{
	size_t H(yRes), W(xRes), T(tRes), S(sRes), C(nChannels);

	if (LFData == NULL && loadLFData()==false){
		cout<<"LF data loading failed!"<<endl;
		return false;
	}

	cv::Mat depthT, cohT, depthS, cohS;	
	
	float maxVal(-1), minVal(1000000);

	// ---------- FOR FIXED T ------------------//		
	depthT.create(H, W, CV_32F);
	cohT.create(H, W, CV_32F);
	for (int iterT=0; iterT<T; iterT++){
		for (int iterY = 0; iterY<H; iterY++){
			// first fix y and t, get x-s slice 
			int nt = iterT;
			int ny = iterY;
			cv::Mat imgSlice, sliceDepth, sliceCoh;
			getXSSlice(ny, nt, imgSlice);
			StructureTensor(imgSlice, sliceDepth, sliceCoh);
			for (int iterX=0; iterX<W; iterX++){
				for (int iterS=0; iterS<S; iterS++){
					//imgTest.at<float>(ny, iterX) = (float)imgJxxBlur.at<float>(S/2, iterX);
					int nOffsetDepthT = iterT*S*H*W*4 + iterS*H*W*4 + iterY*W*4 + iterX*4 + 0;
					output[nOffsetDepthT] = sliceDepth.at<float>(iterS, iterX);
					int nOffsetCohT = iterT*S*H*W*4 + iterS*H*W*4 + iterY*W*4 + iterX*4 + 1; 
					output[nOffsetCohT] = sliceCoh.at<float>(iterS, iterX);
					if (output[nOffsetDepthT]<minVal)	
						minVal = output[nOffsetDepthT];
					if (output[nOffsetDepthT]>maxVal)	
						maxVal = output[nOffsetDepthT];
				}
			}
			//cout<<"T:"<<iterT<<"Y:"<<iterY<<" finished!"<<endl;
		}
	}

	// ------- FOR FIXED S----------------//
	depthS.create(H, W, CV_32F);
	cohS.create(H, W, CV_32F);
	for (int iterS = 0; iterS<S; iterS++){
		for (int iterX = 0; iterX<W; iterX++){
			// first fix x and s, get y-t slice 
			int ns = iterS;
			int nx = iterX;
			cv::Mat imgSlice, sliceDepth, sliceCoh;
			getYTSlice(nx, ns, imgSlice);
			StructureTensor(imgSlice, sliceDepth, sliceCoh);
			for (int iterY=0; iterY<H; iterY++){
				for (int iterT=0; iterT<T; iterT++){
					int nOffsetDepthS = iterT*S*H*W*4 + iterS*H*W*4 + iterY*W*4 + iterX*4 + 2;
					output[nOffsetDepthS] = sliceDepth.at<float>(iterT, iterY);
					int nOffsetCohS = iterT*S*H*W*4 + iterS*H*W*4 + iterY*W*4 + iterX*4 + 3; 
					output[nOffsetCohS] = sliceCoh.at<float>(iterT, iterY);
					if (output[nOffsetDepthS]<minVal)	
						minVal = output[nOffsetDepthS];
					if (output[nOffsetDepthS]>maxVal)	
						maxVal = output[nOffsetDepthS];
				}
			}
			//cout<<"S:"<<iterS<<"X:"<<iterX<<" finished!"<<endl;
		}
	}

	cout<<maxVal<<"  "<<minVal<<endl;
	for (int iterT=0; iterT<T; iterT++){
		for(int iterS=0; iterS<S; iterS++){
			for (int iterY=0; iterY<H; iterY++){
				for (int iterX=0; iterX<W; iterX++){
					int nOffset = iterT*S*H*W*4 + iterS*H*W*4 + iterY*W*4 + iterX*4;
					output[nOffset] = (output[nOffset]-minVal)/maxVal;
					nOffset += 2;
					output[nOffset] = (output[nOffset]-minVal)/maxVal;
					if (bInverseYT){
						output[nOffset] = 1-output[nOffset];
					}
				}
			}
		}
	}

	return true;
}

// old function
bool lightfield::cnvtShort(uchar * buffer)
{
	size_t H(yRes), W(xRes), T(tRes), S(sRes), C(nChannels);

	if (LFData == NULL && loadLFData()==false){
		cout<<"LF data loading failed!"<<endl;
		return false;
	}

	bool flag = true;
	for (int i=0; i<T*S*W*H*C; i++){
		if (LFData[i]<00 || LFData[i]>255){
			cout<<"Warning: overflow for uchar! "<<LFData[i]<<endl;
			flag = false;
		}
		buffer[i] = LFData[i];
	}

	return flag;
}


// ---------------private functions------------------------//
// open a h5 file
bool lightfield::openH5File(string sFilename)
{
	string sDir, sShortName;
	breakupFileName(sFilename, sDir, sShortName);
	cout<< "opening hdf5 container:" << sFilename << endl;
	
	// open the light field container file
	//fileID = H5Fopen(sFilename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
	fileID = H5Fopen(sFilename.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
	
	if (fileID <= 0){
		cout << "ERROR: could not open light field depth file '" << sFilename << "'" << endl;
		bOpen = false;
		return false;
	}else{
		bOpen = true;
		cout << "  container opened successfully." << endl;
	}

	// get resolution info.
	int ndims;
	H5LTget_dataset_ndims(fileID, LFDatasetName.c_str(), &ndims );
	if(ndims==5){
		hsize_t dims[5];
		H5LTget_dataset_info(fileID, LFDatasetName.c_str(), dims, NULL,NULL);
		sRes = dims[0];
		tRes = dims[1];
		xRes = dims[3];
		yRes = dims[2];
		nChannels = dims[4];
	}else if(ndims==4){
		hsize_t dims[4];
		H5LTget_dataset_info(fileID, LFDatasetName.c_str(), dims, NULL,NULL);
		sRes = dims[0];
		tRes = dims[1];
		xRes = dims[3];
		yRes = dims[2];
		nChannels = 1;
	}else {
		cout<<"failed to get resolution info."<<endl;
		//H5Dclose( dset );
		return false;
	}
	//H5Dclose( dset );
	cout<<sRes<<" "<<tRes<<" "<<xRes<<" "<<yRes<<" "<<nChannels<<endl;
	return true;
}

/* 
Load full light field data from h5 file.
Load 4/5 dimension data
	- 4 dimensions for grayscale images
	- 5 dimensions for color images
Did not verify if the specified resolution is correct or not
*/
bool lightfield::loadLFData()
{
	// open dataset
	string dset_name = LFDatasetName;
	hid_t dset = H5Dopen (fileID, dset_name.c_str(), H5P_DEFAULT);
	if ( dset == 0 ) {
		cout<< "could not open light field data set '" << dset_name << endl;
		return false;
	}
	cout<< "  data set '" << dset_name << "' opened." << endl;

	// Retrieve dimension attributes
	int ndims;
	H5LTget_dataset_ndims(fileID, dset_name.c_str(), &ndims );
	if ( ndims!=4 && ndims!=5 ) {
		cout<< "light field data set '" << dset_name << "' is not 4 or 5-dimensional." << endl;
		return false;
	}

	size_t S(sRes), T(tRes), H(yRes), W(xRes), C(nChannels);
	// create buffer for light field data
	LFData = new uchar[ S*T*W*H*C];
	cout<<"memory allocate success!"<<endl;
	if (H5Dget_type(dset)!=H5T_NATIVE_UCHAR
		&& H5Dget_type(dset)!=H5T_NATIVE_USHORT
		&& H5Dget_type(dset)!=H5T_STD_U8BE
		&& H5Dget_type(dset)!=H5T_STD_U8LE){
		cout<<"Warning: unsupported datatype:"<<H5Dget_type(dset)<<endl;
	}
	H5Dread(dset, H5T_NATIVE_UCHAR, H5S_ALL, H5S_ALL, H5P_DEFAULT, LFData);
	//H5LTread_dataset_float( fileID, LFDatasetName.c_str(), LFData );
	cout<<"finish loading data!"<<endl;
	H5Dclose( dset );

	if (bSwapChannel){
		swapChannel();
	}

	return true;
}

// did not test...
// maybe no need to use it
bool lightfield::swapChannel()
{
	if (nChannels != 3)
		return false;

	size_t S(sRes), T(tRes), H(yRes), W(xRes), C(nChannels);	
	for (int iterV=0; iterV<T*S; iterV++){
		int nOffset = iterV*H*W*C;
		for (int iterH=0; iterH<H; iterH++){
			for (int iterW=0; iterW<W; iterW++){
				uchar tmp = LFData[nOffset+2];
				LFData[nOffset+2] = LFData[nOffset+0];
				LFData[nOffset+0] = tmp;
				nOffset += 3;
			}
		}
	}
	return true;
}

/* 
if there is gt infomation, truncate to the range of ground truth
else truncate to 95% majority (to implement....)
*/
void lightfield::truncateResults()
{
	size_t H(yRes), W(xRes), T(tRes), S(sRes), C(nChannels);
	cv::Mat imgGT = getGroundTruth();
	float fMinGT(0), fMaxGT(0);
	if (!imgGT.empty()){
		fMinGT = getGTMin();
		fMaxGT = getGTMax();
	}else{
		fMinGT = -4;
		fMaxGT = 4;
	}
	// truncate local depth estimation
	for (int iterH=0; iterH<H; iterH++){
		for(int iterW=0; iterW<W; iterW++){
			if (m_imgDepthS.at<float>(iterH, iterW)<fMinGT){
				m_imgDepthS.at<float>(iterH, iterW) = fMinGT;
				m_imgCohS.at<float>(iterH, iterW) = 0;
			}else if(m_imgDepthS.at<float>(iterH, iterW)>fMaxGT){
				m_imgDepthS.at<float>(iterH, iterW) = fMaxGT;
				m_imgCohS.at<float>(iterH, iterW) = 0;
			}
			if (m_imgDepthT.at<float>(iterH, iterW)<fMinGT){
				m_imgDepthT.at<float>(iterH, iterW) = fMinGT;
				m_imgCohT.at<float>(iterH, iterW) = 0;
			}else if(m_imgDepthT.at<float>(iterH, iterW)>fMaxGT){
				m_imgDepthT.at<float>(iterH, iterW) = fMaxGT;
				m_imgCohT.at<float>(iterH, iterW) = 0;
			}
		}
	}
	return;
}