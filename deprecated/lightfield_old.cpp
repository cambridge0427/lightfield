
#include <iostream>

#include "lightfield_old.h"
#include "../common/fileOp.h"
#include "../common/imageOp.h"

using namespace std;

const string lightfield_float::LFDatasetName = "LF";
const string lightfield_float::GTDatasetName = "GT";
const string lightfield_float::DepthDatasetName = "Depth";

// constructor
lightfield_float::lightfield_float(string H5Filename):
	bEmpty(true),
	LFData(NULL)
{
	if (openH5File(H5Filename)){
		bEmpty = false;
	}else {
		bEmpty = true;
	}
	return;
}

lightfield_float::lightfield_float():
	bEmpty(true),
	LFData(NULL)
{
}

//destructor
lightfield_float::~lightfield_float()
{
	if(bOpen){
		H5Fclose(fileID);	
	}
	if(LFData!=NULL){
		delete[] LFData;
	}
}

//-----------------simple operation------------------------//
bool lightfield_float::empty(){
	return bEmpty;
}

int lightfield_float::width(){
	return xRes;
}
	
int lightfield_float::height(){
	return yRes;
}

void lightfield_float::releaseLFData()
{
	if(LFData!=NULL){
		delete[] LFData;
		LFData = NULL;
	}
}


//--------------- fetch data------------------------------//
cv::Mat lightfield_float::getViewImage()
{
	size_t S(sRes), T(tRes), H(yRes), W(xRes), C(nChannels);	

	if (LFData == NULL && loadLFData()==false){
		cout<<"LF data loading failed!"<<endl;
		return cv::Mat();
	}

	// get center view
	float *data3d = new float[W*H*C];
	hsize_t nCenterS = S/2;
	hsize_t nCenterT = T/2;
	hsize_t nOffSet = nCenterS*T*W*H*C + nCenterT*W*H*C;
	memcpy(data3d, LFData + nOffSet, sizeof(float)*W*H*C);

	// generate cv image
	if (C == 3){
		cv::Mat viewImg = cv::Mat(H, W, CV_32FC3, data3d);
		//viewImg = viewImg/255;
		return viewImg;
	}else if (C==1){
		cv::Mat viewImg = cv::Mat(H, W, CV_32F, data3d);
		//viewImg = viewImg/255;
		return viewImg;
	}else{
		return cv::Mat();
	}
}

cv::Mat lightfield_float::getGroundTruth(bool bReverse)
{
	string dset_name = GTDatasetName;
	// Open the light field data sets
	hid_t dset = H5Dopen (fileID, dset_name.c_str(), H5P_DEFAULT);
	if ( dset <= 0 ) {
		cout<< "could not open light field data set '" << dset_name << "'" << endl;
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
	H5LTget_dataset_info(fileID, dset_name.c_str(), dims, NULL, NULL);
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

	if (data2d!=NULL){
		//generate cv image
		cv::Mat groundTruthImg = cv::Mat(H, W, CV_32F, data2d);
		// see range
		double minVal, maxVal;
		cv::minMaxLoc(groundTruthImg, &minVal, &maxVal);
		// rescale to 0-255
		//groundTruthImg = (groundTruthImg-minVal)/(maxVal-minVal);
		for (int iterC=0; iterC<groundTruthImg.cols; iterC++){
			for (int iterR=0; iterR<groundTruthImg.rows; iterR++){
				groundTruthImg.at<float>(iterR, iterC) = (groundTruthImg.at<float>(iterR, iterC) - minVal)/(maxVal-minVal);
			}
		}
		// reverse
		if (bReverse){
			//groundTruthImg = 1-groundTruthImg;
			for (int iterC=0; iterC<groundTruthImg.cols; iterC++){
				for (int iterR=0; iterR<groundTruthImg.rows; iterR++){
					groundTruthImg.at<float>(iterR, iterC) = 1-groundTruthImg.at<float>(iterR, iterC);
				}
			}
		}
			
		// groundTruthImg *=255;
		for (int iterC=0; iterC<groundTruthImg.cols; iterC++){
			for (int iterR=0; iterR<groundTruthImg.rows; iterR++){
				groundTruthImg.at<float>(iterR, iterC) = groundTruthImg.at<float>(iterR, iterC)*255;
			}
		}
		cv::Mat groundTruthImg2;
		groundTruthImg.convertTo(groundTruthImg2, CV_8U);

		delete[] data2d;
		return groundTruthImg2;
	}else
		return cv::Mat();
}

bool lightfield_float::getGroundTruth(float * buffer)
{
	if (buffer == NULL){
		return false;
	}	

	string dset_name = GTDatasetName;
	// Open the light field data sets
	hid_t dset = H5Dopen (fileID, dset_name.c_str(), H5P_DEFAULT);
	if ( dset <= 0 ) {
		cout<< "could not open light field data set '" << dset_name << "'" << endl;
		return false;
	}
	cout<< "  data set '" << dset_name << "' opened." << endl;

	// Retrieve dimension attributes
	int ndims;
	H5LTget_dataset_ndims(fileID, dset_name.c_str(), &ndims );
	if ( ndims != 2 ) {
		cout<< "light field data set '" << dset_name << "' is not two-dimensional." << endl;
		return false;
	}

	hsize_t dims[2];
	H5LTget_dataset_info(fileID, dset_name.c_str(), dims, NULL, NULL);
	size_t W = dims[1];
	size_t H = dims[0];
	cout<< "  data set '" << dset_name <<  " views, " << W << " x " << H << endl;
	if (W!=xRes || H!=yRes){
		cout<<"resolution does not match!"<<endl;
		return false;
	}

	H5LTread_dataset_float( fileID, dset_name.c_str(), buffer );
	return true;
}

cv::Mat lightfield_float::getDepthMap()
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

//---------------depth estimation-------------------------//
void lightfield_float::getDepthEstimation(cv::Mat & depthS, cv::Mat & cohS, cv::Mat & depthT, cv::Mat & cohT, cv::Mat & mergedDepth, bool bInverseYT)
{
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
		cv::Mat imgSlice, sliceDepth, sliceCoh;
		getXSSlice(ny, nt, imgSlice);
		StructureTensor(imgSlice, sliceDepth, sliceCoh);
		for (int iterX=0; iterX<W; iterX++){
			//imgTest.at<float>(ny, iterX) = (float)imgJxxBlur.at<float>(S/2, iterX);
			depthT.at<float>(ny, iterX) = sliceDepth.at<float>(S/2, iterX);
			cohT.at<float>(ny, iterX) = sliceCoh.at<float>(S/2, iterX);
		}
		cout<<iterY<<" finished!"<<endl;
	}

	// ------- FOR FIXED S----------------//
	depthS.create(H, W, CV_32F);
	cohS.create(H, W, CV_32F);
	
	for (int iterX = 0; iterX<W; iterX++){
		// first fix x and s, get y-t slice 
		int ns = S/2;
		int nx = iterX;
		cv::Mat imgSlice, sliceDepth, sliceCoh;
		getYTSlice(nx, ns, imgSlice);
		StructureTensor(imgSlice, sliceDepth, sliceCoh);
		for (int iterY=0; iterY<H; iterY++){
			//imgTest.at<float>(iterY, nx) = (float)imgJxxBlur.at<float>(T/2, iterY);
			depthS.at<float>(iterY, nx) = sliceDepth.at<float>(T/2, iterY);
			cohS.at<float>(iterY, nx) = sliceCoh.at<float>(T/2, iterY);
			//imgDet.at<float>(iterY, nx) =  imgJxxBlur.at<float>(T/2, iterY)*imgJyyBlur.at<float>(T/2, iterY) - pow(imgJxyBlur.at<float>(T/2, iterY),2);
			//imgTrace.at<float>(iterY, nx) = imgJxxBlur.at<float>(T/2, iterY) + imgJyyBlur.at<float>(T/2, iterY);
		}
		cout<<iterX<<" finished!"<<endl;
	}

	cv::normalize(depthT, depthT, 0, 1, cv::NORM_MINMAX);
	cv::normalize(depthS, depthS, 0, 1, cv::NORM_MINMAX);

	//bool bInverseYT = true;
	// inverse y-t slice
	if(bInverseYT){
		for (int iterX = 0; iterX<W; iterX++){
			for (int iterY=0; iterY<H; iterY++){
				depthS.at<float>(iterY, iterX) = 1 - depthS.at<float>(iterY, iterX);
			}
		}
	}

	bool bCohImgBlur = false;
	if (bCohImgBlur){
		cv::Mat cohTBlur, cohSBlur;
		GaussianBlur( cohT, cohTBlur, cv::Size(5,5), 0, 0, cv::BORDER_DEFAULT );	cohT = cohTBlur;
		GaussianBlur( cohS, cohSBlur, cv::Size(5,5), 0, 0, cv::BORDER_DEFAULT );	cohS = cohSBlur;
		
	}

	cout<<"depth estimation done!"<<endl;
	// merge depth estimation
	mergedDepth.create(depthS.size(), CV_32F);

	for (int iterR = 0; iterR<H; iterR++){
		for (int iterC = 0; iterC<W; iterC++){
			if (cohS.at<float>(iterR, iterC)<cohT.at<float>(iterR, iterC)){
				mergedDepth.at<float>(iterR, iterC) = depthT.at<float>(iterR, iterC);
			}else{
				mergedDepth.at<float>(iterR, iterC) = depthS.at<float>(iterR, iterC);
			}
		}
	}

	return;
}

bool lightfield_float::getDepthEstimationAll(float* output, bool bInverseYT)
{
	size_t H(yRes), W(xRes), T(tRes), S(sRes), C(nChannels);

	if (LFData == NULL && loadLFData()==false){
		cout<<"LF data loading failed!"<<endl;
		return false;
	}

	cv::Mat depthT, cohT, depthS, cohS;	

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
		if (iterY == 100){
			saveImage("test.jpg",imgSlice);
		}
		StructureTensor(imgSlice, sliceDepth, sliceCoh);
		for (int iterX=0; iterX<W; iterX++){
			for (int iterS=0; iterS<S; iterS++){
				//imgTest.at<float>(ny, iterX) = (float)imgJxxBlur.at<float>(S/2, iterX);
				int nOffsetDepthT = iterT*S*H*W*4 + iterS*H*W*4 + iterY*W*4 + iterX*4 + 0;
				output[nOffsetDepthT] = sliceDepth.at<float>(iterS, iterX);
				int nOffsetCohT = iterT*S*H*W*4 + iterS*H*W*4 + iterY*W*4 + iterX*4 + 1; 
				output[nOffsetCohT] = sliceCoh.at<float>(iterS, iterX);
			}
		}
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
			}
		}
	}
}

	return true;
}

bool lightfield_float::cnvtShort(uchar * buffer, bool bReorder, bool bFlip)
{
	size_t H(yRes), W(xRes), T(tRes), S(sRes), C(nChannels);

	if (LFData == NULL && loadLFData()==false){
		cout<<"LF data loading failed!"<<endl;
		return false;
	}

	bool bFlag = true;
	for (int iterV=0; iterV<T*S; iterV++){
		if (bFlip){
			int nBufferOffset = iterV*H*W*C;
			int nInputOffset = (iterV+1)*H*W*C - C;
			for (int iterH=0; iterH<H; iterH++){
				for (int iterW=0; iterW<W; iterW++){
					if (bReorder && C==3){
						buffer[nBufferOffset+0] = LFData[nInputOffset+2];
						buffer[nBufferOffset+1] = LFData[nInputOffset+1];
						buffer[nBufferOffset+2] = LFData[nInputOffset+0];
						nBufferOffset += 3;
						nInputOffset -= 3;
					}else if (!bReorder && C==3){
						buffer[nBufferOffset+0] = LFData[nInputOffset+0];
						buffer[nBufferOffset+1] = LFData[nInputOffset+1];
						buffer[nBufferOffset+2] = LFData[nInputOffset+2];
						nBufferOffset += 3;
						nInputOffset -= 3;
					}else if (C==1){
						buffer[nBufferOffset] = LFData[nInputOffset];
						nBufferOffset += 1;
						nInputOffset -= 1;
					}else{
						bFlag = false;
					}			
				}
			}
		}else{
			int nBufferOffset = iterV*H*W*C;
			int nInputOffset = iterV*H*W*C;
			for (int iterH=0; iterH<H; iterH++){
				for (int iterW=0; iterW<W; iterW++){
					if (bReorder && C==3){
						buffer[nBufferOffset+0] = LFData[nInputOffset+2];
						buffer[nBufferOffset+1] = LFData[nInputOffset+1];
						buffer[nBufferOffset+2] = LFData[nInputOffset+0];
						nBufferOffset += 3;
						nInputOffset += 3;
					}else if (!bReorder && C==3){
						buffer[nBufferOffset+0] = LFData[nInputOffset+0];
						buffer[nBufferOffset+1] = LFData[nInputOffset+1];
						buffer[nBufferOffset+2] = LFData[nInputOffset+2];
						nBufferOffset += 3;
						nInputOffset += 3;
					}else if (C==1){
						buffer[nBufferOffset] = LFData[nInputOffset];
						nBufferOffset += 1;
						nInputOffset += 1;
					}else{
						bFlag = false;
					}					
				}
			}
		}
	}

	return bFlag;
}


// ---------------private functions------------------------//
// open a h5 file
// did not open dataset, is it ok? Yes
bool lightfield_float::openH5File(string sFilename)
{
	string sDir, sShortName;
	breakupFileName(sFilename, sDir, sShortName);
	cout<< "opening hdf5 container:" << sFilename << endl;
		// open the light field container file
	fileID = H5Fopen(sFilename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
	
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
		sRes = dims[0];  // should dont be TSHWC??
		tRes = dims[1];
		xRes = dims[3];
		yRes = dims[2];
		nChannels = dims[4];
	}else if(ndims==4){
		hsize_t dims[4];
		H5LTget_dataset_info(fileID, LFDatasetName.c_str(), dims, NULL,NULL);
		sRes = dims[0];  // should dont be TSHWC??
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

// load 4/5 dimension data
// did not check resolution, is it ok?
bool lightfield_float::loadLFData()
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
	LFData = new float[ S*T*W*H*C];
	cout<<"memory allocate success!"<<endl;
	H5LTread_dataset_float( fileID, LFDatasetName.c_str(), LFData );
	cout<<"finish loading data!"<<endl;
	H5Dclose( dset );

	return true;
}

void lightfield_float::getYTSlice(const int nx, const int ns, cv::Mat & imgSlice)
{
	size_t S(sRes), T(tRes), H(yRes), W(xRes), C(nChannels);	
	if (LFData == NULL && loadLFData()==false){
		cout<<"LF data loading failed!"<<endl;
		return;
	}
	float* FiveDData = LFData;
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
	
void lightfield_float::getXSSlice(const int ny, const int nt, cv::Mat & imgSlice)
{
	size_t S(sRes), T(tRes), H(yRes), W(xRes), C(nChannels);	
	if (LFData == NULL && loadLFData()==false){
		cout<<"LF data loading failed!"<<endl;
		return;
	}
	float* FiveDData = LFData;
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
