#ifndef __LIGHTFIELD_FLOAT_CLASS
#define __LIGHTFIELD_FLOAT_CLASS

#include <string>

#include <cv.h>

#include <hdf5.h>
#include <hdf5_hl.h>

class lightfield_float{

public:
	//constructor
	lightfield_float(const std::string s);
	lightfield_float();
	//destructor
	~lightfield_float();

private:
	hid_t fileID;
	bool bOpen;
	bool bEmpty;
	
	size_t xRes; //: spatial x resolution of a single image
	size_t yRes; //: spatial y resolution of a single image
	size_t sRes; //: angular vertical resolution of the light field
	size_t tRes; //: angular horizontal resolution of the light field
	size_t nChannels;
	//int dV;	  //: vertical base line
	//int dH;	  //: horizontal base line
	//int focalLength; //: for blender generated light fields the camera angle
	//int camDistance; //: camera distance in blender units
	//float inner; //: inner scale of the structure tensor used for the depth estimation
	//float outer;	//: outer scale of the structure tensor used for the depth estimation
	//int vSampling;	//: vertical sampling matrix
	//int hSampling;	//: horizontal sampling matrix
	//int channels;	//: number of color channels
	
	float *LFData; // view infomation
	//float *DepthData;
	//float *GroundTruthData;

public:

	static const std::string LFDatasetName;
	static const std::string GTDatasetName;
	static const std::string DepthDatasetName;

public:
	cv::Mat getGroundTruth(bool bReverse);
	bool getGroundTruth(float * buffer);
	cv::Mat getViewImage();
	cv::Mat getDepthMap();
	void getDepthEstimation(cv::Mat & depthS, cv::Mat & cohS, cv::Mat & depthT, cv::Mat & cohT, cv::Mat & mergedDepth, bool bInverseYT);
	bool getDepthEstimationAll(float* output, bool bInverseYT);
	bool cnvtShort(uchar* buffer, bool bReorder, bool bFlip);

	bool empty();
	int width();
	int height();
	void releaseLFData();
	int getsRes(){return sRes;}
	int gettRes(){return tRes;}
	int getnChannels(){return nChannels;}

	void getYTSlice(const int nx, const int ns, cv::Mat & imgSlice);
	void getXSSlice(const int ny, const int nt, cv::Mat & imgSlice);

private:
	bool openH5File(std::string sFilename);
	bool loadLFData();
};

#endif
