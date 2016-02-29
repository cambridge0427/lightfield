#ifndef __LIGHTFIELD_CLASS
#define __LIGHTFIELD_CLASS

#include <string>

#include <cv.h>

#include <hdf5.h>
#include <hdf5_hl.h>

class lightfield{

public:
	//constructor
	lightfield(const std::string & s, bool bSwapChannel);
	//destructor
	~lightfield();
	void releaseLFData();

private:
	lightfield();

private:
	hid_t fileID;
	bool bOpen;
	bool bEmpty;
	bool bSwapChannel;
	bool bFlip;
	
	size_t xRes; //: spatial x resolution of a single image
	size_t yRes; //: spatial y resolution of a single image
	size_t sRes; //: angular vertical resolution of the light field
	size_t tRes; //: angular horizontal resolution of the light field
	size_t nChannels;
	//int dV;	  //: vertical base line
	float dH;	  //: horizontal base line
	float fFocalLength; //: for blender generated light fields the camera angle
	float fCamDistance; //: camera distance in blender units
	//float inner; //: inner scale of the structure tensor used for the depth estimation
	//float outer;	//: outer scale of the structure tensor used for the depth estimation
	//int vSampling;	//: vertical sampling matrix
	//int hSampling;	//: horizontal sampling matrix
	//int channels;	//: number of color channels
	float fMinGT;
	float fMaxGT;
	
	uchar *LFData; // view information

	cv::Mat m_imgDepthT, m_imgDepthS;
	cv::Mat m_imgCohT, m_imgCohS;	// original coherence estimation
	cv::Mat m_imgCohTR, m_imgCohSR;	// revised
	cv::Mat m_imgGT;

	//parameters
	static const double dInner;
	static const double dOuter;
	static const double dLowCohThresh;
	static const bool bCohImgBlur;

public:
	static const std::string LFDatasetName;
	static const std::string GTDatasetName;
	static const std::string DepthDatasetName;
	static const std::string CenterViewDatasetName;

public:
	// fetch data
	cv::Mat getGroundTruth();
	cv::Mat getViewImage();
	cv::Mat getDepthMap();
	void getViewImage(int T, int S, cv::Mat & img);
	void getYTSlice(const int nx, const int ns, cv::Mat & imgSlice);
	void getXSSlice(const int ny, const int nt, cv::Mat & imgSlice);
	bool getGTMinFromH5(float& min);
	bool getGTMaxFromH5(float& max);
	
	// depth estimation
	void cheatDepthEstimation(cv::Mat & depthS, cv::Mat & cohS, 
		cv::Mat & depthT, cv::Mat & cohT);
	void getDepthEstimation(cv::Mat & depthS, cv::Mat & cohS, 
		cv::Mat & depthT, cv::Mat & cohT, bool bInverseYT);
	void getDepthEstimation2(cv::Mat & depthS,  cv::Mat & cohS,
		cv::Mat & depthT, cv::Mat & cohT, bool bInverseYT, float fThreshold);
	void reviseCoh(cv::Mat & cohRS, cv::Mat & cohRT, cv::Mat & imgCost);

	// inline functions
	bool empty(){return bEmpty;}
	int width(){return xRes;}
	int height(){return yRes;}
	int getsRes(){return sRes;}
	int gettRes(){return tRes;}
	int getnChannels(){return nChannels;}
	hid_t getFileID(){return fileID;}
	float getGTMin(){return fMinGT;}
	float getGTMax(){return fMaxGT;}

	//operation on h5 file
	bool setGTMinMax();
	bool loadAttributes();
	bool setViewImg();

	//old rarely-used functions
	bool getDepthEstimationAll(float *output, bool bInverseYT = true);
	bool cnvtShort(uchar* buffer);

private:
	bool openH5File(std::string sFilename);
	bool loadLFData();
	bool swapChannel();
	void truncateResults();
};

#endif
