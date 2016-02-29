#ifndef LINEAR_SOLVER_FASTMATTING
#define LINEAR_SOLVER_FASTMATTING

//void fastMatting(double* b, double* D, unsigned char* imageData, double* outputDepth, double* firstGuess,
//	int nPatchSize, int width, int height, int nChannels, int iterMax);

#include <cv.h>

// (L + lambda*C) d = lambda*C*d_tilta
// L <--- I
class fastMattingSolver{

public:
	fastMattingSolver();
	~fastMattingSolver();

private:
	int mnWidth; // full image
	int mnHeight;// full image
	int mnWidthStep;// full image
	//int mnSize;
	int mnChannels;
	int mnPatchSize;
	
	// images 
	cv::Mat mImgResult, mImgView, mImgDepth, mImgCoh, mImgGuess;

	// params about iteration
	double mdLambda;
	int mnMaxIters;
	double mdEpsilon/* = 1*/;
	double mdErrTolorant /*= 0.00001*/;

	// ROI 
	bool mbUsingMask;
	int mnLeftX, mnRightX, mnUpY, mnDownY;

	// arrays
	double *x, *r, *p, *delta_In, *Uk, *pIntegral, *IP, *LP, *Ak, *Bk, *w;
	uchar *imageData;
	double *D, *firstGuess, *outputDepth, *b;
	bool *mask;
	int *mpCntNbrs;

public:
	void setParameters(const double dLambda, const int nMaxIters, const int nPatchSize);
	void loadImage(const cv::Mat& imgI, const cv::Mat& imgd, const cv::Mat& imgC);
	void setMast( bool* pMask);
	void setROI(int nLeftX, int nRightX, int nUpY, int nDownY);
	void setMaxIters(const int nMaxIters);
	void solve(cv::Mat& imgResult);
	void solveWithMask(cv::Mat& imgResult);
	void getResult(cv::Mat& imgResult);
	void setGuess(cv::Mat& imgGuess);

private:
	void setMatrix();
	void multiLp(double *a);
	void clearMem();
	void allocMem();
	void setResult();
};

#endif