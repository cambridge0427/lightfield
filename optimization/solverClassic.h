#ifndef LINEAR_SOLVER_CLASSIC
#define LINEAR_SOLVER_CLASSIC

#include <Eigen/Sparse>
#include <cv.h>
#include <highgui.h>

class classicSolver{
public:
	classicSolver();
	~classicSolver();

private:
	int mnWidth; // full image
	int mnHeight;// full image
	int mnWidthStep;// full image
	//int mnSize;
	int mnChannels;
	int mnPatchRadius;

	// params about iteration
	double mdLambda;
	int mnMaxIters;
	double mdErrTolorant /*= 0.00001*/;

	// images 
	cv::Mat mImgResult, mImgView, mImgDepth, mImgCoh, mImgGuess;
	 //ROI 
	bool mbUsingMask;
	int mnLeftX, mnRightX, mnUpY, mnDownY;
	// arrays
	bool *mask;
	
public:
	void setParameters(const double dLambda, const int nMaxIters, const int nPatchRadius);
	void loadImage(const cv::Mat& imgI, const cv::Mat& imgd, const cv::Mat& imgC);
	void setMast( bool* pMask);
	void setROI(int nLeftX, int nRightX, int nUpY, int nDownY);
	void setMaxIters(const int nMaxIters);
	void solve(cv::Mat& imgResult, bool bUsingMask);
	//void solveWithMask(cv::Mat& imgResult);
	void getResult(cv::Mat& imgResult);

private:
	//void allocMem();
	//void clearMem();
	void setSparseMatL(Eigen::SparseMatrix<double>& output);

	// for debugging
	void print(const Eigen::SparseMatrix<float, Eigen::RowMajor>& mat);
};

//void eqnSolverWithEigen(double* b, double* D, unsigned char* imageData, double* output, 
//	double* firstGuess, int width, int height, int nChannels, int nMatIters);
//void eqnSolverCG(double* b, double* D, unsigned char* imageData, double* output,
//	double* firstGuess, int nWidth, int nHeight, int nChannels, int nMatIters);
//void setSparseMatL(unsigned char* imageData, Eigen::SparseMatrix<double, Eigen::RowMajor>& output,
//	int nPatchRadius, int nWidth, int nHeight, int nChannels);

#endif