#ifndef SOLVER_MRF
#define SOLVER_MRF

#include "solver.h"

#include "../MRF/mrf.h"

enum MRFOptMethod {
	ICM_OPT, 
	EXPANSION,
	SWAP_OPT
};

class solverMRF : public solver{

public:
	// constructor
	solverMRF();
	solverMRF(int numOfLable, int nIterTimes, float fLambda, MRFOptMethod method);
	// destructor
	~solverMRF(){
		delete[] mpDataArray;
		delete[] mpHCue;
		delete[] mpVCue;
	}

private:
	static const int mnNumOfLabels;
	//int mnIterTimes;
	static const float mfLambda4SmoothTerm;
	MRFOptMethod meOptMethod;

	// paramters for smooth term
	static const int k;
	static const MRF::CostVal maxVal;

	cv::Mat mImgDepthRescale;

	MRF::CostVal* mpDataArray;
	MRF::CostVal* mpVCue;
	MRF::CostVal* mpHCue;

	// parameters for weight 
	static const float epsilon;
	static const float gamma;

	// parameters for iteration
	static const int nIterMax;
	static const float fTerminate;

public:
	virtual void solve(cv::Mat& imgResult);
	MRF::CostVal dCost(int pix, int i);
	MRF::CostVal fnCost(int pix1, int pix2, int i, int j);

private:
	void setDataArray();
	void setCueArray();
};

#endif