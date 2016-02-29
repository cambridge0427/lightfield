#include <vector>
#include <iostream>
#include <math.h>
#include <time.h>

#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/SparseCholesky>

#include "../common/basic.h"
#include "solverClassic.h"

using namespace std;

classicSolver::classicSolver()
	:mnPatchRadius(1)
	,mdLambda(0.1)
	,mnMaxIters(-1)
	,mdErrTolorant(0.0001)
	,mbUsingMask(false)
	,mask(NULL)
{

}

classicSolver::~classicSolver()
{
}

void classicSolver::setParameters(const double dLambda, const int nMaxIters, const int nPatchRadius)
{
	mdLambda = dLambda;
	mnMaxIters = nMaxIters;
	mnPatchRadius = mnPatchRadius;
}

void classicSolver::loadImage(const cv::Mat& imgI, const cv::Mat& imgD, const cv::Mat& imgC)
{
	mnHeight = imgI.rows;
	mnWidth = imgI.cols;
	mnWidthStep = mnWidth * imgI.channels();
	mnChannels = imgI.channels();
	mnLeftX = 0; mnRightX = mnWidth;
	mnUpY = 0; mnDownY = mnHeight;

	mImgView = imgI.clone();
	mImgDepth = imgD.clone();
	mImgCoh = imgC.clone();

	mImgResult.create(mnHeight, mnWidth, CV_32F);
	mImgResult.setTo(0);
}

void classicSolver::setMast( bool* pMask)
{
	mask = pMask;
}

void classicSolver::setROI(int nLeftX, int nRightX, int nUpY, int nDownY)
{
	mnLeftX = nLeftX;
	mnRightX = nRightX;
	mnUpY = nUpY;
	mnDownY = nDownY;
}

void classicSolver::getResult(cv::Mat& imgResult)
{
	imgResult = mImgResult.clone();
}

void classicSolver::solve(cv::Mat& imgResult, bool bUsingMask)
{
	mbUsingMask = bUsingMask;
	if(mbUsingMask && mask==NULL)
		return;

	clock_t init, final;
	init=clock();

	int size = (mnDownY-mnUpY)*(mnRightX-mnLeftX);
	cout<<"size:"<<size<<endl;
	short nPatchRadius = 1;
	short nPatchSize = (2*nPatchRadius+1)*(2*nPatchRadius+1);
	//Eigen::SparseMatrix<double, Eigen::RowMajor> sMatLD(size, size); 
	Eigen::SparseMatrix<double> sMatLD;
	//set sparse_L from image Data
	setSparseMatL(sMatLD);
	// add D to sparse_l, get sparse_a
	for (int iterH=mnUpY; iterH<mnDownY; iterH++){
		for(int iterW=mnLeftX; iterW<mnRightX; iterW++){
			int iterPix = (iterH-mnUpY)*(mnRightX-mnLeftX)+(iterW-mnLeftX);
			sMatLD.coeffRef(iterPix, iterPix) += mdLambda*mImgCoh.at<float>(iterH, iterW);
		}
	}

	Eigen::VectorXd x(size);

if(1){
	// solve with Eigen Solver
	// solve Ax = b
	
	//Eigen::ConjugateGradient<Eigen::SparseMatrix<double, Eigen::RowMajor>> solver;
	//solver.setMaxIterations(mnMaxIters);
	Eigen::SimplicialLDLT<Eigen::SparseMatrix<double> > solver;
	//Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> solver;

	solver.compute(sMatLD);
	if(solver.info()!= Eigen::Success) {
		cout<<"decomposition failed!"<<endl;
		return;
	}
	cout<<"decomposition succeeded!"<<endl;

	Eigen::VectorXd vb(size);
	for (int iterH=mnUpY; iterH<mnDownY; iterH++){
		for(int iterW=mnLeftX; iterW<mnRightX; iterW++){
			int iterPix = (iterH-mnUpY)*(mnRightX-mnLeftX)+(iterW-mnLeftX);
			vb(iterPix) = mdLambda*mImgCoh.at<float>(iterH, iterW)
					*mImgDepth.at<float>(iterH, iterW);
		}
	}
	x = solver.solve(vb);
	if(solver.info()!= Eigen::Success) {
		// solving failed
		return;
	}

	// show error step by step
	//Eigen::VectorXd vTmp = Eigen::VectorXd::Random(size);
	//solver.setMaxIterations(1);
	//int nIter = 0;
	//do {
	//	vTmp = solver.solveWithGuess(vb, vTmp);
	//	std::cout << nIter << " : " << solver.error() << std::endl;
	//	++nIter;
	//} while (solver.info()!= Eigen::Success && nIter<100);
	//x = vTmp;

}else{
	if (mnMaxIters<0)
		mnMaxIters = size+1;	
	Eigen::VectorXd r(size);
	Eigen::VectorXd p(size);
	Eigen::VectorXd w(size);
	double rho =0;
	double rho_last = 0;
	double beta = 0;
	double alpha = 0;
	// set vectors
	//x=0, r=b-Ax=b;
	for (int iterH=mnUpY; iterH<mnDownY; iterH++){
		for(int iterW=mnLeftX; iterW<mnRightX; iterW++){
			int iterPix = (iterH-mnUpY)*(mnRightX-mnLeftX)+(iterW-mnLeftX);
			if (mImgGuess.empty()){
				x(iterPix) = mImgDepth.at<float>(iterH, iterW);
			}else{
				x(iterPix) = mImgGuess.at<float>(iterH, iterW);
			}
			int nOffset = (iterH-mnUpY)*(mnRightX-mnLeftX)+(iterW-mnLeftX);
			if(mbUsingMask && !mask[nOffset]){
				x(iterPix) = 0;
			}
		}
	}
	r = sMatLD*x;
	for (int iterH=mnUpY; iterH<mnDownY; iterH++){
		for(int iterW=mnLeftX; iterW<mnRightX; iterW++){
			int iterPix = (iterH-mnUpY)*(mnRightX-mnLeftX)+(iterW-mnLeftX);
			r(iterPix) = mdLambda*mImgCoh.at<float>(iterH, iterW)
				*mImgDepth.at<float>(iterH, iterW)- r(iterPix);
		}
	}
	rho = r.dot(r);
	// p =r;
	for (int i=0; i<size; i++)
	{
		p(i) = r(i);
	}

	double t=rho;		//error tolerance
	int iterCount = 0;		//Ñ­»·´ÎÊý
	//iterative to solve matting equation
	//while(sqrt(rho)>epsilon*sqrt(t) && iterCount<nMaxIters)
	while(sqrt(rho)>mdErrTolorant*sqrt(t) && iterCount<mnMaxIters){
		iterCount++;
		cout<<"iteration times:"<<iterCount<<" ";
		w = sMatLD*p;
		double temp = p.dot(w);
		alpha = rho/temp;
		x = x+alpha*p;
		r = r-alpha*w;
		rho_last = rho;
		rho = r.dot(r);
		cout<<rho<<endl;
		if(rho>rho_last){
			cout<<"CG does not converge!"<<endl;
			x = x-alpha*p;
			break;
		}
		beta = rho/rho_last;
		p = r + beta*p;
	}
}	
	for (int iterH=mnUpY; iterH<mnDownY; iterH++){
		for(int iterW=mnLeftX; iterW<mnRightX; iterW++){
			int iterPix = (iterH-mnUpY)*(mnRightX-mnLeftX)+(iterW-mnLeftX);
			int nOffset = (iterH-mnUpY)*(mnRightX-mnLeftX)+(iterW-mnLeftX);
			if(!mbUsingMask || mask[nOffset]){
				mImgResult.at<float>(iterH, iterW) =  x(iterPix);
				imgResult.at<float>(iterH, iterW) =  x(iterPix);
			}
		}
	}
	mbUsingMask = false;

	final=clock()-init;
	cout<<"time:"<<(double)final / ((double)CLOCKS_PER_SEC)<<endl;
	return;
}

// set sparse set L according to image data
void classicSolver::setSparseMatL(Eigen::SparseMatrix<double>& output)
{
	clock_t init, final;

	cout<<"start to set sparse mat..."<<endl;
	// use l-2 norm to calculate distance
	double dGamma = 125; 	// weight = max(exp(-dis/gamma), 10^-16) ;
	double dMinWeight = pow(10.0, -30);
	short nPatchSize = (2*mnPatchRadius+1)*(2*mnPatchRadius+1);
	int size = (mnRightX-mnLeftX)*(mnDownY-mnUpY);
	Eigen::SparseMatrix<float, Eigen::RowMajor> sMatWeight(size, size); 
	sMatWeight.reserve(Eigen::VectorXi::Constant(size, nPatchSize));

	init = clock();
	// set weight matrix
	bool bNormalize = true;
	double* pWeightPatch = new double[nPatchSize];	//vector<float> vctWeight;
	for (int iterH=mnUpY; iterH<mnDownY; iterH++){
		for (int iterW=mnLeftX; iterW<mnRightX; iterW++){
			unsigned char* centerPt = new unsigned char[mnChannels];
			unsigned char* nbrPt = new unsigned char[mnChannels];
			cv::Vec3b v = mImgView.at<cv::Vec3b>(iterH, iterW);
			centerPt[0] = v[0];
			centerPt[1] = v[1];
			centerPt[2] = v[2];

			//vctWeight.clear();
			memset(pWeightPatch, 0, sizeof(double)*nPatchSize);
			double dTotalWeight = 0;
			int nCntNbrWeight = 0;
			int nLowH = iterH-mnPatchRadius;  //nLowH = nLowH<0 ? 0 : nLowH;
			int nUpH = iterH+mnPatchRadius; //nUpH = nUpH>=nHeight ? nHeight-1 : nUpH;
			int nLowW = iterW-mnPatchRadius;  //nLowW = nLowW<0 ? 0 : nLowW;
			int nUpW = iterW+mnPatchRadius; //nUpW = nUpW>=nWidth ? nWidth-1 : nUpW;
			for (int iterHInner=nLowH; iterHInner<=nUpH; iterHInner++ ){
				for (int iterWInner=nLowW; iterWInner<=nUpW; iterWInner++){
					if (iterHInner<mnUpY || iterHInner>=mnDownY
						|| iterWInner<mnLeftX || iterWInner>=mnRightX){
							continue;
					}
					if(iterHInner==iterH && iterWInner==iterW){
						continue;
					}
					int nOffset = (iterHInner-mnUpY)*(mnRightX-mnLeftX)+(iterWInner-mnLeftX);
					if(!mbUsingMask || mask[nOffset]){
						v = mImgView.at<cv::Vec3b>(iterHInner, iterWInner);
						nbrPt[0] = v[0];
						nbrPt[1] = v[1];
						nbrPt[2] = v[2];

						float dWeight = 0;
						float dDistance = 0;
						for(int iterChan=0; iterChan<mnChannels; iterChan++){
							dDistance+=pow((float)(centerPt[iterChan]-nbrPt[iterChan]),2);
						}
						dDistance = sqrt(dDistance);
						dWeight = exp(-1/dGamma * dDistance);
						if (dWeight<dMinWeight){
							dWeight = dMinWeight;
						}
						pWeightPatch[nCntNbrWeight++] = dWeight; //vctWeight.push_back(dWeight);
						dTotalWeight += dWeight;
					}
				}
			}// end of local window

			if (bNormalize){
				// normalize weight
				for (int iterWeight=0; iterWeight<nPatchSize; iterWeight++){
					if (dTotalWeight!=0)
						pWeightPatch[iterWeight] /= dTotalWeight;
				}
			}

			// put weight into sparse mat
			int iterWeight=0;
			for (int iterHInner=nLowH; iterHInner<=nUpH; iterHInner++ ){
				for (int iterWInner=nLowW; iterWInner<=nUpW; iterWInner++){
					int n1 = (iterH-mnUpY)*(mnRightX-mnLeftX)+(iterW-mnLeftX);
					int n2 = (iterHInner-mnUpY)*(mnRightX-mnLeftX)+(iterWInner-mnLeftX);
					int nOffset = (iterHInner-mnUpY)*(mnRightX-mnLeftX)+(iterWInner-mnLeftX);
					if (iterHInner<mnUpY || iterHInner>=mnDownY
						|| iterWInner<mnLeftX || iterWInner>=mnRightX){
							;
					}else if(n1==n2){
						if (bNormalize){
							sMatWeight.insert(n1, n1) = 1;
						}else{
							sMatWeight.insert(n1, n1) = dTotalWeight;
						}
					}else if(!mbUsingMask || mask[nOffset]){
						sMatWeight.insert(n1, n2) = -1*pWeightPatch[iterWeight++];
						//double dForDebug = dWeight[(iterH*nWidth+iterW)*nPatchSize+nCntNbr];
					}
				}
			}// end of local window
		}
	}//loop all pixels

	sMatWeight.makeCompressed();
	cout<<"finish setting weight matrix!  time:"<<(clock()-init)/ ((double)CLOCKS_PER_SEC)<<endl;

	init = clock();
	output = (sMatWeight.transpose()*sMatWeight).pruned();
	cout<<"finish multiplying weight matrix!  time:"<<(clock()-init)/ ((double)CLOCKS_PER_SEC)<<endl;
}

void classicSolver::print(const Eigen::SparseMatrix<float, Eigen::RowMajor>& mat)
{
	for (int k=0; k<mat.outerSize(); ++k){
		for (Eigen::SparseMatrix<float, Eigen::RowMajor>::InnerIterator it(mat,k); it; ++it)
		{
			cout<<it.value()<<" ";
			cout<<'('<<it.row()<<","; // row index
			cout<<it.col()<<")"; // col index (here it is equal to k)
			//it.index(); // inner index, here it is equal to it.row()
		}
		cout<<endl;
	}
}