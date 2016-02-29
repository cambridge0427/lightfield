#include <vector>
#include <stdio.h>
#include <iostream>
#include <math.h>

#include <highgui.h>

#include "../common/matrixOp.h"
#include "../common/imageOp.h"
#include "solverFastMatting.h"

using namespace std;

fastMattingSolver::fastMattingSolver()
	: mnWidth(0)
	,mnHeight(0)
	,mnWidthStep(0)
	,mdLambda(0.1)
	,mnMaxIters(-1)
	,mdEpsilon(0.001)
	,mdErrTolorant(0.0001)
	,x(NULL)
	,r(NULL)
	,p(NULL)
	,delta_In(NULL)
	,Uk(NULL)
	,pIntegral(NULL)
	,IP(NULL)
	,LP(NULL)
	,Ak(NULL)
	,Bk(NULL)
	,w(NULL)
	,imageData(NULL)
	,D(NULL)
	,firstGuess(NULL)
	,outputDepth(NULL)
	,b(NULL)
	,mask(NULL)
	,mbUsingMask(false)
	,mpCntNbrs(NULL)
{
}

fastMattingSolver::~fastMattingSolver()
{
	clearMem();
}

void fastMattingSolver::clearMem()
{
	delete[] x;		x = NULL;//vector x, size N
	delete[] r;		r = NULL;//vector r
	delete[] p;		p = NULL;
	delete[] pIntegral;	pIntegral = NULL;
	delete[] w;		w = NULL;
	delete[] LP;	LP = NULL;
	delete[] Ak;	Ak = NULL;
	delete[] Bk;	Bk = NULL;
	delete[] delta_In;	delta_In = NULL;
	delete[] Uk;	Uk = NULL;
	delete[] IP;	IP = NULL;
	delete[] imageData;	imageData = NULL;
	delete[] D;		D = NULL;
	delete[] firstGuess;	firstGuess =NULL;
	delete[] outputDepth;	outputDepth = NULL;
	delete[] b;		b = NULL;
	delete[] mpCntNbrs;		mpCntNbrs = NULL;
}

void fastMattingSolver::allocMem()
{
	int size = (mnRightX-mnLeftX)*(mnDownY-mnUpY);

	imageData = new uchar[size*mnChannels];
	D = new double[size];
	firstGuess = new double[size];
	b = new double[size];
	x = new double[size];		//vector x, size N
	r = new double[size];		//vector r
	p = new double[size];	
	delta_In = new double[size*9];
	Uk = new double[size*3];
	pIntegral = new double[size];
	IP = new double[size*3];
	w = new double[size];
	LP = new double[size];
	Ak = new double[size*3];
	Bk = new double[size];
	outputDepth = new double[size];
	mpCntNbrs = new int[size];
	memset(outputDepth, 0, sizeof(double)*size);

	int nOffset = 0;
	for (int iterH=mnUpY; iterH<mnDownY; iterH++){
		for (int iterW=mnLeftX; iterW<mnRightX; iterW++){
			if ( mImgCoh.at<float>(iterH, iterW)>0.5){
				b[nOffset] = mImgDepth.at<float>(iterH, iterW)
					*mImgCoh.at<float>(iterH, iterW)* mdLambda;
			}else{
				b[nOffset]= 0;
			}
			
			if ( mImgCoh.at<float>(iterH, iterW)>0.5){
				D[nOffset] = mImgCoh.at<float>(iterH, iterW)* mdLambda;
			}else{
				D[nOffset] = 0;
			}
		
			cv::Vec3b v = mImgView.at<cv::Vec3b>(iterH, iterW);
			imageData[nOffset*3+0] = v[0];
			imageData[nOffset*3+1] = v[1];
			imageData[nOffset*3+2] = v[2];
			if (mImgGuess.empty()){
				firstGuess[nOffset] = mImgDepth.at<float>(iterH, iterW);
			}else{
				firstGuess[nOffset] = mImgGuess.at<float>(iterH, iterW);
			}	
			nOffset++;
		}
	}
}

void fastMattingSolver::loadImage(const cv::Mat& imgI, const cv::Mat& imgD, const cv::Mat& imgC)
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

void fastMattingSolver::setGuess(cv::Mat& imgGuess)
{
	mImgGuess = imgGuess.clone();
}

void fastMattingSolver::setParameters(const double dLambda, const int nMaxIters, const int nPatchSize)
{
	mdLambda = dLambda;
	mnMaxIters = nMaxIters;
	mnPatchSize = nPatchSize;
}

void fastMattingSolver::setMast(bool* pMask)
{
	mask = pMask;
}

void fastMattingSolver::setROI(int nLeftX, int nRightX, int nUpY, int nDownY)
{
	mnLeftX = nLeftX;
	mnRightX = nRightX;
	mnUpY = nUpY;
	mnDownY = nDownY;
}

void fastMattingSolver::solve(cv::Mat& imgResult)
{
	int iterMax = mnMaxIters;
	int size = (mnRightX-mnLeftX)*(mnDownY-mnUpY);

	allocMem();
	setMatrix(); // Uk, delta_In
	if (iterMax<0){
		iterMax = size+1;
	}

	//set first guess of x
	memcpy(x, firstGuess, size*sizeof(double));
	//r = b-A*x
	//use A*x , the result write into w[]
	multiLp(x);
	for(int nLoopImage=0; nLoopImage<size; nLoopImage++){
		r[nLoopImage] = b[nLoopImage]-w[nLoopImage];
	}
	//p = r
	memcpy(p, r, size*sizeof(double));

	//======================
	double rho = multi(r, r, size); //rho = r.t()*r;
	double rho_last = 0;
	double beta = 0;
	double alpha = 0;
	double t=rho;		//error tolerance

	int iterCount = 0;

	// iterative to solve matting equation
	while(sqrt(rho)>mdErrTolorant*sqrt(t) && iterCount<iterMax)
	//while(sqrt(rho)>mdErrTolorant && iterCount<iterMax)
	{
		iterCount++;
		cout<<iterCount<<" ";
		multiLp(p);

		double temp = multi(w, p, size);
		alpha = rho/temp;
		for(int i=0; i<size; i++)
		{
			x[i] = x[i]+alpha*p[i];
			r[i] = r[i]-alpha*w[i];
		}
		rho_last = rho;
		rho = multi(r, r, size);
		cout<<rho<<endl;
		beta = rho/rho_last;
		for(int i=0; i<size; i++)
		{
			p[i] = r[i]+beta*p[i];
		}
	}

	memcpy(outputDepth,x,sizeof(double)*size);
	setResult();
	for (int iterH=mnUpY; iterH<mnDownY; iterH++){
		for(int iterW=mnLeftX; iterW<mnRightX; iterW++){
			int nOffset = (iterH-mnUpY)*(mnRightX-mnLeftX)+(iterW-mnLeftX);
			imgResult.at<float>(iterH, iterW) = outputDepth[nOffset];
		}
	}
	clearMem();
}

void fastMattingSolver::solveWithMask(cv::Mat& imgResult)
{	
	mbUsingMask = true;
	int iterMax = mnMaxIters;
	int size = (mnDownY-mnUpY)*(mnRightX-mnLeftX);

	if(0){
	// decide lambda and patchsize 
		int nArea = 0;
		int nCntBadPix = 0;
		for (int iterPix=0; iterPix<size; iterPix++){
			if(mask[iterPix])
				nArea++;
		}
		for (int iterH=mnUpY; iterH<mnDownY; iterH++){
			for (int iterW=mnLeftX; iterW<mnRightX; iterW++){
				if(mImgCoh.at<float>(iterH, iterW)<0.5)
					nCntBadPix++;
			}
		}
		int nMaxPatch = 30;
		int nMinPatch = 3;
		if((1-nCntBadPix/(float)nArea)<0.3){
			mnPatchSize = nMaxPatch;
		}else if((1-nCntBadPix/(float)nArea)>0.9){
			mnPatchSize = nMinPatch;
		}else {
			mnPatchSize = sqrt(nArea*1.0)/4;
		}
		if(mnPatchSize> (mnDownY-mnUpY))
			mnPatchSize = (mnDownY-mnUpY);
		if(mnPatchSize > (mnRightX-mnLeftX))
			mnPatchSize = (mnRightX-mnLeftX);
		if(mnPatchSize>nMaxPatch) mnPatchSize = nMaxPatch;
		if(mnPatchSize<nMinPatch) mnPatchSize = nMinPatch;
		mnPatchSize = mnPatchSize/2*2+1;
	}
	
	//mnPatchSize = 5;
	//mdLambda = mnPatchSize*mnPatchSize*0.01;

	allocMem();
	setMatrix(); // Uk, delta_In
	if (iterMax<0){
		iterMax = size+1;
	}

	//set first guess of x
	memcpy(x, firstGuess, size*sizeof(double));
	//memset(x, 0, size*sizeof(double));
	//r = b-A*x
	//use A*x , the result write into w[]
	multiLp(x);
	for(int nLoopImage=0; nLoopImage<size; nLoopImage++){
		if (mbUsingMask && mask[nLoopImage]==false){
			r[nLoopImage] = 0;
		}else{
			r[nLoopImage] = b[nLoopImage]-w[nLoopImage];
		}
	}
	//p = r
	memcpy(p, r, size*sizeof(double));

	//======================
	double rho = multi(r, r, size); //rho = r.t()*r;
	double rho_last = 0;
	double beta = 0;
	double alpha = 0;
	double t=rho;		//error tolerance

	int iterCount = 0;
	// iterative to solve matting equation
	while(sqrt(rho)>mdErrTolorant*sqrt(t) && iterCount<iterMax)
	//while(sqrt(rho)>mdErrTolorant && iterCount<iterMax)
	{
		iterCount++;
		cout<<iterCount<<" ";
		multiLp(p);

		double temp = multi(w, p, size);
		alpha = rho/temp;
		for(int i=0; i<size; i++)
		{
			x[i] = x[i]+alpha*p[i];
			r[i] = r[i]-alpha*w[i];
		}
		rho_last = rho;
		rho = multi(r, r, size);
		cout<<rho<<endl;
		beta = rho/rho_last;
		for(int i=0; i<size; i++)
		{
			p[i] = r[i]+beta*p[i];
		}
	}

	memcpy(outputDepth,x,sizeof(double)*size);
	setResult();
	for (int iterH=mnUpY; iterH<mnDownY; iterH++){
		for(int iterW=mnLeftX; iterW<mnRightX; iterW++){
			int nOffset = (iterH-mnUpY)*(mnRightX-mnLeftX)+(iterW-mnLeftX);
			if(mask[nOffset]){
				imgResult.at<float>(iterH, iterW) = outputDepth[nOffset];
			}	
		}
	}
	clearMem();
	mbUsingMask = false;
}

void fastMattingSolver::getResult(cv::Mat& imgResult)
{
	imgResult=mImgResult.clone();
}

// -----------------private-----------------// 
// set UK, delta_In
void fastMattingSolver::setMatrix()
{
	int size = (mnRightX-mnLeftX)*(mnDownY-mnUpY);
	int width = mnRightX-mnLeftX;
	int height = (mnDownY-mnUpY);
	int widthStep = width*mnChannels;
	int nPatchSize = mnPatchSize;
	int nChannels = mnChannels;

	memset(Uk, 0, sizeof(double)*size*3);
	memset(delta_In, 0, sizeof(double)*size*9);

	//Uk, deltaInversed
	for(int nLoopImage=0; nLoopImage<size; nLoopImage++)
	{
		int centerWidth = nLoopImage%width;
		int centerHeight = nLoopImage/width;
		double sumImageData[3] = {0};
		double diff[3] = {0};		//I - Uk
		int countPixel = 0;
		double covar[3][3] = {0.0};
		double delta[3][3] = {0.0};
		double deltaInversed[3][3] = {0.0};

		//get mean Uk
		for(int patchHeight=centerHeight-nPatchSize/2; 
			patchHeight<=centerHeight+nPatchSize/2; patchHeight++){
			if (patchHeight<0){
				continue;
			}else if (patchHeight>=height){
				break;
			}
			for (int patchWidth = centerWidth-nPatchSize/2; 
				patchWidth<=centerWidth+nPatchSize/2; patchWidth++){
				if (patchWidth<0){
					continue;
				}else if (patchWidth>=width){
					break;
				}
				if (mbUsingMask && mask[patchHeight*width+patchWidth] == false){
					continue;
				}
				sumImageData[0] += imageData[patchHeight*widthStep+patchWidth*nChannels];
				sumImageData[1] += imageData[patchHeight*widthStep+patchWidth*nChannels+1];
				sumImageData[2] += imageData[patchHeight*widthStep+patchWidth*nChannels+2];
				countPixel++;
			}
		}//end of local loop
		Uk[nLoopImage*3+0] = sumImageData[0]/countPixel;
		Uk[nLoopImage*3+1] = sumImageData[1]/countPixel;
		Uk[nLoopImage*3+2] = sumImageData[2]/countPixel;
		mpCntNbrs[nLoopImage] = countPixel;

		//get covar
		for(int patchHeight = centerHeight-nPatchSize/2; 
			patchHeight<=centerHeight+nPatchSize/2; patchHeight++){
			if (patchHeight<0){
				continue;
			}else if (patchHeight>=height){
				break;
			}
			for (int patchWidth = centerWidth-nPatchSize/2; 
				patchWidth<=centerWidth+nPatchSize/2; patchWidth++){
				if (patchWidth<0){
					continue;
				}else if (patchWidth>=width){
					break;
				}
				if (mbUsingMask && mask[patchHeight*width+patchWidth] == false){
					continue;
				}
				diff[0] = imageData[patchHeight*widthStep+patchWidth*nChannels]
				- Uk[centerHeight*widthStep+centerWidth*nChannels];
				diff[1] = imageData[patchHeight*widthStep+patchWidth*nChannels+1]
				- Uk[centerHeight*widthStep+centerWidth*nChannels+1];
				diff[2] = imageData[patchHeight*widthStep+patchWidth*nChannels+2]
				- Uk[centerHeight*widthStep+centerWidth*nChannels+2];
				covar[0][0] += diff[0]*diff[0];
				covar[0][1] += diff[0]*diff[1];
				covar[0][2] += diff[0]*diff[2];
				covar[1][1] += diff[1]*diff[1];
				covar[1][2] += diff[1]*diff[2];
				covar[2][2] += diff[2]*diff[2];
			}
		}
		covar[1][0] = covar[0][1];
		covar[2][0] = covar[0][2];
		covar[2][1] = covar[1][2];

		//delta
		for (int i=0; i<3; i++){
			for(int j=0; j<3; j++){
				delta[i][j] = covar[i][j]/(countPixel-1);
				if (i==j){
					delta[i][j] += mdEpsilon/countPixel;
				}
			}
		}

		//covar -> inverse
		getInversed(delta, deltaInversed);
		memcpy(delta_In+nLoopImage*9, deltaInversed, sizeof(double)*9);
	}//end of getting Uk, delta_IN
}

// multiply sparse matrix L to a
void fastMattingSolver::multiLp(double *a)
{
	int size = (mnRightX-mnLeftX)*(mnDownY-mnUpY);
	int width = mnRightX-mnLeftX;
	int height = (mnDownY-mnUpY);
	int widthStep = width*mnChannels;
	int nPatchSize = mnPatchSize;
	int nChannels = mnChannels;

	// IP
	for(int nLoopImage=0; nLoopImage<size; nLoopImage++)
	{
		int centerWidth = nLoopImage%width;
		int centerHeight = nLoopImage/width;
		if(mbUsingMask && mask[nLoopImage]==false){
			IP[nLoopImage*3+0] = IP[nLoopImage*3+1] = IP[nLoopImage*3+2] = 0;
		}else{
			//I*p
			IP[nLoopImage*3+0] = imageData[centerHeight*widthStep+centerWidth*nChannels]* a[nLoopImage];
			IP[nLoopImage*3+1] = imageData[centerHeight*widthStep+centerWidth*nChannels+1]* a[nLoopImage];
			IP[nLoopImage*3+2] = imageData[centerHeight*widthStep+centerWidth*nChannels+2]* a[nLoopImage];
		}
	}
	integralImage(IP, width, height, 3, IP);
	if (mbUsingMask){
		for(int iterPix=0; iterPix<size; iterPix++){
			if(mask[iterPix]==false)
				a[iterPix] = 0;
		}
	}
	integralImage(a, width, height, 1, pIntegral);

	memset(Ak, 0, sizeof(double)*size*3);
	memset(Bk, 0, sizeof(double)*size);

	//fast multi: get w = A*p
	//get I*p£¬integral image
	for(int nLoopImage=0; nLoopImage<size; nLoopImage++)
	{
		if(mbUsingMask && mask[nLoopImage]==false){
			continue;
		}
		int centerWidth = nLoopImage%width;
		int centerHeight = nLoopImage/width;
		int ptWidthRight = (centerWidth + nPatchSize/2)>=width ? width-1 : centerWidth + nPatchSize/2;
		int ptHeightDown = (centerHeight + nPatchSize/2)>=height ? height-1 : centerHeight + nPatchSize/2;
		int ptHeightTop = (centerHeight - nPatchSize/2)<0 ? 0 : centerHeight - nPatchSize/2;
		int ptWidthLeft = (centerWidth - nPatchSize/2)<0 ? 0 : centerWidth - nPatchSize/2;
		//int countPixel = (ptWidthRight-ptWidthLeft+1)*(ptHeightDown-ptHeightTop+1);
		int countPixel = mpCntNbrs[nLoopImage];

		double sumP = getSum(pIntegral, width, height, centerWidth, centerHeight, nPatchSize);
		double meanP = sumP/countPixel;
		//get Ak
		for (int i=0; i<3; i++){
			for (int j=0; j<3; j++){
				double sumIP = getSum(IP, width, height,centerWidth, centerHeight, nPatchSize, 3, j);
				Ak[nLoopImage*3+i] += (sumIP/countPixel -Uk[nLoopImage*3+j]*meanP)
					*delta_In[nLoopImage*9+i*3+j];
			}
		}
		//Bk
		Bk[nLoopImage] = meanP;
		for (int i=0; i<3; i++)
		{
			Bk[nLoopImage] -= Ak[nLoopImage*3+i]*Uk[nLoopImage*3+i];
		}
	}//end of for: image Loop
	integralImage(Ak, width, height, 3, Ak);
	integralImage(Bk, width, height, 1, Bk);

	memset(w, 0, sizeof(double)*size);
	for (int nLoopImage=0; nLoopImage<size; nLoopImage++)
	{
		if (mbUsingMask && mask[nLoopImage]==false){
			continue;
		}
		int centerWidth = nLoopImage%width;
		int centerHeight = nLoopImage/width;
		int ptWidthRight = (centerWidth + nPatchSize/2)>=width ? width-1 : centerWidth + nPatchSize/2;
		int ptHeightDown = (centerHeight + nPatchSize/2)>=height ? height-1 : centerHeight + nPatchSize/2;
		int ptHeightTop = (centerHeight - nPatchSize/2)<0 ? 0 : centerHeight - nPatchSize/2;
		int ptWidthLeft = (centerWidth - nPatchSize/2)<0 ? 0 : centerWidth - nPatchSize/2;
		double temp = 0;
		//int countPixel = (ptWidthRight-ptWidthLeft+1)*(ptHeightDown-ptHeightTop+1);
		int countPixel = mpCntNbrs[nLoopImage];
		for (int i = 0; i<3; i++)
		{
			double sumAk = getSum(Ak, width, height, centerWidth, centerHeight, nPatchSize, 3, i);
			temp += sumAk*imageData[centerHeight*widthStep+centerWidth*nChannels+i];
		}
		double sumBk = getSum(Bk, width, height, centerWidth, centerHeight, nPatchSize);
		LP[nLoopImage] = countPixel*a[nLoopImage] - temp - sumBk;
		w[nLoopImage] = LP[nLoopImage] + a[nLoopImage]*D[nLoopImage];
	}
}

// outputdepth->imgResult
void fastMattingSolver::setResult()
{
	if(mImgResult.empty())
		return;

	int nOffset = 0;
	for (int iterH=mnUpY; iterH<mnDownY; iterH++){
		for (int iterW=mnLeftX; iterW<mnRightX; iterW++){
			if (!mbUsingMask){
				mImgResult.at<float>(iterH, iterW) = outputDepth[nOffset];
			}else if(mask[nOffset]==true){
				mImgResult.at<float>(iterH, iterW) = outputDepth[nOffset];
			}
			nOffset++;
		//	mImgResult.at<float>(iterH, iterW) = mpCntNbrs[nOffset++];
		}
	}
}