#include <stack>
#include <iostream>

#include <highgui.h>

#include "KDSegment.h"
#include "../common/matrixOp.h"
#include "../common/imageOp.h"

using namespace std;

cv::Mat imgOutput; // for debug

void splitNode(double *dMatCohIntegrate, int nWidth, int nHeight,
	KDNode ndToSplit, stack<KDNode>& listToSplit)
{
	// find the best place to cut
	int nROIXLeft(ndToSplit.nXLeft), nROIXRight(ndToSplit.nXRight);
	int nROIYUp(ndToSplit.nYUp), nROIYDown(ndToSplit.nYDown);
	int nMinPatchRadius = (nROIXRight-nROIXLeft)>(nROIYDown-nROIYUp) ? 
		(nROIXRight-nROIXLeft)/10 : (nROIYDown-nROIYUp)/10;
	if (nMinPatchRadius<10) nMinPatchRadius = 15;

	double dMinCostRow, dMinCostCol;
	dMinCostRow = dMinCostCol = (nROIXRight-nROIXLeft)*(nROIYDown-nROIYUp);
	int nMinIdxY = -1; int nMinIdxX = -1;

	for (int iterC=nROIXLeft+nMinPatchRadius; iterC<nROIXRight-nMinPatchRadius; iterC++){
		int nCntPixL = (nROIYDown-nROIYUp)*(iterC-nROIXLeft);
		int nCntPixR = (nROIYDown-nROIYUp)*(nROIXRight-iterC-1);
		double dSumCohL = getSumRect(dMatCohIntegrate, nWidth, nHeight, 
			nROIYUp, nROIYDown, nROIXLeft, iterC);
		double dSumCohR = getSumRect(dMatCohIntegrate, nWidth, nHeight,
			nROIYUp, nROIYDown, iterC, nROIXRight);
		dSumCohL = (dSumCohL-0)>(nCntPixL-dSumCohL) ? (nCntPixL-dSumCohL) : (dSumCohL-0);
		dSumCohR = (dSumCohR-0)>(nCntPixR-dSumCohR) ? (nCntPixR-dSumCohR) : (dSumCohR-0);
		double dSumCost = dSumCohL+dSumCohR;
		dSumCost += nROIXLeft+nROIXRight - 2*iterC;
		if(dSumCost<dMinCostCol){
			dMinCostCol = dSumCost;
			nMinIdxX = iterC;
		}
	}
	for (int iterR=nROIYUp+nMinPatchRadius; iterR<nROIYDown-nMinPatchRadius; iterR++){
		int nCntPixD = (nROIYDown-iterR-1)*(nROIXRight-nROIXLeft);
		int nCntPixU = (iterR-nROIYUp)*(nROIXRight-nROIXLeft);
		double dSumCohU = getSumRect(dMatCohIntegrate, nWidth, nHeight, 
			nROIYUp, iterR, nROIXLeft, nROIXRight);
		double dSumCohD = getSumRect(dMatCohIntegrate, nWidth, nHeight,
			iterR, nROIYDown, nROIXLeft, nROIXRight);
		dSumCohU = (dSumCohU-0)>(nCntPixU-dSumCohU) ? (nCntPixU-dSumCohU) : (dSumCohU-0);
		dSumCohD = (dSumCohD-0)>(nCntPixD-dSumCohD) ? (nCntPixD-dSumCohD) : (dSumCohD-0);
		double dSumCost = dSumCohU+dSumCohD;
		dSumCost += nROIYUp+nROIYDown-2*iterR;
		if(dSumCost<dMinCostRow){
			dMinCostRow = dSumCost;
			nMinIdxY = iterR;
		}
	}
	if (dMinCostRow<dMinCostCol && nMinIdxY!=-1){
		// cut along nMinIdxY
		KDNode ndUp, ndDown;
		ndUp.nXLeft = nROIXLeft; ndUp.nXRight = nROIXRight;
		ndUp.nYUp = nROIYUp; ndUp.nYDown = nMinIdxY;
		ndUp.eNodeType = ndUp.TOBESPLIT;
		ndDown.nXLeft = nROIXLeft; ndDown.nXRight = nROIXRight;
		ndDown.nYUp = nMinIdxY; ndDown.nYDown = nROIYDown;
		ndDown.eNodeType = ndDown.TOBESPLIT;
		listToSplit.push(ndUp);
		listToSplit.push(ndDown);
		// draw a line on imgOutput
		for (int iterC=nROIXLeft; iterC<nROIXRight; iterC++){
			cv::Vec3b v(0, 0 ,255);
			imgOutput.at<cv::Vec3b>(nMinIdxY, iterC) = v;
		}
		cout<<"Y="<<nMinIdxY<<endl;
	}else if(nMinIdxX!=-1){
		// cut along nMinIdxX
		KDNode ndLeft, ndRight;
		ndLeft.nYUp = ndRight.nYUp = nROIYUp;
		ndLeft.nYDown = ndRight.nYDown = nROIYDown;
		ndLeft.nXLeft = nROIXLeft;
		ndLeft.nXRight = nMinIdxX;
		ndRight.nXLeft = nMinIdxX;
		ndRight.nXRight = nROIXRight;
		ndRight.eNodeType = ndLeft.eNodeType = ndLeft.TOBESPLIT;
		listToSplit.push(ndLeft);
		listToSplit.push(ndRight);
		// draw a line on imgOutput
		for (int iterR=nROIYUp; iterR<nROIYDown; iterR++){
			cv::Vec3b v(0, 0 ,255);
			imgOutput.at<cv::Vec3b>(iterR, nMinIdxX) = v;
		}
		cout<<"X="<<nMinIdxX<<endl;
	}else{
		cout<<"something is wrong!"<<endl;
	}
	//showImage(imgOutput);

	return;
}

void splitNode2(double *dMatCohIntegrate, int nWidth, int nHeight,
	KDNode ndToSplit, stack<KDNode>& listToSplit)
{
	// find the best place to cut
	int nROIXLeft(ndToSplit.nXLeft), nROIXRight(ndToSplit.nXRight);
	int nROIYUp(ndToSplit.nYUp), nROIYDown(ndToSplit.nYDown);
	//int nMinPatchRadius = (nROIXRight-nROIXLeft)>(nROIYDown-nROIYUp) ? 
	//	(nROIXRight-nROIXLeft)/10 : (nROIYDown-nROIYUp)/10;
	//if (nMinPatchRadius<10) nMinPatchRadius = 15;
	int nMinPatchRadius = 15;
	if ((ndToSplit.nXRight-ndToSplit.nXLeft)<2*nMinPatchRadius
		&& (ndToSplit.nYDown-ndToSplit.nYUp)<2*nMinPatchRadius){
		nMinPatchRadius = 1;
	}

	double dMinCostRow, dMinCostCol;
	dMinCostRow = dMinCostCol = (nROIXRight-nROIXLeft)*(nROIYDown-nROIYUp);
	int nMinIdxY = -1; int nMinIdxX = -1;

	for (int iterC=nROIXLeft+nMinPatchRadius; iterC<nROIXRight-nMinPatchRadius; iterC++){
		int nCntPixL = (nROIYDown-nROIYUp)*(iterC-nROIXLeft);
		int nCntPixR = (nROIYDown-nROIYUp)*(nROIXRight-iterC-1);
		double dSumCohL = getSumRect(dMatCohIntegrate, nWidth, nHeight, 
			nROIYUp, nROIYDown, nROIXLeft, iterC);
		double dSumCohR = getSumRect(dMatCohIntegrate, nWidth, nHeight,
			nROIYUp, nROIYDown, iterC, nROIXRight);
		dSumCohL = (dSumCohL/nCntPixL-0)>(1-dSumCohL/nCntPixL) ? 
			(1-dSumCohL/nCntPixL) : (dSumCohL/nCntPixL-0);
		dSumCohR = (dSumCohR/nCntPixR-0)>(1-dSumCohR/nCntPixR) ? 
			(1-dSumCohR/nCntPixR) : (dSumCohR/nCntPixR-0);
		double dSumCost = dSumCohL+dSumCohR;
		//dSumCost += nROIXLeft+nROIXRight/* - 2*iterC*/;
		if(dSumCost<dMinCostCol){
			dMinCostCol = dSumCost;
			nMinIdxX = iterC;
		}
	}
	for (int iterR=nROIYUp+nMinPatchRadius; iterR<nROIYDown-nMinPatchRadius; iterR++){
		int nCntPixD = (nROIYDown-iterR-1)*(nROIXRight-nROIXLeft);
		int nCntPixU = (iterR-nROIYUp)*(nROIXRight-nROIXLeft);
		double dSumCohU = getSumRect(dMatCohIntegrate, nWidth, nHeight, 
			nROIYUp, iterR, nROIXLeft, nROIXRight);
		double dSumCohD = getSumRect(dMatCohIntegrate, nWidth, nHeight,
			iterR, nROIYDown, nROIXLeft, nROIXRight);
		dSumCohU = (dSumCohU/nCntPixU-0)>(1-dSumCohU/nCntPixU) 
			? (1-dSumCohU/nCntPixU) : (dSumCohU/nCntPixU-0);
		dSumCohD = (dSumCohD/nCntPixD-0)>(1-dSumCohD/nCntPixD) 
			? (1-dSumCohD/nCntPixD) : (dSumCohD/nCntPixD-0);
		double dSumCost = dSumCohU+dSumCohD;
		//dSumCost += nROIYUp+nROIYDown/*-2*iterR*/;
		if(dSumCost<dMinCostRow){
			dMinCostRow = dSumCost;
			nMinIdxY = iterR;
		}
	}
	if (dMinCostRow<dMinCostCol && nMinIdxY!=-1){
		// cut along nMinIdxY
		KDNode ndUp, ndDown;
		ndUp.nXLeft = nROIXLeft; ndUp.nXRight = nROIXRight;
		ndUp.nYUp = nROIYUp; ndUp.nYDown = nMinIdxY;
		ndUp.eNodeType = ndUp.TOBESPLIT;
		ndDown.nXLeft = nROIXLeft; ndDown.nXRight = nROIXRight;
		ndDown.nYUp = nMinIdxY; ndDown.nYDown = nROIYDown;
		ndDown.eNodeType = ndDown.TOBESPLIT;
		listToSplit.push(ndUp);
		listToSplit.push(ndDown);
		// draw a line on imgOutput
		for (int iterC=nROIXLeft; iterC<nROIXRight; iterC++){
			cv::Vec3b v(0, 0 ,255);
			imgOutput.at<cv::Vec3b>(nMinIdxY, iterC) = v;
		}
		cout<<"Y="<<nMinIdxY<<endl;
	}else if(nMinIdxX!=-1){
		// cut along nMinIdxX
		KDNode ndLeft, ndRight;
		ndLeft.nYUp = ndRight.nYUp = nROIYUp;
		ndLeft.nYDown = ndRight.nYDown = nROIYDown;
		ndLeft.nXLeft = nROIXLeft;
		ndLeft.nXRight = nMinIdxX;
		ndRight.nXLeft = nMinIdxX;
		ndRight.nXRight = nROIXRight;
		ndRight.eNodeType = ndLeft.eNodeType = ndLeft.TOBESPLIT;
		listToSplit.push(ndLeft);
		listToSplit.push(ndRight);
		// draw a line on imgOutput
		for (int iterR=nROIYUp; iterR<nROIYDown; iterR++){
			cv::Vec3b v(0, 0 ,255);
			imgOutput.at<cv::Vec3b>(iterR, nMinIdxX) = v;
		}
		cout<<"X="<<nMinIdxX<<endl;
	}else{
		cout<<"something is wrong!"<<endl;
	}
	//showImage(imgOutput);

	return;
}

vector<KDNode> buidKDSegments(double* dMatCoh, int nWidth, int nHeight)
{	
	double* dMatCohIntegrate = new double[nWidth*nHeight];
	//integralImage(dMatCoh, nWidth, nHeight, 1, dMatCohIntegrate);
	double* dMatchDisc = new double[nWidth*nHeight];
	for (int i=0; i<nWidth*nHeight; i++){
		if (dMatCoh[i]>0.9){
			dMatchDisc[i] = 1;
		}else{
			dMatchDisc[i] = 0;
		}
	}
	integralImage(dMatchDisc, nWidth, nHeight, 1, dMatCohIntegrate);
	
	imgOutput.create(nHeight, nWidth, CV_8UC3); // for debug
	for (int iterR=0; iterR<nHeight; iterR++){
		for(int iterC=0; iterC<nWidth; iterC++){
			cv::Vec3b v;
			v[0] = v[1] = v[2] = dMatchDisc[iterR*nWidth+iterC]*255;
			imgOutput.at<cv::Vec3b>(iterR, iterC) = v;
		}
	}

	vector<KDNode> listSegments;
	stack<KDNode> listToSplit;

	// push the whole image into the stack
	KDNode ndStart;
	ndStart.nXLeft = 0;
	ndStart.nXRight = nWidth;
	ndStart.nYUp = 0;
	ndStart.nYDown = nHeight;
	ndStart.eNodeType = ndStart.TOBESPLIT;
	//ndStart.dCohRate = getSumRect(dMatCohIntegrate, nWidth, nHeight, 0, nHeight, 0, nWidth)/(nWidth*nHeight);
	listToSplit.push(ndStart);

	int nSmallPatchThresh = 500;
	double dCohLowThresh = 0.10;
	double dCohUpThresh = 0.90;
	while(!listToSplit.empty()){
		KDNode ndToSplit = listToSplit.top();
		listToSplit.pop();
		ndToSplit.dCohRate = getSumRect(dMatCohIntegrate, nWidth, nHeight, 
			ndToSplit.nYUp, ndToSplit.nYDown, ndToSplit.nXLeft, ndToSplit.nXRight)
 			/((ndToSplit.nYDown-ndToSplit.nYUp)*(ndToSplit.nXRight-ndToSplit.nXLeft));
		double dSizePenalty = 1;
		if (ndToSplit.getArea()<(nWidth*nHeight)/50){
			dSizePenalty = 1 - ndToSplit.getArea()/((nWidth*nHeight)/50.0);
			dCohLowThresh = 0.10*(dSizePenalty+1);
			dCohUpThresh = 0.90*(1-dSizePenalty);
		}else{
			dCohLowThresh = 0.10;
			dCohUpThresh = 0.90;
		}

		if(ndToSplit.dCohRate<dCohLowThresh ){
			//&& (-ndToSplit.nYUp+ndToSplit.nYDown)*(ndToSplit.nXRight-ndToSplit.nXLeft)<50000){
			// low coh
			ndToSplit.eNodeType = ndToSplit.LOWCOH;
			listSegments.push_back(ndToSplit);
		}else if(ndToSplit.dCohRate>dCohUpThresh){
			// high coh
			ndToSplit.eNodeType = ndToSplit.HIGHCOH;
			listSegments.push_back(ndToSplit);
		}else if(ndToSplit.getArea()<nSmallPatchThresh){
			// small piece
			ndToSplit.eNodeType = ndToSplit.SMALLPATCH;
			listSegments.push_back(ndToSplit);
		}else{
			splitNode2(dMatCohIntegrate, nWidth, nHeight, ndToSplit, listToSplit);
		}
	}

	cv::imwrite("segmentation.jpg", imgOutput);
	//showImage(imgOutput);
	delete[] dMatCohIntegrate;
	return listSegments;
}

