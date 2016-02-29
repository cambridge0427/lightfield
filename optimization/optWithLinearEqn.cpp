/*
    input:
        data name: without extension. eg, buddha.
        use revised initial results: 0 or 1
            Use revised initial result or original result from structure tensor
        optimization method:
            NoOpt = 0: no optimization
            MRFOpt = 1: use Markov Random Field
            FastMatting = 2: use fasting matting to solve the linear equation
            ClassicOpt = 3: use Conjugate Gradient to solve the linear equation
            MixOpt = 4: use mixed method to solve the linear equation
        segment: 0 or 1
            This is only applicable for FastMatting, ClassicOpt and MixOpt
    Usage:
        optimize <data name> <use revised initial results> <optimization method> [<segment>]
*/

/*
    The input h5 file should contain:
        centerview, 
        initial depth estimations (DEPTHS, DEPTHT)
        coherence with or without revision (COHS, COHT, COHSR, COHTR)
*/

#include <stdio.h>
#include <iostream>
#include <vector>
#include <time.h>
#include <fstream>

#include <hdf5.h>
#include <hdf5_hl.h>
#include <highgui.h>

#include "../common/matrixOp.h"
#include "../common/imageOp.h"

#include "../MSSegment/msImageProcessor.h"
#include "../MSSegment/BgImage.h"
#include "../MSSegment/BgEdgeDetect.h"

#include "solverClassic.h"
#include "solverFastMatting.h"
#include "solverMRF.h"

using namespace std;

enum OptMethod { NoOpt, MRFOpt, FastMatting, ClassicOpt, MixOpt }; // 0, 1, 2, 3, 4
string strBaseInputDir = "../data/initial_results/";
string strBaseOutputDir = "../data/results/";

cv::Mat optimize(cv::Mat& imgDepthT, cv::Mat& imgCoh, cv::Mat& imgView, cv::Mat& imgLabels,
    int nOptMethod, bool bSegment, string strFilename);

bool segmentImg(const cv::Mat& imgInput, const cv::Mat& imgDepth, cv::Mat& imgResult,
    int** lables, int& regionCount);

bool importData(cv::Mat& imgView, cv::Mat& imgDepthS, cv::Mat& imgDepthT,
    cv::Mat& imgCohS, cv::Mat& imgCohT, cv::Mat& imgGT,
    const char* const strFilename, const bool bRevise);

void countErrors(const cv::Mat& imgView, const cv::Mat& imgGT,
    const cv::Mat& imgResult, cv::Mat& imgDiff, ofstream& outputDir);

void setLabelImg(const cv::Mat& imgView, const int nData,
    const int *pLabels, const int nRegionCnt, cv::Mat& imgLabels);

void reviseCoh(cv::Mat& imgCohS, cv::Mat& imgCohT, float fThreshold);

void truncateDepth(cv::Mat& imgDepth, float fMin, float fMax);

void mergeDepth(const cv::Mat& imgDepthS, const cv::Mat& imgDepthT, cv::Mat& imgCohS,
    const cv::Mat& imgCohT, cv::Mat& imgDepth, cv::Mat& imgCoh);

int main(int argc, char* argv[])
{
    if (argc < 4){
        cerr<<"Usage: optimize <data name> <use revised initial results> <optimization method> [<segment>]"<<endl;
        cerr<<"Please see the document for more explanation."<<endl;
        return -1;
    }

    string strFilename = argv[1];
    bool bRevised = (bool)atoi(argv[2]);
    OptMethod nOptMethod = (OptMethod)atoi(argv[3]);
    bool bSegment = true;
    if (nOptMethod == FastMatting || nOptMethod == ClassicOpt || nOptMethod == MixOpt){
        if (argc < 5){
            cout<<"\'segment\' not specified. Use default value: 1."<<endl;
        }
        else{
            bSegment = (bool)atoi(argv[4]);
        }
    }else{
        bSegment = false;
    }

    cv::Mat imgView, imgDepthS, imgDepthT, imgCohS, imgCohT, imgGT, imgResult, imgDiff, imgLabels;
    string strInputFilename = strBaseInputDir + "/" + strFilename + "/" + "localEstimation.h5";
    if (!importData(imgView, imgDepthS, imgDepthT, imgCohS, imgCohT, imgGT, strInputFilename.c_str(), bRevised)){
        cerr<<"Unable to load data"<<endl;
        return -1;
    }

    string strOutputDir = strBaseOutputDir + strFilename;
    // Create output dir
    string sys_cmd = "mkdir " + strOutputDir;
    system(sys_cmd.c_str());

    //showImage(imgDepthT);showImage(imgCohT);showImage(imgCohTR);
    //showImage(imgDepthS);showImage(imgCohS);showImage(imgCohSR);
    int nWidth = imgView.cols;
    int nHeight = imgView.rows;
    int nChannels = imgView.channels();

    // merge depth maps
    cv::Mat imgDepth, imgCoh;
    imgDepth.create(nHeight, nWidth, CV_32F);
    imgCoh.create(nHeight, nWidth, CV_32F);
    reviseCoh(imgCohS, imgCohT, 0.95);
    mergeDepth(imgDepthS, imgDepthT, imgCohS, imgCohT, imgDepth, imgCoh);

    clock_t init, final;
    init=clock();

    imgResult = optimize(imgDepth, imgCoh, imgView, imgLabels, nOptMethod, bSegment, strFilename);
    final=clock()-init;

    double dMin, dMax;
    cv::minMaxLoc(imgDepth, &dMin, &dMax);
    truncateDepth(imgResult, dMin, dMax);

    // count err
    // output
    ofstream output;
    output.open(strOutputDir + "/record.txt", ios::app);
    output<<"Data name:"<<strFilename<<", revise initial depth value:"<<bRevised;
    output<<", optimization method:"<<nOptMethod<<", segmented:"<<bSegment<<endl;
    countErrors(imgView, imgGT, imgResult, imgDiff, output);
    output<<", time:"<<(double)final / ((double)CLOCKS_PER_SEC)<<endl;
    output.close();

    // save result
    // diff image
    if(!imgDiff.empty()){
        string strDiffImageName = strOutputDir + "/diff_opt" + to_string(nOptMethod);
        if (bSegment){
            strDiffImageName += "_seg";
        }
        strDiffImageName += ".jpg";
        saveImage(strDiffImageName.c_str(), imgDiff);
    }
    // segmentation result
    if(!imgLabels.empty()){
        string strLabelImageName = strOutputDir + "/segmentation.jpg";
        saveImage(strLabelImageName.c_str(), imgLabels);
    }
    // actual optimized result
    string strResultImageName = strOutputDir + "/result_opt" + to_string(nOptMethod);
    if (bSegment){
        strResultImageName += "_seg";
    }
    strResultImageName += ".jpg";
    saveImage(strResultImageName.c_str(), imgResult);
    saveImage((strOutputDir + "/imgView.jpg").c_str(), imgView);
    saveImage((strOutputDir + "/imgCohMerged.jpg").c_str(), imgCoh);

    cout<<"Done!!!"<<endl;
    return 0;
}

cv::Mat optimize(cv::Mat& imgDepth, cv::Mat& imgCoh, cv::Mat& imgView, cv::Mat& imgLabels,
    int nOptMethod, bool bSegment, string strFilename)
{
    int nWidth = imgView.cols; 
    int nHeight = imgView.rows;
    int nChannels = imgView.channels();

    cv::Mat imgResult;
    if (nOptMethod == NoOpt){
        imgResult = imgDepth.clone();
        return imgResult;
    }else if(nOptMethod == MRFOpt){
        solverMRF *solver = new solverMRF;
        solver->loadImage(imgView, imgDepth, imgCoh);
        imgResult.create(nHeight, nWidth, CV_32F);
        solver->solve(imgResult);
        solver->getResult(imgResult);
        delete solver;
        return imgResult;
    }else if(nOptMethod == FastMatting && bSegment == false){
        imgResult.create(nHeight, nWidth, CV_32F);
        int nPatchSize = 5;
        double dLambda = 0.25;
        fastMattingSolver *solver = new fastMattingSolver;
        solver->setParameters(dLambda, 1000, nPatchSize);
        solver->loadImage(imgView, imgDepth, imgCoh);
        //solver->setROI(0, 100, 0, 100);
        solver->solve(imgResult);
        solver->getResult(imgResult);
        delete solver;
        return imgResult;
    }else if(nOptMethod == ClassicOpt && bSegment == false){
        imgResult.create(nHeight, nWidth, CV_32F);
        // parameters are good, do NOT change it!!
        int nPatchSize = 9;
        double dLambda = nPatchSize * nPatchSize *0.0001;
        classicSolver *solver = new classicSolver;
        solver->setParameters(dLambda, -1, nPatchSize/2);
        solver->loadImage(imgView, imgDepth, imgCoh);
        //solver->setROI(200, 300, 275, 375);
        //solver->setROI(455, 500, 455, 500);
        imgResult.create(nHeight, nWidth, CV_32F);
        solver->solve(imgResult, false);
        solver->getResult(imgResult);
        delete solver;
        return imgResult;
    }else if (bSegment){
        
        // segmentation
        int nRegionCnt = 0;
        int *pLabels(NULL);
        bool bSegSuc = segmentImg(imgView, imgDepth, imgLabels, &pLabels, nRegionCnt);

        int nPatchFast = 15;
        int nPatchClassic = 9;
        double dLambdaFast = nPatchFast*nPatchFast*0.01;
        double dLambdaClassic = nPatchClassic*nPatchClassic*0.0001;
        
        // optimization
        if (nOptMethod == FastMatting){
            imgResult.create(nHeight, nWidth, CV_32F);
            fastMattingSolver *solver = new fastMattingSolver;
            solver->setParameters(dLambdaFast, -1, nPatchFast);
            solver->loadImage(imgView, imgDepth, imgCoh);

            cv::Mat imgMask(nHeight, nWidth, CV_8U);
            for (int iterRegion=0; iterRegion<nRegionCnt; iterRegion++){
                // find ROI
                int nLeftX(nWidth), nRightX(0), nUpY(nHeight), nDownY(0);
                for (int iterPix=0; iterPix<nHeight*nWidth; iterPix++){
                    if (pLabels[iterPix] == iterRegion){
                        imgMask.at<uchar>(iterPix/nWidth, iterPix%nWidth) = 255;
                        int nW = iterPix%nWidth;
                        int nH = iterPix/nWidth;
                        if (nW<nLeftX)  nLeftX = nW;
                        if (nW>=nRightX)    nRightX = nW+1;
                        if (nH<nUpY) nUpY = nH;
                        if (nH>=nDownY) nDownY = nH+1;
                    }else{
                        imgMask.at<uchar>(iterPix/nWidth, iterPix%nWidth) = 0;
                    }
                }
                int nSize = (nDownY-nUpY)*(nRightX-nLeftX);
                bool *pMask = new bool[nSize];
                memset(pMask, 0, sizeof(bool)*nSize);
                for (int iterH=nUpY; iterH<nDownY; iterH++){
                    for(int iterW=nLeftX; iterW<nRightX; iterW++){
                        int nOffset = (iterH-nUpY)*(nRightX-nLeftX)+(iterW-nLeftX);
                        if (pLabels[iterH*nWidth+iterW] == iterRegion){
                            pMask[nOffset] = true;
                        }   
                    }
                }
                solver->setROI(nLeftX, nRightX, nUpY, nDownY);
                solver->setMast(pMask);
                solver->solveWithMask(imgResult);
                solver->getResult(imgResult);
                //showImage(imgResult);
                delete[] pMask;
            }
            solver->getResult(imgResult);
            delete solver;
            delete[] pLabels;
            //return imgResult;
        }else if(nOptMethod == ClassicOpt){
            imgResult.create(nHeight, nWidth, CV_32F);
            classicSolver *solver = new classicSolver;
            solver->setParameters(dLambdaClassic, 10, nPatchClassic/2);
            solver->loadImage(imgView, imgDepth, imgCoh);
        
            cv::Mat imgMask(nHeight, nWidth, CV_8U);
            for (int iterRegion=0; iterRegion<nRegionCnt; iterRegion++){
                // find ROI
                int nLeftX(nWidth), nRightX(0), nUpY(nHeight), nDownY(0);
                for (int iterPix=0; iterPix<nHeight*nWidth; iterPix++){
                    if (pLabels[iterPix] == iterRegion){
                        imgMask.at<uchar>(iterPix/nWidth, iterPix%nWidth) = 255;
                        int nW = iterPix%nWidth;
                        int nH = iterPix/nWidth;
                        if (nW<nLeftX)  nLeftX = nW;
                        if (nW>=nRightX)    nRightX = nW+1;
                        if (nH<nUpY) nUpY = nH;
                        if (nH>=nDownY) nDownY = nH+1;
                    }else{
                        imgMask.at<uchar>(iterPix/nWidth, iterPix%nWidth) = 0;
                    }
                }
                int nSize = (nDownY-nUpY)*(nRightX-nLeftX);
                bool *pMask = new bool[nSize];
                memset(pMask, 0, sizeof(bool)*nSize);
                for (int iterH=nUpY; iterH<nDownY; iterH++){
                    for(int iterW=nLeftX; iterW<nRightX; iterW++){
                        int nOffset = (iterH-nUpY)*(nRightX-nLeftX)+(iterW-nLeftX);
                        if (pLabels[iterH*nWidth+iterW] == iterRegion){
                            pMask[nOffset] = true;
                        }   
                    }
                }
                solver->setROI(nLeftX, nRightX, nUpY, nDownY);
                solver->setMast(pMask);
                solver->solve(imgResult, true);
                solver->getResult(imgResult);
                //showImage(imgResult);
                delete[] pMask;
            }
            solver->getResult(imgResult);
            delete solver;
            delete[] pLabels;
            //return imgResult;
        }else if(nOptMethod == MixOpt){
            imgResult.create(nHeight, nWidth, CV_32F);
            fastMattingSolver *solverFast = new fastMattingSolver;
            classicSolver *solverClassic = new classicSolver;
            solverFast->loadImage(imgView, imgDepth, imgCoh);
            solverClassic->loadImage(imgView, imgDepth, imgCoh);

            solverFast->setParameters(dLambdaFast, 100, nPatchFast);
            solverClassic->setParameters(dLambdaClassic, 100, nPatchClassic/2);

            cv::Mat imgMask(nHeight, nWidth, CV_8U);
            for (int iterRegion=0; iterRegion<nRegionCnt; iterRegion++){
                // find ROI
                int nLeftX(nWidth), nRightX(0), nUpY(nHeight), nDownY(0);
                for (int iterPix=0; iterPix<nHeight*nWidth; iterPix++){
                    if (pLabels[iterPix] == iterRegion){
                        imgMask.at<uchar>(iterPix/nWidth, iterPix%nWidth) = 255;
                        int nW = iterPix%nWidth;
                        int nH = iterPix/nWidth;
                        if (nW<nLeftX)  nLeftX = nW;
                        if (nW>=nRightX)    nRightX = nW+1;
                        if (nH<nUpY) nUpY = nH;
                        if (nH>=nDownY) nDownY = nH+1;
                    }else{
                        imgMask.at<uchar>(iterPix/nWidth, iterPix%nWidth) = 0;
                    }
                }
                int nSize = (nDownY-nUpY)*(nRightX-nLeftX);
                bool *pMask = new bool[nSize];
                memset(pMask, 0, sizeof(bool)*nSize);
                for (int iterH=nUpY; iterH<nDownY; iterH++){
                    for(int iterW=nLeftX; iterW<nRightX; iterW++){
                        int nOffset = (iterH-nUpY)*(nRightX-nLeftX)+(iterW-nLeftX);
                        if (pLabels[iterH*nWidth+iterW] == iterRegion){
                            pMask[nOffset] = true;
                        }
                    }
                }
                if(nSize>10000){
                    solverFast->setROI(nLeftX, nRightX, nUpY, nDownY);
                    solverFast->setMast(pMask);
                    solverFast->solveWithMask(imgResult);
                    //solverFast->getResult(imgResult);
                }else{
                    solverClassic->setROI(nLeftX, nRightX, nUpY, nDownY);
                    solverClassic->setMast(pMask);
                    solverClassic->solve(imgResult, true);
                    //solverClassic->getResult(imgResult);
                }
                //showImage(imgResult);
                delete[] pMask;
            }
            delete solverClassic;
            delete solverFast;
            delete[] pLabels;
            //return imgResult;
        }else{
            return cv::Mat(nHeight, nWidth, CV_32F);
        }
        return imgResult;
    }

    return cv::Mat();
}

bool segmentImg(const cv::Mat& imgInput, const cv::Mat& imgDepth, cv::Mat& imgResult,
    int** labels, int& regionCount)
{
    int nWidth = imgInput.cols;
    int nHeight = imgInput.rows;
    
    //obtain image type (color or grayscale)
    imageType gtype;
    if(imgInput.channels()==3)
        gtype = COLOR;
    else
        gtype = GRAYSCALE;

    BgImage* cbgImage_ = new BgImage();
    BgImage* segmImage_ = new BgImage();

    // segmentation parameters
    int sigmaS(16), kernelSize(2);
    //int  minRegion= nWidth*nHeight*0.01;
    int minRegion = 100;
    float sigmaR(8), aij(0.3), epsilon(0.3);
    float *gradMap_(NULL), *confMap_(NULL), *weightMap_(NULL), *customMap_(NULL);
    SpeedUpLevel speedUpLevel_ = MED_SPEEDUP /*NO_SPEEDUP*/ /*HIGH_SPEEDUP*/;
    float speedUpThreshold_(0.1);
    float fDepthThresh(0.015);

    if (gtype==COLOR){
        cbgImage_->SetImage(imgInput.data, nWidth, nHeight, true);
    }else{
        cbgImage_->SetImage(imgInput.data, nWidth, nHeight, false);
    }

    //if gradient and confidence maps are not defined, 
    //and synergistic segmentation is requested, then compute them;
    //also compute them if the parameters have changed
    bool bUseWeightMap = true;
    if (bUseWeightMap){
        confMap_ = new float[nWidth*nHeight];
        gradMap_ = new float[nWidth*nHeight];
        
        //compute gradient and confidence maps
        BgEdgeDetect edgeDetector(kernelSize);
        edgeDetector.ComputeEdgeInfo(cbgImage_, confMap_, gradMap_);

        //compute weight map...
        //allocate memory for weight map
        weightMap_ = new float[nWidth*nHeight];

        //compute weight map using gradient and confidence maps
        int i;
        for (i=0; i<nWidth*nHeight; i++)
        {
            if (gradMap_[i] > 0.02)
                weightMap_[i] = aij*gradMap_[i] + (1 - aij)*confMap_[i];
            else
                weightMap_[i] = 0;
        }
    }

    //create instance of image processor class
    msImageProcessor *iProc = new msImageProcessor();
    iProc->DefineImage(cbgImage_->im_, gtype, nHeight, nWidth);
    iProc->SetWeightMap(weightMap_, epsilon);

    // prepare data which is 0-1 float
    assert(!imgDepth.empty() && imgDepth.type()==CV_32F);
    float *pDepth = new float[nWidth*nHeight];
    for (int iterH=0; iterH<nHeight; iterH++){
        for (int iterW=0; iterW<nWidth; iterW++){
            pDepth[iterH*nWidth+iterW] = imgDepth.at<float>(iterH, iterW)/*/255.0*/;
        }
    }
    //iProc->FuseWithDepth(pDepth, 0.05);
    iProc->SetDepth(pDepth, fDepthThresh);

    //perform image segmentation or filtering....
    iProc->Segment(sigmaS, sigmaR, minRegion, speedUpLevel_, true);

    // fetch results
    segmImage_->Resize(nWidth, nHeight, cbgImage_->colorIm_);
    iProc->GetResults(segmImage_->im_);

    //save result
    cv::Mat imgInnerResult;
    if (segmImage_->colorIm_){
        imgInnerResult = cv::Mat(nHeight, nWidth, CV_8UC3, segmImage_->im_);
    }else{
        imgInnerResult = cv::Mat(nHeight, nWidth, CV_8U, segmImage_->im_);
    }
    imgResult = imgInnerResult.clone();

    int* pLabelsInner;
    float* modes;
    int* modePointCounts;
    regionCount = iProc->GetRegions(&pLabelsInner, &modes, &modePointCounts);
    *labels = new int[nHeight*nWidth];
    memcpy(*labels, pLabelsInner, sizeof(int)*nHeight*nWidth);

    delete segmImage_;
    delete cbgImage_;
    if (customMap_) delete [] customMap_;
    if (confMap_)   delete [] confMap_;
    if (gradMap_)   delete [] gradMap_;
    if (weightMap_) delete [] weightMap_;
    delete iProc;
    return 1;
}

bool importData(cv::Mat& imgView, cv::Mat& imgDepthS, cv::Mat& imgDepthT,
    cv::Mat& imgCohS, cv::Mat& imgCohT, cv::Mat& imgGT,
    const char* const strFilename, const bool bRevise)
{
    int nWidth, nHeight, nChannels;

    // read local depth
    // open the file
    hid_t fileID = H5Fopen(strFilename, H5F_ACC_RDONLY, H5P_DEFAULT );
    if (fileID<0){
        cerr<<"failed to open h5 file!"<<endl;
        return false;
    }

    // get dims
    hid_t dset = H5Dopen (fileID, "CENTERVIEW", H5P_DEFAULT);
    hsize_t dims[3];
    H5LTget_dataset_info(fileID, "CENTERVIEW", dims, NULL, NULL);
    nHeight = dims[0]; nWidth = dims[1]; nChannels = dims[2];
    // create buffer for light field data
    uchar *data3d = new uchar[nWidth*nHeight*nChannels];
    herr_t bErr = H5Dread(dset, H5T_NATIVE_UCHAR, H5S_ALL, H5S_ALL, H5P_DEFAULT, data3d);
    cv::Mat imgTmp;
    if(dims[2]==3){
        imgTmp = cv::Mat(nHeight, nWidth, CV_8UC3, data3d);
        imgView = imgTmp.clone();
    }else{
        imgTmp = cv::Mat(nHeight, nWidth, CV_8U, data3d);
        imgView.create(nHeight, nWidth, CV_8UC3);
        for(int iterH=0; iterH<dims[0]; iterH++){
            for(int iterW=0; iterW<dims[1]; iterW++){
                uchar tmp = imgTmp.at<uchar>(iterH, iterW);
                imgView.at
                    <cv::Vec3b>(iterH, iterW) = cv::Vec3b(tmp, tmp, tmp);
            }
        }
    }
    H5Dclose(dset);

    float *data2d = new float[nHeight*nWidth];
    H5LTread_dataset_float( fileID, "DEPTHS", data2d);
    imgTmp = cv::Mat(nHeight, nWidth, CV_32F, data2d);
    imgDepthS = imgTmp.clone();
    H5LTread_dataset_float( fileID, "DEPTHT", data2d);
    imgTmp = cv::Mat(nHeight, nWidth, CV_32F, data2d);
    imgDepthT = imgTmp.clone();
    if(bRevise){
        H5LTread_dataset_float( fileID, "COHSR", data2d);
        imgTmp = cv::Mat(nHeight, nWidth, CV_32F, data2d);
        imgCohS = imgTmp.clone();
        H5LTread_dataset_float( fileID, "COHTR", data2d);
        imgTmp = cv::Mat(nHeight, nWidth, CV_32F, data2d);
        imgCohT = imgTmp.clone();
    }else{
        H5LTread_dataset_float( fileID, "COHS", data2d);
        imgTmp = cv::Mat(nHeight, nWidth, CV_32F, data2d);
        imgCohS = imgTmp.clone();
        H5LTread_dataset_float( fileID, "COHT", data2d );
        imgTmp = cv::Mat(nHeight, nWidth, CV_32F, data2d);
        imgCohT = imgTmp.clone();
    }
    H5LTread_dataset_float( fileID, "GT", data2d );
    imgTmp = cv::Mat(nHeight, nWidth, CV_32F, data2d);
    imgGT = imgTmp.clone();

    H5Fclose(fileID);
    delete[] data3d;
    delete[] data2d;
    return true;
}

void countErrors(const cv::Mat& imgView, const cv::Mat& imgGT,
    const cv::Mat& imgResult, cv::Mat& imgDiff, ofstream& output){
    if (imgResult.empty() || imgGT.empty()){
        return;
    }
    
    int nHeight = imgResult.rows;
    int nWidth = imgResult.cols;
    
    float fTotalDiff = 0;
    float fRMS = 0;
    float fL1Diff = 0;
    
    double dThreshold = 0.032;
    double dErrRate(-1);
    double dMinVal, dMaxVal;

if (1){
    // RMS
    for(int iterH=0; iterH<nHeight; iterH++){
        for(int iterW=0; iterW<nWidth; iterW++){
            fTotalDiff += pow(imgResult.at<float>(iterH, iterW) - imgGT.at<float>(iterH, iterW), 2);
        }
    }
    fRMS = sqrt(fTotalDiff/(nHeight*nWidth));

    // L-1 Norm
    fTotalDiff = 0;
    for(int iterH=0; iterH<nHeight; iterH++){
        for(int iterW=0; iterW<nWidth; iterW++){
            fTotalDiff += abs(imgResult.at<float>(iterH, iterW) - imgGT.at<float>(iterH, iterW));
        }
    }
    fL1Diff = fTotalDiff/(nHeight*nWidth);

    // bad match
    minMaxLoc3D(imgGT, &dMinVal, &dMaxVal);
    double dDepthRange = dMaxVal - dMinVal;
    int nCntErr = 0;
    imgDiff.create(nHeight, nWidth, CV_32F);
    for(int iterH=0; iterH<nHeight; iterH++){
        for(int iterW=0; iterW<nWidth; iterW++){
            float fDiff = abs(imgResult.at<float>(iterH, iterW) - imgGT.at<float>(iterH, iterW));
            imgDiff.at<float>(iterH, iterW) = fDiff;
            if(fDiff/dDepthRange>dThreshold){
                nCntErr++;
            }
        }
    }
    dErrRate = nCntErr/((double)nHeight*nWidth);

}else{      // exclude background

    int nCnt = 0;
    // RMS
    for(int iterH=0; iterH<nHeight; iterH++){
        for(int iterW=0; iterW<nWidth; iterW++){
            cv::Vec3b v = imgView.at<cv::Vec3b>(iterH, iterW);
            if (v[0]!=0 || v[1]!=0 || v[2]!=0){
                fTotalDiff += pow(imgResult.at<float>(iterH, iterW) - imgGT.at<float>(iterH, iterW), 2);
                nCnt++;
            }
        }
    }
    fRMS = sqrt(fTotalDiff/nCnt);

    // L-1 Norm
    fTotalDiff = 0;
    for(int iterH=0; iterH<nHeight; iterH++){
        for(int iterW=0; iterW<nWidth; iterW++){
            cv::Vec3b v = imgView.at<cv::Vec3b>(iterH, iterW);
            if (v[0]!=0 || v[1]!=0 || v[2]!=0){
                fTotalDiff += abs(imgResult.at<float>(iterH, iterW) - imgGT.at<float>(iterH, iterW));
            }
        }
    }
    fL1Diff = fTotalDiff/nCnt;

    // bad match
    minMaxLoc3D(imgGT, &dMinVal, &dMaxVal);
    double dDepthRange = dMaxVal - dMinVal;
    int nCntErr = 0;
    imgDiff.create(nHeight, nWidth, CV_32F);
    for(int iterH=0; iterH<nHeight; iterH++){
        for(int iterW=0; iterW<nWidth; iterW++){
            cv::Vec3b v = imgView.at<cv::Vec3b>(iterH, iterW);
            if (v[0]!=0 || v[1]!=0 || v[2]!=0){
                float fDiff = abs(imgResult.at<float>(iterH, iterW) - imgGT.at<float>(iterH, iterW));
                imgDiff.at<float>(iterH, iterW) = fDiff;
                if(fDiff/dDepthRange>dThreshold){
                    nCntErr++;
                }
            }
        }
    }
    dErrRate = nCntErr/((double)nCnt);
}
            
    output<<", threshold:"<<dThreshold;
    output<<", err rate:"<<dErrRate;
    output<<", RMS:"<<fRMS;
    output<<", L1Diff:"<<fL1Diff;

    return;
}

// NOT in use
void setLabelImg(const cv::Mat& imgView, const int nData,
    const int *pLabels, const int nRegionCnt, cv::Mat& imgLabels)
{
    int nHeight = imgView.rows;
    int nWidth = imgView.cols;
    imgLabels.create(nHeight, nWidth, CV_8UC3);

    long colors[1000][4] = {0};
    for (int l=0; l<nRegionCnt; l++){
        for (int i=0; i<nHeight; i++){
            for (int j=1; j<nWidth; j++){
                if (pLabels[i*nWidth+j]==l){
                    cv::Vec3b v = imgView.at<cv::Vec3b>(i,j);
                    colors[l][0] += v[0];
                    colors[l][1] += v[1];
                    colors[l][2] += v[2];
                    colors[l][3] += 1;
                }
            }
        }
    }
    for (int l = 0; l<nRegionCnt; l++){
        for (int i=0; i<nHeight; i++){
            for (int j=1; j<nWidth; j++){
                if (pLabels[i*nWidth+j]==l){
                    cv::Vec3b v;
                    v[0] = colors[l][0]/(float)colors[l][3];
                    v[1] = colors[l][1]/(float)colors[l][3];
                    v[2] = colors[l][2]/(float)colors[l][3];            
                    imgLabels.at<cv::Vec3b>(i,j) = v;
                }
            }
        }
    }
    char strBaseDir[100];
    sprintf(strBaseDir, "C:\\work\\data4\\data%02d", nData);
    char strTmp[100];
    sprintf(strTmp, "%s\\imgLabels.jpg", strBaseDir);
    saveImage(strTmp, imgLabels, false);
}

// right now it does NOT do anything. 
// need to find out a better way.
void reviseCoh(cv::Mat& imgCohS, cv::Mat& imgCohT, float fThreshold)
{
    int nWidth = imgCohS.cols; 
    int nHeight = imgCohS.rows;

    //revise coherence
    for (int iterR=0; iterR<nHeight; iterR++){
        for(int iterC=0; iterC<nWidth; iterC++){
            float fCohT = imgCohT.at<float>(iterR, iterC);
            float fCohS = imgCohS.at<float>(iterR, iterC);
            imgCohT.at<float>(iterR, iterC) = fCohT<fThreshold ? 0 : (fCohT - fThreshold)/(1-fThreshold);
            imgCohS.at<float>(iterR, iterC) = fCohS<fThreshold ? 0 : (fCohS - fThreshold)/(1-fThreshold);
        }
    }
}

void truncateDepth(cv::Mat& imgDepth, float fMinVal, float fMaxVal)
{
    int nWidth = imgDepth.cols;
    int nHeight = imgDepth.rows;
    int nChannels = imgDepth.channels();
    for(int iterH=0; iterH<nHeight; iterH++){
        for(int iterW=0; iterW<nWidth; iterW++){
            if(imgDepth.at<float>(iterH, iterW) < fMinVal){
                imgDepth.at<float>(iterH, iterW) = fMinVal;
            } else if(imgDepth.at<float>(iterH, iterW) > fMaxVal){
                imgDepth.at<float>(iterH, iterW) = fMaxVal;
            }
        }
    }
}

void mergeDepth(const cv::Mat& imgDepthS, const cv::Mat& imgDepthT, cv::Mat& imgCohS, const cv::Mat& imgCohT
    , cv::Mat& imgDepth, cv::Mat& imgCoh)
{
    int nWidth = imgDepthS.cols;
    int nHeight = imgDepthS.rows;
    int nChannels = imgDepthS.channels();
    for (int iterR=0; iterR<nHeight; iterR++){
        for(int iterC=0; iterC<nWidth; iterC++){
            int nOffset = iterR*nWidth+iterC;
            float fCohT = imgCohT.at<float>(iterR, iterC);
            float fCohS = imgCohS.at<float>(iterR, iterC);
            float fDepthT = imgDepthT.at<float>(iterR, iterC);
            float fDepthS = imgDepthS.at<float>(iterR, iterC);
            if (fCohT<fCohS){
                imgCoh.at<float>(iterR, iterC) = fCohS;
                imgDepth.at<float>(iterR, iterC) = fDepthS;
            }else if(fCohT>fCohS){
                imgCoh.at<float>(iterR, iterC) = fCohT;
                imgDepth.at<float>(iterR, iterC) = fDepthT;
            }else{
                // intuitively averaging makes more sense.
                // however, the results is worse than taking the last case as else ...
                imgCoh.at<float>(iterR, iterC) = (fCohT+fCohS)/2;
                imgDepth.at<float>(iterR, iterC) = (fDepthT+fDepthS)/2;
            }
        }
    }
}