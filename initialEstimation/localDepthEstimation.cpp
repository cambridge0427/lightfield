#include <iostream>
#include <fstream>
#include <string>

#include <hdf5.h>
#include <hdf5_hl.h>

#include <cv.h>
#include <highgui.h>

#include "../common/imageOp.h"
#include "../common/fileOp.h"
#include "lightfield.h"

using namespace std;

/* -----------------------------------------------
- Load the light field data from a h5 file
- Get the initial depth estimation and save it to an output h5 file
- At the meantime, put center view image, groud truth in the output file too.
-------------------------------------------------*/

int main(int argc, char* argv[])
{
    if (argc < 2) {
        cerr<<"Usage: localEstimation <input file name without extenstion>"<<endl;
        return -1;
    }

    string strInputFileName = argv[1];
    string strInputData = "../data/input/" + strInputFileName + ".h5";
    string sOutputDir = "../data/initial_results/" + strInputFileName;
    // Create output dir
    string sys_cmd = "mkdir " + sOutputDir;
    system(sys_cmd.c_str());

    // In old data the XY color channels are swapped. 
    // Set bSwapChan to true to get correct image view.
    bool bSwapChan = true;
    bool bInverseYT = true;

    lightfield lf(strInputData, bSwapChan);
    int H = lf.height();
    int W = lf.width();
    int T = lf.gettRes();
    int S = lf.getsRes();
    int C = lf.getnChannels();

    cv::Mat imgDepthS, imgDepthT, imgCohS, imgCohT, imgGT, imgView, imgCost;

    lf.getDepthEstimation(imgDepthS, imgCohS, imgDepthT, imgCohT, bInverseYT);
    saveImage((sOutputDir + "/imgDepthS.jpg").c_str(), imgDepthS);
    saveImage((sOutputDir + "/imgDepthT.jpg").c_str(), imgDepthT);
    saveImage((sOutputDir + "/imgCohS.jpg").c_str(), imgCohS);
    saveImage((sOutputDir + "/imgCohT.jpg").c_str(), imgCohT);

    string strOutputFilename = sOutputDir + "/localEstimation.h5";
    hid_t file_id = H5Fopen(strOutputFilename.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
    if (file_id <= 0){
        file_id = H5Fcreate(strOutputFilename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
        if (file_id <= 0){
            cout<<"failed to create the file!"<<endl;
            return -1;
        }
    }
    hsize_t dim2D[2];
    dim2D[0] = H;
    dim2D[1] = W;
    // write the result into h5 file
    hid_t set_id = H5Dopen(file_id, "DEPTHS", H5P_DEFAULT);
    if (set_id<0){
        hid_t dataspace_id = H5Screate_simple(2, dim2D, NULL);
        set_id = H5Dcreate(file_id,  "DEPTHS", H5T_NATIVE_FLOAT,  dataspace_id, 
            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT); 
    }
    H5Dwrite(set_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, imgDepthS.data);
    H5Dclose(set_id);
        
    set_id = H5Dopen(file_id, "DEPTHT", H5P_DEFAULT);
    if (set_id<0){
        hid_t dataspace_id = H5Screate_simple(2, dim2D, NULL);
        set_id = H5Dcreate(file_id,  "DEPTHT", H5T_NATIVE_FLOAT,  dataspace_id, 
            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT); 
    }
    H5Dwrite(set_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, imgDepthT.data);
    H5Dclose(set_id);

    set_id = H5Dopen(file_id, "COHS", H5P_DEFAULT);
    if (set_id<0){
        hid_t dataspace_id = H5Screate_simple(2, dim2D, NULL);
        set_id = H5Dcreate(file_id,  "COHS", H5T_NATIVE_FLOAT,  dataspace_id, 
            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT); 
    }
    H5Dwrite(set_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, imgCohS.data);
    H5Dclose(set_id);

    set_id = H5Dopen(file_id, "COHT", H5P_DEFAULT);
    if (set_id<0){
        hid_t dataspace_id = H5Screate_simple(2, dim2D, NULL);
        set_id = H5Dcreate(file_id,  "COHT", H5T_NATIVE_FLOAT,  dataspace_id, 
            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT); 
    }
    H5Dwrite(set_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, imgCohT.data);
    H5Dclose(set_id);

    // reviesed information
    lf.reviseCoh(imgCohS, imgCohT, imgCost);
    saveImage((sOutputDir + "/imgCohSRevised.jpg").c_str(), imgCohS);
    saveImage((sOutputDir + "/imgCohTRevised.jpg").c_str(), imgCohT);

    set_id = H5Dopen(file_id, "COHSR", H5P_DEFAULT);
    if (set_id<0){
        hid_t dataspace_id = H5Screate_simple(2, dim2D, NULL);
        set_id = H5Dcreate(file_id,  "COHSR", H5T_NATIVE_FLOAT,  dataspace_id, 
            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT); 
    }
    H5Dwrite(set_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, imgCohS.data);
    H5Dclose(set_id);

    set_id = H5Dopen(file_id, "COHTR", H5P_DEFAULT);
    if (set_id<0){
        hid_t dataspace_id = H5Screate_simple(2, dim2D, NULL);
        set_id = H5Dcreate(file_id,  "COHTR", H5T_NATIVE_FLOAT,  dataspace_id, 
            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT); 
    }
    H5Dwrite(set_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, imgCohT.data);
    H5Dclose(set_id);

    // write ground truth into h5 file
    imgGT = lf.getGroundTruth();
    if (!imgGT.empty()){
        saveImage((sOutputDir + "/GroundTruth.jpg").c_str(), imgGT);
        set_id = H5Dopen(file_id, "GT", H5P_DEFAULT);
        if (set_id<0){
            hid_t dataspace_id = H5Screate_simple(2, dim2D, NULL);
            set_id = H5Dcreate(file_id, "GT", H5T_NATIVE_FLOAT,  dataspace_id, 
                    H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT); 
        }
        H5Dwrite(set_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, imgGT.data);
        H5Dclose(set_id);
    }

    // write center view into h5 file
    imgView = lf.getViewImage();
    saveImage((sOutputDir + "/imgView.jpg").c_str(), imgView);
    hsize_t dim3D[3];
    dim3D[0] = H; dim3D[1] = W; dim3D[2] = C;
    hid_t dataspace_id = H5Screate_simple(3, dim3D, NULL);
    hid_t dataset_id = H5Dcreate(file_id, "CENTERVIEW", H5T_NATIVE_UCHAR, dataspace_id, 
        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (dataset_id<0){
        dataset_id = H5Dopen(file_id, "CENTERVIEW", H5P_DEFAULT);
    }
    H5Dwrite(dataset_id, H5T_NATIVE_UCHAR, H5S_ALL, H5S_ALL, H5P_DEFAULT, imgView.data);
    H5Dclose(dataset_id);
    H5Sclose(dataspace_id);
    H5Fclose(file_id);

    cout<<"Done!!!"<<endl;

    return 0;
}