//
//
// This file is to convert the Stanford lightford data into h5 format,
// which is consistent with HCI data, so that we can run experiment on them together.
// --- Jianiqao
//

#include <dirent.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <string>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <fstream>
#include <vector>

#include <cv.h>
#include <highgui.h>

#include "lightfield.h"

using namespace std;

struct filenameStruct{
    int s;
    int t;
    string filename;
};

string sInputPath[9];
string sOutputPath[9];

int main(int argc, char* argv[])
{
    sInputPath[0] =  "/cs/vml2/jla291/LightFieldData/Stanford/bulldozerSmall";
    sInputPath[1] =  "/cs/vml2/jla291/LightFieldData/Stanford/amethyst";
    sInputPath[2] =  "/cs/vml2/jla291/LightFieldData/Stanford/bracelet";
    sInputPath[3] =  "/cs/vml2/jla291/LightFieldData/Stanford/bulldozer";
    sInputPath[4] =  "/cs/vml2/jla291/LightFieldData/Stanford/bunny";
    sInputPath[5] =  "/cs/vml2/jla291/LightFieldData/Stanford/chess";
    sInputPath[6] =  "/cs/vml2/jla291/LightFieldData/Stanford/flower";
    sInputPath[7] =  "/cs/vml2/jla291/LightFieldData/Stanford/treasureChest";
    sInputPath[8] =  "/cs/vml2/jla291/LightFieldData/Stanford/truck";

    sOutputPath[0] =  "/cs/vml2/jla291/LightFieldData/Stanford/bulldozerSmall/bulldozerSmall.h5";
    sOutputPath[1] =  "/cs/vml2/jla291/LightFieldData/Stanford/amethyst/amethyst.h5";
    sOutputPath[2] =  "/cs/vml2/jla291/LightFieldData/Stanford/bracelet/bracelet.h5";
    sOutputPath[3] =  "/cs/vml2/jla291/LightFieldData/Stanford/bulldozer/bulldozer.h5";
    sOutputPath[4] =  "/cs/vml2/jla291/LightFieldData/Stanford/bunny/bunny.h5";
    sOutputPath[5] =  "/cs/vml2/jla291/LightFieldData/Stanford/chess/chess.h5";
    sOutputPath[6] =  "/cs/vml2/jla291/LightFieldData/Stanford/flower/flower.h5";
    sOutputPath[7] =  "/cs/vml2/jla291/LightFieldData/Stanford/treasureChest/treasureChest.h5";
    sOutputPath[8] =  "/cs/vml2/jla291/LightFieldData/Stanford/truck/truck.h5";

    string sInputDir, fileName;
        
    if (argc == 1){
        sInputDir = sInputPath[0];
        fileName = sOutputPath[0];
    }else{
        sInputDir = sInputPath[atoi(argv[1])];
        fileName = sOutputPath[atoi(argv[1])];
    }


//for (int i=4; i<9; i++){

//  sInputDir = sInputPath[i];
//  fileName = sOutputPath[i];

    DIR *pDir;
    struct dirent *pDirHead;
    struct stat filestat;

    
    pDir = opendir( sInputDir.c_str() );
    if (pDir == NULL){
        cout << "no such directory" << endl;
            return -1;
        }

    //lightfield lf;
    int nSRes(0), nTRes(0), nYRes(0), nXRes(0), nChannels(0);
    vector<filenameStruct> files;

    // load data into lf
    bool flag = true;
    while ((pDirHead=readdir(pDir)) != NULL) {
        string sFileName = string(pDirHead->d_name);
        
        // get s and t value
        size_t found = sFileName.find("_");
        if (found>sFileName.length()) continue;
        int s = atoi(sFileName.substr(found+1, 2).c_str());
        if (s>nSRes) nSRes = s;

        found = sFileName.find("_", found+1);
        if (found>sFileName.length()) continue;
        int t = atoi(sFileName.substr(found+1, 2).c_str());
        if (t>nTRes) nTRes = t;

        string sFilePath = sInputDir+"/"+sFileName;
        filenameStruct fnStruct;
        fnStruct.filename = sFilePath;
        fnStruct.s = s; 
        fnStruct.t = t;
        files.push_back(fnStruct);

        if (flag == true){
            // Open a image and get its resolution
            cv::Mat img = cv::imread(sFilePath, -1);
            nYRes = img.rows;
            nXRes = img.cols;
            //cout<<sFilePath<<" "<<nYRes<<"  "<<nXRes<<endl;
            //cout<<img.step<<endl;
            //cout<<img.depth();
            nChannels = img.channels();
            flag = false;
        }
    }
    closedir(pDir);
    nTRes++; nSRes++;
    cout<<nSRes<<"  "<<nTRes<<"  "<<nYRes<<"  "<<nXRes<<"  "<<nChannels<<endl;

    //nSRes = 17; nTRes = 17; nYRes = 461; nXRes = 615; nChannels = 3;
    float* data = new float[nSRes*nTRes*nYRes*nXRes*nChannels];
    cout<<"allocate success"<<endl;
    for (vector<filenameStruct>::iterator iterFiles=files.begin(); 
        iterFiles<files.end();
        iterFiles++){
        cv::Mat img = cv::imread(iterFiles->filename, -1);
        int t = iterFiles->s;
        int s = iterFiles->t;

        float* dataDest = data + t*nSRes*nYRes*nXRes*nChannels + s*nYRes*nXRes*nChannels;
        //nOffSet = iterT*S*H*W*C + ns*H*W*C + nx*C;
        
        // copy data
        for (int iterR = 0; iterR<img.rows; iterR++){
            for (int iterC=0; iterC<img.cols; iterC++){
                int nOffset = iterR*nXRes*nChannels+iterC*nChannels;
                cv::Vec3b v = img.at<cv::Vec3b>(iterR, iterC);
                *(dataDest+nOffset) = (float)v[0];
                *(dataDest+nOffset+1) = (float)v[1];
                *(dataDest+nOffset+2) = (float)v[2];
            }
        }
        cout<<"Finish copying from "<<iterFiles->filename<<endl;
    }

    
    hid_t file_id = H5Fcreate(fileName.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file_id <=0){
        cout<<"failed to create the file!"<<endl;
        return -1;
    }
        hsize_t dims[5];
    hsize_t ndims = 5;
    dims[0] = nSRes; dims[1]=nTRes; dims[2]=nYRes; dims[3]=nXRes; dims[4]=nChannels;
    hid_t dataspace_id = H5Screate_simple(ndims, dims, NULL);
    hid_t dataset_id = H5Dcreate(file_id, "LF",H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);  
    H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);

    H5Dclose(dataset_id);
    H5Sclose(dataspace_id);
        H5Fclose(file_id); 
    delete[] data;

//}

    return 0;
}
