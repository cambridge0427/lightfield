#ifndef MATRIX_OP
#define MATRIX_OP

#include <stdio.h>

#include <highgui.h>

double multi(double* inVector1, double* inVector2, int length);
void multi(double para, double* inVector, double* outVector, int length);
void multi(double* inVector1, double* inVector2, double* outVector, int length);
void getInversed(double[3][3], double[3][3]);
void integralImage(double* data, int width, int height, int step, double* outData);
double getSum(double* a, int width, int height, int centerWidth, int centerHeight, int patch, 
	int step=1, int loop=0);
double getSumRect(double* a, int width, int height, int nYTop, int nYButtom, int nXLeft, int nXRight);
bool checkMatrix(double* m, int length);
void sparseMul (const cv::SparseMat_<float> & a, const cv::Mat b, cv::Mat& c);

#endif