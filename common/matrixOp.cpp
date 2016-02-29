#include <math.h>
#include <float.h>
#include <stdlib.h>
#include <time.h>

#include "matrixOp.h"
#include "basic.h"

// vector dot product vector
double multi(double* inVector1, double* inVector2, int length)
{
	if (inVector1 == NULL || inVector2==NULL)
	{	
		return -1.0;
	}
	double result = 0;
	for (int i=0; i<length; i++)
	{
		result += inVector1[i]*inVector2[i];
		if (!IsNumber(inVector1[i]) || !IsNumber(inVector2[i]) || !IsNumber(result)){
			int a = 1;
		}

	//	cout<<result<<endl;
	}
	return result;
}

// scalar * vector
void multi(double para, double* inVector, double* outVector, int length)
{
	if (inVector == NULL || outVector == NULL)
	{
		return;
	}
	for (int i=0; i<length; i++)
	{
		outVector[i] = para * inVector[i];
	}
	return;
}

// vector times vector: element by element
void multi(double* inVector1, double* inVector2, double* outVector, int length)
{
	if (inVector1 == NULL || outVector == NULL || inVector2 == NULL)
	{
		return;
	}
	for (int i=0; i<length; i++)
	{
		outVector[i] = inVector1[i] * inVector2[i];
	}
	return;
}

// inverse
void getInversed(double inMatrix[3][3], double outMatrix[3][3])
{
	srand ( time(NULL) );
	double detA = inMatrix[0][0]*inMatrix[1][1]*inMatrix[2][2]
				+inMatrix[0][1]*inMatrix[1][2]*inMatrix[2][0]
				+inMatrix[0][2]*inMatrix[1][0]*inMatrix[2][1]
				-inMatrix[0][2]*inMatrix[1][1]*inMatrix[2][0]
				-inMatrix[0][0]*inMatrix[1][2]*inMatrix[2][1]
				-inMatrix[0][1]*inMatrix[1][0]*inMatrix[2][2];
	while (detA==0){
		for (int i=0; i<3; i++){
			for(int j=0; j<3; j++){
				inMatrix[i][j] += (rand()%100)/(double)10000000;
			}
		}
		detA = inMatrix[0][0]*inMatrix[1][1]*inMatrix[2][2]
				+inMatrix[0][1]*inMatrix[1][2]*inMatrix[2][0]
				+inMatrix[0][2]*inMatrix[1][0]*inMatrix[2][1]
				-inMatrix[0][2]*inMatrix[1][1]*inMatrix[2][0]
				-inMatrix[0][0]*inMatrix[1][2]*inMatrix[2][1]
				-inMatrix[0][1]*inMatrix[1][0]*inMatrix[2][2];
	}
	outMatrix[0][0] = (inMatrix[1][1]*inMatrix[2][2] - inMatrix[1][2]*inMatrix[2][1])/detA;
	outMatrix[0][1] = (- inMatrix[0][1]*inMatrix[2][2] + inMatrix[0][2]*inMatrix[2][1])/detA;
	outMatrix[0][2] = (inMatrix[0][1]*inMatrix[1][2] - inMatrix[0][2]*inMatrix[1][1])/detA;
	outMatrix[1][0] = (- inMatrix[1][0]*inMatrix[2][2] + inMatrix[1][2]*inMatrix[2][0])/detA;
	outMatrix[1][1] = (inMatrix[0][0]*inMatrix[2][2] - inMatrix[0][2]*inMatrix[2][0])/detA;
	outMatrix[1][2] = (- inMatrix[0][0]*inMatrix[1][2] + inMatrix[0][2]*inMatrix[1][0])/detA;
	outMatrix[2][0] = (inMatrix[1][0]*inMatrix[2][1] - inMatrix[1][1]*inMatrix[2][0])/detA;
	outMatrix[2][1] = (- inMatrix[0][0]*inMatrix[2][1] + inMatrix[0][1]*inMatrix[2][0])/detA;
	outMatrix[2][2] = (inMatrix[0][0]*inMatrix[1][1] - inMatrix[0][1]*inMatrix[1][0])/detA;
	return;
}

//在积分图中求和
double getSum(double* a, int width, int height, int centerWidth, int centerHeight, int patch, int step, int loop)
{
	double sum = 0;
	int patch2 = patch>>1;
	int ptWidthRight = (centerWidth + patch2)>=width ? width-1 : centerWidth + patch2;
	int ptHeightDown = (centerHeight + patch2)>=height ? height-1 : centerHeight + patch2;
	int ptHeightTop = (centerHeight - patch2-1);
	int ptWidthLeft = (centerWidth - patch2-1);

	if (ptHeightTop >=0 && ptWidthLeft >=0)
	{
		sum = a[(ptHeightDown*width+ptWidthRight)*step+loop]
		- a[(ptHeightTop*width+ptWidthRight)*step+loop]
		- a[(ptHeightDown*width+ptWidthLeft)*step+loop]
		+ a[(ptHeightTop*width+ptWidthLeft)*step+loop];
	}else if (ptHeightTop<0 && ptWidthLeft>=0)
	{
		sum = a[(ptHeightDown*width+ptWidthRight)*step+loop]
		- a[(ptHeightDown*width+ptWidthLeft)*step+loop];
	}else if (ptHeightTop>=0 && ptWidthLeft<0)
	{
		sum = a[(ptHeightDown*width+ptWidthRight)*step+loop]
		- a[(ptHeightTop*width+ptWidthRight)*step+loop];
	}else
	{
		sum = a[(ptHeightDown*width+ptWidthRight)*step+loop];
	}
	return sum;
}

double getSumRect(double* a, int width, int height, int nYTop, int nYButtom, int nXLeft, int nXRight)
{
	int ptWidthRight = nXRight >= width ? width-1 : nXRight;
	int ptHeightDown = nYButtom >=height ? height-1 : nYButtom;
	int ptHeightTop = nYTop;
	int ptWidthLeft = nXLeft;
	int step = 1;
	int loop = 0; 
	double sum = 0;

	if (ptHeightTop >=0 && ptWidthLeft >=0)
	{
		sum = a[(ptHeightDown*width+ptWidthRight)*step+loop]
		- a[(ptHeightTop*width+ptWidthRight)*step+loop]
		- a[(ptHeightDown*width+ptWidthLeft)*step+loop]
		+ a[(ptHeightTop*width+ptWidthLeft)*step+loop];
	}else if (ptHeightTop<0 && ptWidthLeft>=0)
	{
		sum = a[(ptHeightDown*width+ptWidthRight)*step+loop]
		- a[(ptHeightDown*width+ptWidthLeft)*step+loop];
	}else if (ptHeightTop>=0 && ptWidthLeft<0)
	{
		sum = a[(ptHeightDown*width+ptWidthRight)*step+loop]
		- a[(ptHeightTop*width+ptWidthRight)*step+loop];
	}else
	{
		sum = a[(ptHeightDown*width+ptWidthRight)*step+loop];
	}
	return sum;
}

//求积分图
void integralImage(double* inData, int width, int height, int step, double* outData)
{
	for (int h=0; h<height; h++)
	{
		for (int w=0; w<width; w++)
		{
			if (w>=1 && h>=1)
			{
				for (int i=0; i<step; i++)
				{
					outData[(h*width+w)*step+i] = inData[(h*width+w)*step+i]
					+ outData[(h*width - width + w)*step+i]
					+ outData[(h*width + w - 1)*step+i]
					- outData[(h*width - width + w - 1)*step+i];
				}
			}else if (w == 0 && h >= 1)
			{
				for (int i=0; i<step; i++)
				{
					outData[(h*width+w)*step+i] = inData[(h*width+w)*step+i]
					+ outData[(h*width - width + w)*step+i];
				}
			}else if (w >= 1 && h == 0)
			{
				for (int i=0; i<step; i++)
				{
					outData[(h*width+w)*step+i] = inData[(h*width+w)*step+i]
					+ outData[(h*width + w - 1)*step+i];
				}
			}else
			{
				for (int i=0; i<step; i++)
				{
					outData[(h*width+w)*step+i] = inData[(h*width+w)*step+i];
				}
			}
		}
	}
}

bool checkMatrix(double* m, int nLength)
{
	for (int i=0; i<nLength; i++){
		if (!IsNumber(m[i])){
			return false;
		}
		if (!IsFiniteNumber(m[i])){
			return false;
		}
	}
	return true;
}

// a*b = c 
void sparseMul (const cv::SparseMat_<float> & a, const cv::Mat b, cv::Mat& c)
{
	c.setTo(0);
	cv::SparseMatConstIterator_<float>
		it = a.begin(), 
		it_end = a.end();
	//cv::SparseMat::iterator iterator_m;
	//CvSparseNode* node = cvInitSparseMatIterator(a,&iterator_m);
	for(; it != it_end; ++it)
	{
		//int* idx = CV_NODE_IDX(a,node);
		//float val = *(float*)CV_NODE_VAL(a,node);
		//CV_MAT_ELEM(*c,float,idx[0],0) += val*CV_MAT_ELEM(*b,float,idx[1],0);
		const cv::SparseMat::Node *n = it.node();
		float val = it.value<float>();
		int idx1 = n->idx[0];
		int idx2 = n->idx[1];
		c.at<float>(idx1, 0) += val*b.at<float>(idx1, 0);
	}
}