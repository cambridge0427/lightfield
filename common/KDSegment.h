#ifndef DK_SEGMENT
#define DK_SEGMENT

#include <vector>

struct KDNode{
	int nXLeft;
	int nXRight;
	int nYUp;
	int nYDown;
	double dCohRate;
	enum{
		LOWCOH,
		HIGHCOH,
		SMALLPATCH,
		TOBESPLIT
	}eNodeType;

	int getArea(){
		return (nYDown-nYUp)*(nXRight-nXLeft);
	}
};

std::vector<KDNode> buidKDSegments(double* dMatCoh, 	int nWidth, int nHeight);

#endif