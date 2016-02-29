
#include <string>

#include "fileOp.h"

using namespace std;

void breakupFileName(const string wholePath, string& dir, string& filename)
{
    size_t nPos = wholePath.rfind('\\');
    if (nPos < wholePath.length() && nPos>0){
        dir = wholePath;
        filename = wholePath;
        dir.erase(nPos);
        filename.erase(0, nPos+1);
    }
    return;
}