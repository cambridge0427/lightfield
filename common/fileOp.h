#ifndef _FILE_OPERATION
#define _FILE_OPERATION

#include <string>

// break a whole path to dir and filename
void breakupFileName(const std::string wholePath, std::string & dir, std::string & filename);

#endif
