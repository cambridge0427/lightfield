OPENCV ?= /usr/local/Cellar/opencv/2.4.12_2
EIGEN ?= /usr/local/Cellar/eigen/3.2.8
HDF ?= /usr/local/Cellar/hdf5/1.8.16_1

CXX = clang++
CPPFLAGS = -O2
LDLIBS = -L$(OPENCV)/lib -lopencv_core -lopencv_highgui -lopencv_imgproc -L$(HDF)/lib -lhdf5_hl -lhdf5
INDIRS = -I$(OPENCV)/include/opencv -I$(OPENCV)/include/ -I$(EIGEN)/include/eigen3/ -I$(HDF)/include
SRCS = $(shell find ../common -name "*.cpp") $(shell find ../MSSegment -name "*.cpp" -not -name "segment.cpp") $(shell find ../MRF -name "*.cpp" -not -name "example.cpp") $(shell find . -name "*.cpp")
OBJS = $(SRCS:%.cpp=%.o)
EXEC = optimize

all: $(EXEC)

$(EXEC):$(OBJS)
	$(CXX) $(OBJS) $(LDLIBS) -o $@

%.o:%.cpp
	$(CXX) $(CPPFLAGS) -c $(INDIRS) $^ -o $@

clean:
	rm -rf *.o $(EXEC)