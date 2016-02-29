OPENCV ?= /usr/local/Cellar/opencv/2.4.12_2
HDF ?= /usr/local/Cellar/hdf5/1.8.16_1
VIGRA ?= /usr/local/Cellar/vigra/1.10.0_1

CXX = clang++
CPPFLAGS = -O2
LDLIBS = -L$(OPENCV)/lib -lopencv_core -lopencv_highgui -lopencv_imgproc -L$(HDF)/lib -lhdf5_hl -lhdf5 -L$(VIGRA)/lib
INDIRS = -I$(HDF)/include -I$(VIGRA)/include  -I$(OPENCV)/include/opencv -I$(OPENCV)/include/
SRCS = $(shell find ../common -name "*.cpp") $(shell find . -name "*.cpp")
OBJS = $(SRCS:%.cpp=%.o)
EXEC = initial_estimation

all: $(EXEC)

$(EXEC):$(OBJS)
	$(CXX) $(OBJS) $(LDLIBS) -o $@

%.o:%.cpp
	$(CXX) $(CPPFLAGS) -c $(INDIRS) $^ -o $@

clean:
	rm -rf *.o $(EXEC)
