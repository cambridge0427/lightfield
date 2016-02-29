
CXX = g++
CPPFLAGS = -Wall
LDLIBS = -L/cs/vml1/vml/OpenCV-2.0.0/lib -lcv -lhighgui -L/cs/vml2/jla291/hdf5_64bit/lib -lhdf5_hl -lhdf5 -lsz -lz
INDIRS = -I/cs/vml1/vml/OpenCV-2.0.0/include/opencv -I/cs/vml2/jla291/hdf5_64bit/include
SRCS = lightfield.cpp imageOp.cpp fileOp.cpp generateLF.cpp
OBJS = $(SRCS:%.cpp=%.o)
EXEC = generateLF

all: $(EXEC)

$(EXEC):$(OBJS)
	$(CXX) $(OBJS) $(LDLIBS) -o $@

%.o:%.cpp
	$(CXX) $(CPPFLAGS) -c $(INDIRS) $^ -o $@

clean:
	rm -rf *o $(EXEC)
