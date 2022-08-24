# TODO : right now only works with one driver file in the makefile at a time;
# Need to figure out how to have funcitoning makefile for multiple drivers..
# Problem is executable needs to be in the list of source files, probably to
# have it in the list of object files, but cannot have multiple executables
# in the object files as they interdepend


# Compiler, compile options, and link options

# Ben's stuff
MFEM_DIR = /home/southworth/c_libs/mfem
INST_DIR = /home/southworth/c_libs/install
CXX      = mpicxx
FLAGS    = -O3 -std=c++14 -fopenmp
# FLAGS    = -g -std=c++14 -fopenmp
INC      = -I$(MFEM_DIR)/bens_build -I$(HYPRE_DIR)/include -I$(METIS_DIR)/include -I./
LDFLAGS  = -L$(MFEM_DIR)/bens_build -L$(HYPRE_DIR)/lib -L$(METIS_DIR)/libmetis
# INC      = -I$(INST_DIR)/include
# LDFLAGS  = -L$(INST_DIR)/lib
LDLIBS   = -lmfem -lHYPRE -lmetis
CXXFLAGS = $(FLAGS) $(INC) $(LDLIBS) $(LDFLAGS)

# Files to compile
SRCS  = IRK_utils.cpp IMEX_utils.cpp IRK.cpp
SRCS1 = test-imex-dg-adv-diff.cpp
SRCS2 = imex-dg-adv-diff.cpp
SRCS3 = nl-imex-dg-adv-diff.cpp
SRCS4 = reaction-test.cpp
SRCS5 = reaction-test-ODE.cpp

OBJS  = ${SRCS:.cpp=.o}
OBJS1 = ${SRCS1:.cpp=.o}
OBJS2 = ${SRCS2:.cpp=.o}
OBJS3 = ${SRCS3:.cpp=.o}
OBJS4 = ${SRCS4:.cpp=.o}
OBJS5 = ${SRCS5:.cpp=.o}
EXEC1 = test-imex-dg-adv-diff
EXEC2 = imex-dg-adv-diff
EXEC3 = nl-imex-dg-adv-diff
EXEC4 = reaction-test
EXEC5 = reaction-test-ODE

all: $(EXEC1) $(EXEC2) $(EXEC3) $(EXEC4) $(EXEC5)

.hpp.o:
	$(CXX) $(CXXFLAGS) -c $<

.cpp.o:
	$(CXX) $(CXXFLAGS) -c $<

$(EXEC1): $(OBJS) $(OBJS1) 
	$(CXX) $(LDLIBS) $(OBJS) $(OBJS1) $(CXXFLAGS) -o $(EXEC1)

$(EXEC2): $(OBJS) $(OBJS2) 
	$(CXX) $(LDLIBS) $(OBJS) $(OBJS2) $(CXXFLAGS) -o $(EXEC2)

$(EXEC3): $(OBJS) $(OBJS3) 
	$(CXX) $(LDLIBS) $(OBJS) $(OBJS3) $(CXXFLAGS) -o $(EXEC3)

$(EXEC4): $(OBJS) $(OBJS4) 
	$(CXX) $(LDLIBS) $(OBJS) $(OBJS4) $(CXXFLAGS) -o $(EXEC4)

$(EXEC5): $(OBJS) $(OBJS5) 
	$(CXX) $(LDLIBS) $(OBJS) $(OBJS5) $(CXXFLAGS) -o $(EXEC5)

.PHONY: clean 

# Remove all object files and executables
clean:
	rm -f *.o $(EXEC_FILES)
	rm -rf *.dSYM *.TVD.*breakpoints

