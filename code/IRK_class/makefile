# Compiler, compile options, and link options

# Ben's stuff
#MFEM_DIR  = /g/g19/bs/cab/mfem
#HYPRE_DIR = $(MFEM_DIR)/../hypre/src/hypre
#METIS_DIR = $(MFEM_DIR)/../metis/build/Linux-x86_64
#CXX      = mpiicpc
#FLAGS    = -g -std=c++11
#INC      = -I$(MFEM_DIR)/bens_build -I$(HYPRE_DIR)/include -I$(METIS_DIR)/include -I./src
#CXXFLAGS = $(FLAGS) $(INC)
#LDFLAGS  = -L$(MFEM_DIR)/bens_build -L$(HYPRE_DIR)/lib -L$(METIS_DIR)/libmetis
#LDLIBS   = -lmfem -lHYPRE -lmetis

# Oliver's stuff
MFEM_DIR  = $(HOME)/Software/mfem-v4.0-AIR-branch
#MFEM_DIR  = $(HOME)/Software/mfem
HYPRE_DIR = $(HOME)/Software/hypre-v2.18.2/src/hypre
METIS_DIR = $(HOME)/Software/metis-5.1.0/build/olivers_build
CXX      = mpic++
FLAGS    = -g -std=c++11
INC      = -I$(MFEM_DIR) -I$(HYPRE_DIR)/include -I$(METIS_DIR)/include -I./src
CXXFLAGS = $(FLAGS) $(INC)
LDFLAGS  = -L$(MFEM_DIR) -L$(HYPRE_DIR)/lib -L$(METIS_DIR)/lib
LDLIBS   = -lmfem -lHYPRE -lmetis



# Files to compile
#	TODO : Maybe should put class files in subdirectory so can use wildcard.
# SOURCE_FILES  = $(wildcard *.cpp)
SOURCE_FILES  = IRK.cpp IRKOperator.cpp FDadvection.cpp $(wildcard ./src/*.cpp)
OBJECT_FILES  = $(SOURCE_FILES:.cpp=.o)
EXEC_FILES    = driver

.PHONY: all clean clean-build clean-exec

# Compile source files if the hpp or cpp has changed.
%.o: %.cpp %.hpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Recompile executables if cpp files or object dependencies have changed
%: %.cpp $(OBJECT_FILES)
	$(CXX) $(CXXFLAGS) $< -o $@ $(OBJECT_FILES) $(LDFLAGS) $(LDLIBS)

# Specify that 'make' compiles all examples.
all: $(OBJECT_FILES) $(EXEC_FILES)

# Remove all object files and executables
clean: clean-build clean-exec $(SUBDIRS_CLEAN)

clean-build:
	rm -f *.o *~ $(OBJECT_FILES) $(EXEC_FILES)
	rm -rf *.dSYM *.TVD.*breakpoints

clean-exec:
	@rm -f refined.mesh displaced.mesh mesh.*

