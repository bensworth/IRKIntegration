# Compiler, compile options, and link options

MFEM_DIR  = $(HOME)/Documents/Coding/mfem
HYPRE_DIR  = $(HOME)/Documents/Coding/hypre/src/hypre
METIS_DIR = $(HOME)/Documents/Coding/metis-5.0
SUITE_SPARSE_DIR = $(HOME)/Documents/Coding/SuiteSparse
CXX      = mpic++
#FLAGS    = -g -std=c++11 -O3
FLAGS    = -g -std=c++11
INC      = -I$(MFEM_DIR) -I$(HYPRE_DIR)/include -I$(METIS_DIR)/include -Isolver_lib -I$(SUITE_SPARSE_DIR)/include
CXXFLAGS = $(FLAGS) $(INC) -DMFEM_DIR=\"$(MFEM_DIR)/\"
LDFLAGS  = -L$(MFEM_DIR) -L$(HYPRE_DIR)/lib -lz -Lsolver_lib -L$(SUITE_SPARSE_DIR)/lib
LDLIBS   = -lmfem -lHYPRE -las -lumfpack -lklu -lmetis

SOURCE_FILES  = $(wildcard *.cpp)
OBJECT_FILES  = $(SOURCE_FILES:.cpp=.o)
EXEC_FILES    = heat

.PHONY: all clean clean-build clean-exec

# Compile source files if the hpp or cpp has changed.
%.o: %.cpp %.hpp makefile
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Recompile executables if cpp files or object dependencies have changed
$(EXEC_FILES): $(OBJECT_FILES)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJECT_FILES) $(LDFLAGS) $(LDLIBS)

# Specify that 'make' compiles all examples.
all: $(OBJECT_FILES) $(EXEC_FILES)

# Remove all object files and executables
clean: clean-build clean-exec $(SUBDIRS_CLEAN)

clean-build:
	rm -f *.o *~ $(OBJECT_FILES) $(EXEC_FILES)
	rm -rf *.dSYM *.TVD.*breakpoints

clean-exec:
	@rm -f refined.mesh displaced.mesh mesh.*
