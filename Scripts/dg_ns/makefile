# Compiler, compile options, and link options

-include Make.user

HDF_FLAGS = -lhdf5 -lhdf5_cpp -lhdf5_hl
3DG_FLAGS = $(HDF_FLAGS) -lblas -llapack -ldgmpi -ldgapps -lumfpack -lklu
CXX      = mpic++
#FLAGS    = -g -std=c++11 -O3
FLAGS    = -g -std=c++11
INC      = -I$(MFEM_DIR) -I$(HYPRE_DIR)/include -I$(METIS_DIR)/include  -I$(3DG_DIR)/include -I$(SUITESPARSE_DIR)/include -I$(HDF_DIR)/include
CXXFLAGS = $(FLAGS) $(INC) -DISMPI -DHAVE_HDF5 -DMFEM_DIR=\"$(MFEM_DIR)/\"
LDFLAGS  = -Wl,-rpath,$(HDF_DIR)/lib -L$(MFEM_DIR) -L$(HYPRE_DIR)/lib -L$(METIS_DIR)/lib -lz -L$(3DG_DIR)/lib/$(3DG_ARCH) -L$(SUITESPARSE_DIR)/lib -L$(HDF_DIR)/lib
LDLIBS   = -lmfem -lHYPRE -lmetis $(3DG_FLAGS)

SOURCE_FILES  = $(wildcard *.cpp)
OBJECT_FILES  = $(SOURCE_FILES:.cpp=.o)
EXEC_FILES    = dg_ns

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
