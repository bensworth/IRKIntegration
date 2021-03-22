MFEM_DIR = ${DIR_mfem}
MFEM_BUILD_DIR = ${DIR_mfem}
CONFIG_MK = $(MFEM_BUILD_DIR)/config/config.mk

-include $(CONFIG_MK)

# Overall structure
BUILD_DIR=build
DEPENDS=$(BUILD_DIR)/.depends
SOURCES=$(wildcard *.cpp)
OBJECTS=$(SOURCES:%.cpp=$(BUILD_DIR)/%.o)
LIB=libas.a

CXXFLAGS=-g $(MFEM_CXXFLAGS)

# Use compiler configuration from MFEM
LFLGAS=${MFEM_LIBS} ${MFEM_EXT_LIBS}
INCFLAGS=${MFEM_INCFLAGS}

.PHONY: all clean style
all: $(LIB)

# Build the executable
$(LIB): $(OBJECTS)
	$(AR) $(ARFLAGS) $(@) $(OBJECTS)

$(BUILD_DIR)/%.o: makefile | $(BUILD_DIR)
	$(MFEM_CXX) -c $(CXXFLAGS) $(INCFLAGS) -o $@ $*.cpp

# Use the compiler to determine dependencies on header files
# Some awk magic in the next target
# Prefix all lines matching the regexp /^.*\.o/ with build/ (to match e.g. "file.o:")
# and leave other lines alone
$(DEPENDS): $(SOURCES) | $(BUILD_DIR)
	$(MFEM_CXX) -c -std=c++11 $(INCFLAGS) -isystem$(MFEM_DIR) -MM $(SOURCES) \
		| awk '/^.*\.o:/{ print "$(BUILD_DIR)/" $$0; next } 1'  >$@

# Rebuild dependencies unless doing "make clean" or "make style"
ifneq ($(MAKECMDGOALS),clean)
ifneq ($(MAKECMDGOALS),style)
-include $(DEPENDS)
endif
endif

# Ensure build directory exists
$(BUILD_DIR):
	mkdir -p $@

clean:
	rm -rf $(LIB) $(BUILD_DIR)

FORMAT_FILES = $(wildcard *.?pp)
ASTYLE = astyle --options=$(MFEM_DIR)/config/mfem.astylerc
style:
	@if ! $(ASTYLE) $(FORMAT_FILES) | grep Formatted; then\
	   echo "No source files were changed.";\
	fi
