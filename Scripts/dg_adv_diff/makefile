-include Make.user

ifndef MFEM_DIR
$(error MFEM_DIR is not set. Please create a Make.user file to set MFEM_DIR)
endif

MFEM_BUILD_DIR = $(MFEM_DIR)
CONFIG_MK = $(MFEM_BUILD_DIR)/config/config.mk

-include $(CONFIG_MK)

# Overall structure
BUILD_DIR=build
SOURCES=$(wildcard *.cpp)
OBJECTS=$(SOURCES:%.cpp=$(BUILD_DIR)/%.o)
APP=dg_adv_diff

# Use compiler configuration from MFEM
LFLGAS=$(MFEM_LIBS)
INCFLAGS=$(MFEM_INCFLAGS)

.PHONY: all clean style

all: $(APP)

# Build the executable
$(APP): $(OBJECTS) $(MFEM_LIB_FILE)
	$(MFEM_CXX) $(MFEM_LINK_FLAGS) $(OBJECTS) $(LFLGAS) -o $@

$(BUILD_DIR)/%.o: %.cpp makefile | $(BUILD_DIR)
	$(MFEM_CXX) -c $(MFEM_CXXFLAGS) $(INCFLAGS) -o $@ $*.cpp

# Ensure build directory exists
$(BUILD_DIR):
	mkdir -p $@

clean:
	rm -rf $(EXE) $(BUILD_DIR)

FORMAT_FILES = $(wildcard *.?pp)
ASTYLE = astyle --options=$(MFEM_DIR)/config/mfem.astylerc
style:
	@if ! $(ASTYLE) $(FORMAT_FILES) | grep Formatted; then\
	   echo "No source files were changed.";\
	fi
