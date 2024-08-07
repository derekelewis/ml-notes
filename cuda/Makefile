NVCC := nvcc
NVCC_FLAGS := -g -G
BUILD_DIR := build
PYTHON_VER := 3.12
MLENV_DIR := ~/mlenv/lib/python$(PYTHON_VER)/site-packages/torch
TORCH_INCLUDE := $(MLENV_DIR)/include
TORCH_LIB := $(MLENV_DIR)/lib

# Find all .cu files in the current directory
SRCS := $(filter-out torchMatrixMultiply.cu, $(wildcard *.cu))
# Generate corresponding object file names in the build directory
OBJS := $(patsubst %.cu,$(BUILD_DIR)/%.o,$(SRCS))
# Generate executable names from the object files
EXECS := $(patsubst $(BUILD_DIR)/%.o,$(BUILD_DIR)/%,$(OBJS))

# Ensure the build directory exists
$(shell mkdir -p $(BUILD_DIR))

.PHONY: all clean

all: $(EXECS) $(BUILD_DIR)/torchMatrixMultiply.so

# Rule to build executables from object files
$(BUILD_DIR)/%: $(BUILD_DIR)/%.o
	$(NVCC) $(NVCC_FLAGS) $< -o $@

# Rule to compile .cu files into object files
$(BUILD_DIR)/%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) --compiler-options '-fPIC' -c $< -o $@

.SECONDARY: $(OBJS)

$(BUILD_DIR)/torchMatrixMultiply.so: torchMatrixMultiply.cu
	$(NVCC) $(NVCC_FLAGS) -shared --compiler-options '-fPIC' -L ~/mlenv/lib/python3.12/site-packages/torch/lib \
	-lc10 -ltorch_cpu -ltorch -ltorch_python -lc10_cuda -ltorch_cuda \
	-isystem $(TORCH_INCLUDE)/torch/csrc/api/include \
	-isystem $(TORCH_INCLUDE) \
	torchMatrixMultiply.cu $(BUILD_DIR)/matrixMultiply.o -o $(BUILD_DIR)/torchMatrixMultiply.so \
	-DTORCH_API_INCLUDE_EXTENSION_H -DPYBIND11_COMPILER_TYPE=\"_gcc\" -DPYBIND11_STDLIB=\"_libstdcpp\" \
	-DPYBIND11_BUILD_ABI=\"_cxxabi1011\" -isystem /usr/include/python$(PYTHON_VER) -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++17

clean:
	rm -rf $(BUILD_DIR)