NVCC := nvcc
NVCC_FLAGS = -g -G

SRC_DIR := src
BUILD_DIR := build

# Ensure the build directory exists
$(shell mkdir -p $(BUILD_DIR))

all: matrixMultiply matrixVectorMultiply

matrixMultiply:
	$(NVCC) $(NVCC_FLAGS) $@.cu -o $(BUILD_DIR)/$@

matrixVectorMultiply:
	$(NVCC) $(NVCC_FLAGS) $@.cu -o $(BUILD_DIR)/$@

clean:
	rm -r $(BUILD_DIR)