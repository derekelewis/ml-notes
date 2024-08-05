NVCC := nvcc
NVCC_FLAGS := -g -G
BUILD_DIR := build

# Find all .cu files in the current directory
SRCS := $(wildcard *.cu)
# Generate corresponding object file names in the build directory
OBJS := $(patsubst %.cu,$(BUILD_DIR)/%.o,$(SRCS))
# Generate executable names from the object files
EXECS := $(patsubst $(BUILD_DIR)/%.o,$(BUILD_DIR)/%,$(OBJS))

# Ensure the build directory exists
$(shell mkdir -p $(BUILD_DIR))

.PHONY: all clean

all: $(EXECS)

# Rule to build executables from object files
$(BUILD_DIR)/%: $(BUILD_DIR)/%.o
	$(NVCC) $(NVCC_FLAGS) $< -o $@

# Rule to compile .cu files into object files
$(BUILD_DIR)/%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

clean:
	rm -rf $(BUILD_DIR)