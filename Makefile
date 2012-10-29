# CUDA install path
CUDA_INSTALL_PATH ?= /usr/local/cuda
# Debug
DEBUG ?= 0

# Target executable
TARGET := idct


# C files
CFILES := idct.c idct_cpu.c idct_fcpu.c
# CUDA files
CUFILES := idct_gpu.cu

# Compilers
CC := gcc -fPIC
LINK := g++ -fPIC
NVCC := $(CUDA_INSTALL_PATH)/bin/nvcc -Xcompiler -fPIC

# Debug
ifeq ($(DEBUG),1)
    COMMONFLAGS += -g -D_DEBUG -O0
    NVCCFLAGS += -G
else
    COMMONFLAGS += -O2
endif

# Common flags
COMMONFLAGS += -I. -I$(CUDA_INSTALL_PATH)/include
# C flags
CFLAGS += $(COMMONFLAGS) -std=c99
# CUDA flags
NVCCFLAGS += $(COMMONFLAGS) \
	--ptxas-options="-v" \
	-gencode arch=compute_30,code=sm_30 \
	-gencode arch=compute_20,code=sm_20 \
	-gencode arch=compute_11,code=sm_11 \
	-gencode arch=compute_10,code=sm_10

# Do 32bit vs. 64bit setup
LBITS := $(shell getconf LONG_BIT)
ifeq ($(LBITS),64)
    # 64bit
    LDFLAGS += -L$(CUDA_INSTALL_PATH)/lib64
else
    # 32bit
    LDFLAGS += -L$(CUDA_INSTALL_PATH)/lib
endif
LDFLAGS += -lcudart

build: $(TARGET)

# Clean
clean:
	rm -f *.o $(TARGET) 
	rm -f *.i *.ii 
	rm -f *.cudafe1.c *.cudafe1.cpp *.cudafe1.gpu *.cudafe1.stub.c
	rm -f *.cudafe2.c *.cudafe2.gpu *.cudafe2.stub.c
	rm -f *.fatbin *.fatbin.c *.ptx *.hash *.cubin *.cu.cpp
	
# Lists of object files
COBJS=$(CFILES:.c=.c.o)
CUOBJS=$(CUFILES:.cu=.cu.o)

# Build
$(TARGET): $(COBJS) $(CUOBJS)
	$(LINK) $(COBJS) $(CUOBJS) $(LDFLAGS) -o $(TARGET);    

# Set suffix for CUDA files
.SUFFIXES: .cu

# Pattern rule for compiling C files
%.c.o: %.c 
	$(CC) $(CFLAGS) -c $< -o $@

# Pattern rule for compiling CUDA files
%.cu.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@;
