CFILES := dct_cpu.c
DEBUG ?= 0

# Target executable
TARGET := dct_cpu

CC := gcc
LINK := gcc
CFLAGS := -std=c99 
LDFLAGS=-lm

# Debug
ifeq ($(DEBUG),1)
    COMMONFLAGS += -g -D_DEBUG -O0
else
    COMMONFLAGS += -O2
endif

COBJS=$(CFILES:.c=.c.o)

build: $(TARGET)

# Build
$(TARGET): $(COBJS) $(CUOBJS)
	$(LINK) $(COBJS) $(CUOBJS) $(LDFLAGS) -o $(TARGET);    

# Pattern rule for compiling C files
%.c.o: %.c 
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f *.o $(TARGET) 
