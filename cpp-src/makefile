include ../makefile.in

LIB_A    = libhpytorch.a
LIB_SO   = libhpytorch.so
CFLAGS   = $(INCS) -Wall -g -std=gnu11 -O2 -fPIC -fopenmp $(DEFS)
CXXFLAGS = $(INCS) -Wall -g -std=c++14 -O2 -fPIC -fopenmp $(DEFS)
AR       = ar rcs

ifeq ($(shell $(CC) --version 2>&1 | grep -c "icc"), 1)
CFLAGS   += -xHost
endif

ifeq ($(shell $(CC) --version 2>&1 | grep -c "Free Software Foundation"), 1)
CFLAGS   += -march=native -Wno-unused-result -Wno-unused-function
endif

ifeq ($(shell $(CXX) --version 2>&1 | grep -c "icpc"), 1)
CXXFLAGS += -xHost
endif

ifeq ($(shell $(CXX) --version 2>&1 | grep -c "Free Software Foundation"), 1)
CXXFLAGS += -march=native -Wno-unused-result -Wno-unused-function
endif

C_SRCS   = $(wildcard *.c) $(wildcard kernels/*.c) $(wildcard solvers/*.c) \
           $(wildcard h2mat/*.c) $(wildcard gp/*.c)
C_OBJS   = $(C_SRCS:.c=.c.o)
CXX_SRCS = $(wildcard *.cpp) $(wildcard kernels/*.cpp) $(wildcard solvers/*.cpp) \
           $(wildcard h2mat/*.cpp) $(wildcard gp/*.cpp)
CXX_OBJS = $(CXX_SRCS:.cpp=.cpp.o)
OBJS     = $(C_OBJS) $(CXX_OBJS)

# Delete the default old-fashion double-suffix rules
.SUFFIXES:

.SECONDARY: $(OBJS)

all: install

install: $(LIB_A) $(LIB_SO)
	mkdir -p ../cpp-lib/lib
	cp -u $(LIB_A)  ../cpp-lib/lib/$(LIB_A)
	cp -u $(LIB_SO) ../cpp-lib/lib/$(LIB_SO)
	mkdir -p ../cpp-lib/include
	mkdir -p ../cpp-lib/include/kernels
	mkdir -p ../cpp-lib/include/solvers
	mkdir -p ../cpp-lib/include/h2mat
	mkdir -p ../cpp-lib/include/gp
	cp -u *.h ../cpp-lib/include/
	cp -u *.hpp ../cpp-lib/include/
	cp -u kernels/*.h ../cpp-lib/include/kernels/
	cp -u solvers/*.h ../cpp-lib/include/solvers/
	cp -u h2mat/*.h   ../cpp-lib/include/h2mat/
	cp -u gp/*.h      ../cpp-lib/include/gp/

$(LIB_A): $(OBJS)
	$(AR) $@ $^

$(LIB_SO): $(OBJS)
	$(CC) -shared -o $@ $^

%.c.o: %.c
	$(CC) $(CFLAGS) -c $^ -o $@

%.cpp.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $^ -o $@

clean:
	rm -rf $(OBJS) $(LIB_A) $(LIB_SO) ../cpp-lib
