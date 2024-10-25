include ../makefile.in

INCS     = -I ../cpp-lib/include
CFLAGS   = $(INCS) -Wall -g -std=gnu11 -O2 -fPIC -fopenmp $(DEFS)
CXXFLAGS = $(INCS) -Wall -g -std=c++14 -O2 -fPIC -fopenmp $(DEFS)
LDFLAGS  = -g -O3 -fopenmp -L$(CPU_LINALG_LD_PATH)
LIBS     = $(CPU_LINALG_LIBS)
AR       = ar rcs

ifeq ($(shell $(CC) --version 2>&1 | grep -c "icc"), 1)
CFLAGS   += -xHost
endif

ifeq ($(shell $(CC) --version 2>&1 | grep -c "Free Software Foundation"), 1)
CFLAGS   += -march=native -Wno-unused-result -Wno-unused-function
LIBS     += -lgfortran -lm
endif

ifeq ($(shell $(CXX) --version 2>&1 | grep -c "icpc"), 1)
CXXFLAGS += -xHost
endif

ifeq ($(shell $(CXX) --version 2>&1 | grep -c "Free Software Foundation"), 1)
CXXFLAGS += -march=native -Wno-unused-result -Wno-unused-function
LIBS     += -lgfortran -lm
endif

C_SRCS   = $(wildcard *.c)
C_OBJS   = $(C_SRCS:.c=.c.o)
CXX_SRCS = $(wildcard *.cpp)
CXX_OBJS = $(CXX_SRCS:.cpp=.cpp.o)
OBJS     = $(C_OBJS) $(CXX_OBJS)

LIB_A = ../cpp-lib/lib/libhpytorch.a
EXES  = test_kernels.exe test_dkmat.exe test_tsolvers.exe test_csr_trsm.exe \
        test_octree.exe test_id_ppqr.exe test_h2mat.exe test_nys.exe \
		test_afn.exe test_bpcg.exe test_exact_gpr.exe test_precond_gpr.exe \
        test_exact_gpc.exe test_precond_gpc.exe
		

# Delete the default old-fashion double-suffix rules
.SUFFIXES:

.SECONDARY: $(OBJS)

all: $(EXES)

%.c.o: %.c
	$(CC) $(CFLAGS) -c $^ -o $@

%.cpp.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $^ -o $@

# Order of testing:

test_kernels.exe: test_kernels.cpp.o $(LIB_A)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LIBS)

test_dkmat.exe: test_dkmat.cpp.o $(LIB_A)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LIBS)

test_tsolvers.exe: test_tsolvers.cpp.o $(LIB_A)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LIBS)

test_csr_trsm.exe: test_csr_trsm.cpp.o $(LIB_A)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LIBS)

test_octree.exe: test_octree.cpp.o $(LIB_A)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LIBS)

test_id_ppqr.exe: test_id_ppqr.cpp.o $(LIB_A)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LIBS)

test_h2mat.exe: test_h2mat.cpp.o $(LIB_A)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LIBS)

test_nys.exe: test_nys.cpp.o $(LIB_A)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LIBS)

test_afn.exe: test_afn.cpp.o $(LIB_A)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LIBS)

test_bpcg.exe: test_bpcg.cpp.o $(LIB_A)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LIBS)

test_exact_gpr.exe: test_exact_gpr.cpp.o $(LIB_A)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LIBS)

test_precond_gpr.exe: test_precond_gpr.cpp.o $(LIB_A)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LIBS)

test_exact_gpc.exe: test_exact_gpc.cpp.o $(LIB_A)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LIBS)

test_precond_gpc.exe: test_precond_gpc.cpp.o $(LIB_A)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LIBS)

clean:
	rm -f $(OBJS) $(EXES) 