SOFT_DEV_DIR=	../
SRW_SRC_DIR=	$(SOFT_DEV_DIR)/src
SRW_SRC_GEN_DIR=	$(SRW_SRC_DIR)/core
SRW_SRC_LIB_DIR=	$(SRW_SRC_DIR)/lib
SRW_SRC_GENESIS_DIR=	$(SRW_SRC_DIR)/ext/genesis/genesis_july08
SH_SRC_PARSE_DIR=	$(SRW_SRC_DIR)/ext/auxparse
SH_SRC_GEN_MATH_DIR=	$(SRW_SRC_DIR)/ext/genmath
LIB_DIR=	$(SOFT_DEV_DIR)/../ext_lib

MODE ?= 0

#CC  = gcc
#CXX = g++
CC  = cc
CXX = c++
NVCC = nvcc
NVCXX = nvc++

SRW_SRC_DEF=	-D_GNU_SOURCE -D__USE_XOPEN2K8 -DFFTW_ENABLE_FLOAT -D_GM_WITHOUT_BASE -DSRWLIB_STATIC -DNO_TIMER -DANSI_DECLARATORS -DTRILIBRARY -DLINUX# -D_DEBUG
SRW_INCLUDES= -I$(SRW_SRC_GEN_DIR) -I$(SRW_SRC_LIB_DIR) -I$(SH_SRC_PARSE_DIR) -I$(SH_SRC_GEN_MATH_DIR)
SRW_FLAGS=-fPIC -mavx2 -O3
LDFLAGS=-L$(LIB_DIR) -lm 

ifeq ($(MODE),cuda)
SRW_SRC_DEF += -D_OFFLOAD_GPU -DUSE_CUDA -D_FFTW3
SRW_INCLUDES += -I$(NVCOMPILERS)/$(NVARCH)/$(NVVERSION)/math_libs/include -I$(NVCOMPILERS)/$(NVARCH)/$(NVVERSION)/cuda/include 
SRW_FLAGS += -std=c++17
LDFLAGS += -L$(NVCOMPILERS)/$(NVARCH)/$(NVVERSION)/cuda/lib64 -lcudart_static -lcudadevrt -lcufft -lrt
NVCFLAGS = -O3 -arch=sm_80 -dlto -rdc=true
#CUDA_OBJ= gmfftgpu0.o srradmnpgpu.o srradstrgpu.o sroptelmgpu.o #srradintgpu.o
CUDA_OBJ=gmfftgpu0.o srradstrgpu.o sroptelmgpu.o sroptdrfgpu.o sroptgtrgpu.o srradmnpgpu.o
else
ifeq ($(MODE), omp)
SRW_SRC_DEF+= -D_WITH_OMP
SRW_FLAGS+= -fopenmp -Wno-write-strings
CFLAGS+= -D_WITH_OMP -fopenmp -Wno-write-strings 
LDFLAGS+= -lfftw 
else
ifeq ($(MODE), 0)
SRW_SRC_DEF+= -D_FFTW3 
LDFLAGS+= -lfftw3f -lfftw3 
else
$(error Unknown SRW compilation option)
endif
endif
endif

CFLAGS=	$(SRW_FLAGS) $(SRW_INCLUDES) $(SRW_SRC_DEF) 
#PYPATH=/nsls2/projects/bldev/bldev-env
PYPATH=/usr
#PYFLAGS=-I$(PYPATH)/include/python3.2m -L$(PYPATH)/lib/python3.2
#PYFLAGS=-I$(PYPATH)/include/python2.7 -L$(PYPATH)/lib/python2.7
#PYFLAGS=-I$(PYPATH)/include/python2.6 -L$(PYPATH)/lib/python2.6

OBJ=	utidev.o auxparse.o gmfft.o gmfit.o gminterp.o gmmeth.o gmtrans.o srclcuti.o srcradint.o srctrjdt.o sremitpr.o srgsnbm.o srgtrjdt.o srisosrc.o srmagcnt.o srmagfld.o srmatsta.o sroptapt.o sroptcnt.o sroptdrf.o sroptel2.o sroptel3.o sroptelm.o sroptfoc.o sroptgrat.o sroptgtr.o sropthck.o sroptcryst.o sroptmat.o sroptpsh.o sroptshp.o sroptsmr.o sroptwgr.o sroptzp.o sroptzps.o srpersto.o srpowden.o srprdint.o srprgind.o srpropme.o srptrjdt.o srradinc.o srradint.o srradmnp.o srradstr.o srremflp.o srsase.o srsend.o srstowig.o srsysuti.o srthckbm.o srthckbm2.o srtrjaux.o srtrjdat.o srtrjdat3d.o all_com.o check.o diagno.o esource.o field.o incoherent.o initrun.o input.o loadbeam.o loadrad.o magfield.o main.o math.o mpi.o output.o partsim.o pushp.o rpos.o scan.o source.o stepz.o string.o tdepend.o timerec.o track.o srerror.o srwlib.o 

PRG=	libsrw.a

ifeq ($(MODE),cuda)
lib:	$(OBJ) $(CMN_OBJ) $(CUDA_OBJ)
	$(NVCC) $(NVCFLAGS) -Xcompiler="$(SRW_FLAGS)" -dlink -o srwl_link.o *.o $(LDFLAGS)
	ar -cvq $(PRG) *.o
	#cp $(PRG) $(LIB_DIR)/
	rm -f *.o

%.o: $(SRW_SRC_LIB_DIR)/%.cpp
	$(CC) $(CFLAGS) -c $<
#	$(NVCC) -x=cu -dc $(NVCFLAGS) $(SRW_INCLUDES) $(SRW_SRC_DEF) -Xcompiler="$(CFLAGS)" -c $<

%.o: $(SH_SRC_GEN_MATH_DIR)/%.cpp
	$(CC) $(CFLAGS) -c $<
#	$(NVCC) -x=cu -dc $(NVCFLAGS) $(SRW_INCLUDES) $(SRW_SRC_DEF) -Xcompiler="$(CFLAGS)" -c $<

%.o: $(SRW_SRC_GEN_DIR)/%.cpp
	$(CC) $(CFLAGS) -c $<
#	$(NVCC) -x=cu -dc $(NVCFLAGS) $(SRW_INCLUDES) $(SRW_SRC_DEF) -Xcompiler="$(CFLAGS)" -c $<

%.o: $(SH_SRC_PARSE_DIR)/%.cpp
	$(CC) $(CFLAGS) -c $<
#	$(NVCC) -x=cu -dc $(NVCFLAGS) $(SRW_INCLUDES) $(SRW_SRC_DEF) -Xcompiler="$(CFLAGS)" -c $<

%.o: $(SRW_SRC_GENESIS_DIR)/%.c
	$(CC) $(CFLAGS) -c $<
#	$(NVCC) -x=c -dc $(NVCFLAGS) $(SRW_INCLUDES) $(SRW_SRC_DEF) -Xcompiler="$(CFLAGS)" -c $<

%.o: $(SRW_SRC_LIB_DIR)/%.cu
	$(NVCC) -dc $(NVCFLAGS) $(SRW_INCLUDES) $(SRW_SRC_DEF) -Xcompiler="$(CFLAGS)" -c $<
	
%.o: $(SH_SRC_GEN_MATH_DIR)/%.cu
	$(NVCC) -dc $(NVCFLAGS) $(SRW_INCLUDES) $(SRW_SRC_DEF) -Xcompiler="$(CFLAGS)" -c $<

%.o: $(SRW_SRC_GEN_DIR)/%.cu
	$(NVCC) -dc $(NVCFLAGS) $(SRW_INCLUDES) $(SRW_SRC_DEF) -Xcompiler="$(CFLAGS)" -c $<

else
lib:	$(OBJ) $(CMN_OBJ)
	ar -cvq $(PRG) *.o
	#cp $(PRG) $(LIB_DIR)/
	rm -f *.o

%.o: $(SRW_SRC_LIB_DIR)/%.cpp
	$(CXX) $(CFLAGS) -c $<

%.o: $(SH_SRC_GEN_MATH_DIR)/%.cpp
	$(CXX) $(CFLAGS) -c $<

%.o: $(SRW_SRC_GEN_DIR)/%.cpp
	$(CXX) $(CFLAGS) -c $<

%.o: $(SH_SRC_PARSE_DIR)/%.cpp
	$(CXX) $(CFLAGS) -c $<

%.o: $(SRW_SRC_GENESIS_DIR)/%.c
	$(CC) $(CFLAGS) -c $<
	
%.o: $(SRW_SRC_LIB_DIR)/%.cu
	$(NVCC) -x=c++ -Xcompiler="$(CFLAGS)" -c $<
	
%.o: $(SH_SRC_GEN_MATH_DIR)/%.cu
	$(NVCC) -x=c++ -Xcompiler="$(CFLAGS)" -c $<

%.o: $(SRW_SRC_GEN_DIR)/%.cu
	$(NVCC) -x=c++ -Xcompiler="$(CFLAGS)" -c $<
endif

pylib:
	$(CXX) -shared $(CFLAGS) $(PYFLAGS) -o srwlpy.so ../src/clients/python/srwlpy.cpp libsrw.a $(LDFLAGS)

srwlclient: $(SRW_SRC_DIR)/clients/c/srwlclient.cpp
	$(CXX) $(CFLAGS) -O3 -o srwlclient ../src/clients/c/srwlclient.cpp libsrw.a $(LDFLAGS)

clean:
	rm -f *.o *.so *.a srwlclient

all: lib pylib srwlclient
