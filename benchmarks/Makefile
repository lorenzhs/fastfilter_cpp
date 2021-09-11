# Uncomment one of the following to switch between debug and opt mode
#OPT = -O3 -DNDEBUG
OPT=-O0 -DNDEBUG
#OPT = -g -ggdb

CXXFLAGS += -fno-strict-aliasing -Wall -std=c++17 -I. -I../src/ \
    -I../src/bumpribbon/tlx -I../src/bumpribbon/ips2ra/include -I../src/bumpribbon/DySECT \
    -I../src/lpqfilter \
    $(OPT)

UNAME_P := $(shell uname -p)
ifneq ($(AVX512),)
	CXXFLAGS += -march=skylake-avx512
else
ifeq ($(UNAME_P),aarch64)
        CXXFLAGS +=
else
ifeq ($(UNAME_P),unknown)
        CXXFLAGS += -march=native
else
	CXXFLAGS += -march=native
endif
endif
endif
ifneq ($(STATIC),)
	CXXFLAGS += -static
endif

LDFLAGS = sorter.o ../src/bumpribbon/tlx/build/tlx/libtlx.a -lpthread

HEADERS = $(wildcard ../src/*.h \
    ../src/bloom/*.h ../src/cuckoo/*.h ../src/gcs/*.h \
    ../src/gqf/*.h ../src/morton/*.h ../src/xorfilter/*.h \
    ) *.h *.hpp

.PHONY: all

BINS = bulk-insert-and-query.exe par-bench.exe
ID := $(shell hostname | cut -d 'c' -f 2)

all: $(BINS)

clean:
	/bin/rm -f $(BINS)

sorter.o: ../src/bumpribbon/sorter.cpp ../src/bumpribbon/sorter.hpp ../src/bumpribbon/minimal_hasher.hpp Makefile tlx
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.exe: %.cc sorter.o ${HEADERS}  Makefile tlx
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

%-ID: %.cc sorter.o ${HEADERS} Makefile tlx
	$(CXX) $(CXXFLAGS) $< -o $*-$(ID).exe $(LDFLAGS)

tlx:
	mkdir -p ../src/bumpribbon/tlx/build; \
	cd ../src/bumpribbon/tlx/build; \
	cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS=-g0 -GNinja ..; \
	ninja
