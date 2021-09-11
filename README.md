# fastfilter_cpp

Fast Filter: Fast approximate membership filter implementations (C++)

This is a research library currently. It is not meant for production use.

Originally, this library was written for Xor filters. It has since been expanded to cover many different data structures. This fork extends it to consider various ribbon-based data structures, most notably [BuRR](https://github.com/lorenzhs/BuRR).



## Prerequisites

- A  C++17 compiler such as GNU G++ or LLVM Clang++
- Make, CMake, and Ninja

Expectations:

- Though it should be possible to run this benchmark on any operating system, we expect Linux and use its performance counters to measure performance.
- We expect an x64 processor with AVX2 support though most filters work on any processor, if you compile on a machine that does not support AVX2 instructions, the corresponding filters that depend on AVX2 will be disabled.

## Usage

Make sure to select the right GNU GCC compiler (e.g., via `export export CXX=g++-8`).
You may want to disable hyperthreading and adjust page sizes. Run the benchmark
on a quiet machine.


```
git clone https://github.com/lorenzhs/fastfilter_cpp.git
cd fastfilter_cpp
git submodule update --init --recursive
cd benchmarks
make
./ribbon-benchmark.sh
```

Your results will depend on the hardware, on the compiler and how the system is configured. A sample output is as follows:

```
$ ./bulk-insert-and-query.exe 1000000
Using seed 894891901
Using add_count = 1000000 = 1 million input items and actual_sample_size = 1000000 = 1 million queries
                                              find    find    find    1Xadd+                        optimal   wasted million
                                       add      0%     50%    100%    3Xfind       ε%  bits/item  bits/item   space%    keys
                            Xor8     90.74    5.46    5.46    5.46    107.12   0.3886      9.840      8.007   22.888   1.000
                           Xor12     88.81    8.00    7.99    7.99    112.79   0.0240     14.761     12.027   22.730   1.000
                           Xor16     81.62    5.66    5.66    5.67     98.61   0.0015     19.681     15.999   23.011   1.000
                        XorPlus8     93.16   15.10   15.08   14.95    138.28   0.3955      9.157      7.982   14.718   1.000
                       XorPlus16     93.95   15.68   15.57   15.46    140.67   0.0016     17.819     15.969   11.589   1.000
... # many more lines omitted
```

As part of the benchmark, we check the correctness of the implementation.

## Benchmarking

The shell script `benchmarks/ribbon-benchmark.sh` runs the benchmark for 1 million, 10 million, and 100 million keys.
The benchmark is run 50 times, 5 times, and once, respectively.
It stores the results in the files `ribbon-results-$(hostname)-{1,10,100}-raw.txt`.
To get a low error, it is best run on a Linux machine that is not otherwise in use.

There is also a parallel benchmark in `benchmarks/run-par.sh`. This is described [in the BuRR paper](https://arxiv.org/abs/2109.01892) in more detail.

Quotient filter benchmarks are run separately from `benchmarks/qf-bench.sh` and `benchmarks/qf-par-bench.sh` because they require differing input sizes.

## Where is your code?

The filter implementations are in `src/<type>/`. Most implementations depend on `src/hashutil.h`. Examples:

* src/bloom/bloom.h
* src/xorfilter/xorfilter.h

## Credit

This repository is based on https://github.com/FastFilter/fastfilter_cpp and Peter C. Dillinger's fork thereof, https://github.com/pdillinger/fastfilter_cpp/.

The cuckoo filter and the benchmark are derived from https://github.com/efficient/cuckoofilter by Bin Fan et al.
The SIMD blocked Bloom filter is from https://github.com/apache/impala (via the cuckoo filter).
The Morton filter is from https://github.com/AMDComputeLibraries/morton_filter.
The Counting Quotient Filter (CQF) is from https://github.com/splatlab/cqf.
Further Quotient Filter code is from https://github.com/TooBiased/lpqfilter.
Various retrieval-based filters are from https://github.com/sekti/retrieval-test.
The BuRR (ribbon filters) code lives at https://github.com/lorenzhs/BuRR/.
