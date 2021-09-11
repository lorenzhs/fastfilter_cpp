#!/bin/zsh

export ID=${1:-$(hostname | cut -d 'c' -f 2)}

ulimit -c 0

# create algolist_par as follows (LMSS doesn't work as it uses static members
# and Quotient Filters are measured separately):
# ./bulk-insert-and-query-${ID}.exe | grep -v -E '(LMSS|QuotientFilter)' | cut -d ':' -f 1 | tail -n +5 | head -n -2 | perl -pe 'chomp if eof' | tr -d ' ' | tr '\n' ',' > algolist_par

./par-bench-${ID}.exe 64 1280 $((10**7)) $(cat algolist_par) 2>&1 | tee log_par_${ID}_1e7_t64_k1280
# ./par-bench.exe 64 128 $((10**8)) all | tee log_par_1e7_t64_k128
