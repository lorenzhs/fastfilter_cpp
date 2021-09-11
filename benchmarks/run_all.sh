#!/bin/bash

ID=${1:-$(hostname | cut -d 'c' -f 2)}

DIR=./logs_run7
mkdir -p ${DIR} ${DIR}/results
# symlink evaluation scripts
for f in combine-results.sh results_post.sh ribbon-benchmark-post.sh ribbon-par-post.sh summarize-results.sh; do ln -s ../$f ${DIR}/$f; done

./ribbon-benchmark.sh ${ID}
mv -n results_*.txt ${DIR}/results
mv -n ribbon-results-${ID}-*-raw.txt ${DIR}

# generate algorithm list for parallel benchmark

./bulk-insert-and-query-${ID}.exe | grep -v -E '(LMSS|QuotientFilter)' | cut -d ':' -f 1 | tail -n +5 | head -n -2 | perl -pe 'chomp if eof' | tr -d ' ' | tr '\n' ',' > algolist_par

./run-par.sh ${ID}
mv -n results_*.txt ${DIR}/results
mv -i log_par_${ID}_* ${DIR}/ribbon-par-${ID}-raw.txt

# Now run quotient filter benchmark

./qf-bench.sh ${ID}
mv -i ribbon-results-qf-${ID}-*-raw.txt ${DIR}

./qf-par-bench.sh ${ID}
mv -i ribbon-par-qf-${ID}-*-raw.txt ${DIR}

mv -n results_qf_* ${DIR}
