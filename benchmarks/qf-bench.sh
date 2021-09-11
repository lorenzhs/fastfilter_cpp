#!/bin/bash

export ID=${1:-$(hostname | cut -d 'c' -f 2)}

declare -a EXP=([1]=20 [10]=24 [100]=27)

for MKEYS in 1 10 100; do
    export ITERS=$((49 / $MKEYS + 1))
    for FILL in 0.7 0.75 0.8 0.85 0.9 0.95; do
        RESFOLDER=results_qf_${FILL}
        mkdir -p ${RESFOLDER}
        export KEYS=$(echo "2^${EXP[$MKEYS]}*${FILL}" | bc)
        echo "MKEYS=${MKEYS} FILL=${FILL} KEYS=${KEYS}"
        export ALGS=30,31,21,33
        if [[ ${FILL} != 0.95 ]]; then
            export ALGS=${ALGS},8500,8501,8502
        fi
        (for I in $(seq 1 ${ITERS}); do ./bulk-insert-and-query-${ID}.exe ${KEYS} ${ALGS}; done) 2>&1 | tee ribbon-results-qf-${ID}-${MKEYS}-${FILL}-raw.txt
        mv results_*.txt ${RESFOLDER}
    done
done
