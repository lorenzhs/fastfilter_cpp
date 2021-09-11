#!/bin/bash

ulimit -c 0

export ID=${1:-$(hostname | cut -d 'c' -f 2)}

for FILL in 0.7 0.75 0.8 0.85 0.9 0.95; do
    RESFOLDER=results_qf_${FILL}
    mkdir -p ${RESFOLDER}
    KEYS=$(echo "2^24*${FILL}" | bc)
    # num filters = next-lower multiple of 64 so that keys * filters <= 1.28*10^10
    FILTERS=$(echo "12800000000/(64*${KEYS})*64" | bc)
    ALGS=30,31,21,33
    if [[ ${FILL} != 0.95 ]]; then
        ALGS=${ALGS},8500,8501,8502
    fi
    ./par-bench-${ID}.exe 64 ${FILTERS} ${KEYS} ${ALGS} 2>&1 | tee ribbon-par-qf-${ID}-${FILL}-raw.txt
    mv results_*.txt ${RESFOLDER}
done
