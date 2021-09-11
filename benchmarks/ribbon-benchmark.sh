#!/bin/bash

export ID=${1:-$(hostname | cut -d 'c' -f 2)}

for MKEYS in 1 10 100; do
    export ITERS=$((49 / $MKEYS + 1))
    (for I in $(seq 1 ${ITERS}); do ./bulk-insert-and-query-${ID}.exe ${MKEYS}000000 all; done) 2>&1 | tee ribbon-results-${ID}-${MKEYS}-raw.txt
done
