#!/bin/bash

for MKEYS in 1 10 100; do
  ./bulk-insert-and-query.exe ${MKEYS}000000 all 2>&1 | tee ribbon-results-${MKEYS}-raw.txt
  cat ribbon-results-${MKEYS}-raw.txt | tr '\r' '\n' | \
    awk '{ if ($12 != "") { short=$1; gsub(/[^A-Za-z].*/, "", short); print $7, "\\" short "/" $7 "/" $8 "/" $11 ", %" $1 }}' | \
    sort -n -r | awk '{ print $2, $3 }' > ribbon-results-${MKEYS}.tex
done
