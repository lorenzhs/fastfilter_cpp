#!/bin/bash

for MKEYS in 1 10 100; do
  ./bulk-insert-and-query.exe ${MKEYS}000000 all 2>&1 | tee ribbon-results-${MKEYS}-raw.txt
  cat ribbon-results-${MKEYS}-raw.txt | tr '\r' '\n' | \
    awk '{ if ($12 != "") { gsub(/[^A-Za-z].*/, "", $1); print $7, "\\" $1 "/" $7 "/" $8 "/" $11 ","}}' | \
    sort -n -r | awk '{ print $2 }' > ribbon-results-${MKEYS}.tex
done
