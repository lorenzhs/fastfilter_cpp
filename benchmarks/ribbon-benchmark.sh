#!/bin/bash

for MKEYS in 1 10 100; do
  export ITERS=$((49 / $MKEYS + 1))
  (for I in `seq 1 $ITERS`; do ./bulk-insert-and-query.exe ${MKEYS}000000 all; done) 2>&1 | tee ribbon-results-${MKEYS}-raw.txt
  cat ribbon-results-${MKEYS}-raw.txt | ./combine-results.sh | grep -v PowTwo | grep -v Fuse | \
    awk '{ if ($12 != "") { short=$1;
                            gsub(/[^A-Za-z].*/, "", short);
                            gsub(/[A-Za-z]*[^d]Bloom/, "Bloom", short);
                            print $7, "\\" short "/" $7 "/" $8 "/" $11 ", %" $1 }}' | \
    sort -n -r | awk '{ print $2, $3 }' > ribbon-results-${MKEYS}.tex
done
(./combine-results.sh < ribbon-results-100-raw.txt; ./combine-results.sh < ribbon-results-1-raw.txt) | ./summarize-results.sh | \
  awk '{ if ($11 != "") {
           if (($3 - $8 > 1.7 || $8 - $3 > 1.7) && ($3 / $8 > 1.1 || $8 / $3 > 1.1)) {
             ovr = int($3 + 0.5) ";" int($8 + 0.5);
           } else {
             ovr = sprintf("%.1f", int(($3 + $8) * 5 + 0.5)/10.0);
           }
printf "%28s & %5s & %3d & $%3d \\pm %d$ & %3d & $%3d \\pm %d$ \\\\", $1, ovr, $4, $5, $6, $9, $10, $11; print ""; }}' > ribbon-results-table.tex
./combine-results.sh < ribbon-results-1-raw.txt | ./summarize-results.sh | egrep '(og|ed)Ribbon' | egrep '^[^_]*_1?[13579] ' | tr '_' ' ' | grep -v 128Pack | sed 's/Pack//;s/Ribbon/ /' | awk '{ a[$1][$2][$3] = $5; } END { for (s in a) { for (b in a[s]) { printf "\\draw plot coordinates {"; PROCINFO["sorted_in"] = "@ind_num_asc"; for (r in a[s][b]) { printf "(-log2(%f),%d) ", a[s][b][r]/100.0, r } printf "}; %%%s%d\n", s, b } } }' > ribbon-results-plot.tex
