#!/bin/bash

# postprocessing of parallel results

export ID=${1:-$(hostname | cut -d 'c' -f 2)}

echo "%\\name/\\timems/\\fprperc/\\wastedperc/\\queryms/\\addms
\\def\\dataP{" > ribbon-results-${ID}-par.tex
tail -n +8 ribbon-par-${ID}-raw.txt | ./combine-results.sh | egrep -v '(Using|Failed|FAILURE|Error|EXCEPTION)' | sed 's,1B_,OB_,;s,2B_,TB_,;s,_int,I,;s,_cls,C,;s,_SC,S,' | \
        # columns: $6 = add + 3xquery, $7 = eps%, $10 = wasted space, $1 = full name
        awk '{ if ($11 != "") { short=$1;
                                gsub(/[^A-Za-z].*/, "", short);
                                gsub(/[A-Za-z]*[^d]Bloom/, "Bloom", short);
                                if (short ~ /((Balanced|Bump)Ribbon|Coupled|LMSS|GOV|TwoBlock[0-9]+k)/) {
                                   split($1,a,"_");
                                   bits = a[length(a)];
                                   ovr=($8-bits)/bits*100.0
                                } else {
                                  ovr=$10;
                                }
                                print $6, "\\" short "/" $6 "/" $7 "/" ovr "/" $3 "/" $2 ", %" $1 }}'  | \
        sort -n -r | awk '{ print $2, $3 }' >> ribbon-results-${ID}-par.tex
echo "\\DUMMY/10000/1/200/1000/1000}" >> ribbon-results-${ID}-par.tex

(./combine-results.sh < ribbon-results-${ID}-100-raw.txt; ./combine-results.sh < ribbon-results-${ID}-1-raw.txt; ./combine-results.sh < ribbon-par-${ID}-raw.txt) | ./summarize-results.sh  | \
    # input columns: 1 name, (2/7/12 size, 3/8/13 overhead, 4/9/14 construction, 5/10/15 avg of pos&neg query, 6/11/16 Query plus/minus)
    # -> if large overhead difference (1.7% absolute, 10% relative), use both, else average of the two
  awk '{ if ($16 != "") {
           if (($3 - $13 > 1.7 || $13 - $3 > 1.7) && ($3 / $13 > 1.1 || $13 / $3 > 1.1)) {
             ovr = int($3 + 0.5) ";" int($13 + 0.5);
           } else {
             ovr = ($3 + $13) / 2.0;
             if (ovr < 1.0) {
               ovr = sprintf("%.2f", ovr);
             } else {
               ovr = sprintf("%.1f", ovr);
             }
           }
printf "%28s & %7s & %4d & %3d & %3d & %4d & %4d & %3d & %4d & %3d & %3d \\\\", $1, ovr, $4, $5, $6, $9, $10, $11, $14, $15, $16; print ""; }}' > ribbon-results-par-${ID}-table.tex
# columns: name, overhead, cons, query, query_plusminus

# a[name][L][optimal bits] = overhead%, print: x = -log2(overhead), y = optimal bits
# output of combine-results preserves original columns + as the last one how many reps
# 1 name | 2 add | 3 Query 0% | 4 Q 50% | 5 Q 100% | 6 add + 3Xfind | 7 ε% | 8 bits/item | 9 optimal bits/item | 10 wasted space% | 11 million keys | 12 #reps
# after processing through grep and tr, $1 is split into basename | ribbon width | filter bits (-> offset 2)
./combine-results.sh < ribbon-par-${ID}-raw.txt | egrep '(og|ed)Ribbon' | egrep '^[^_]*_1?[13579] ' | tr '_' ' ' | grep -v 128Pack | sed 's/Pack//;s/Ribbon/ /' | awk '{ a[$1][$2][$11] = ($10 - $3)/$3*100.0; } END { for (s in a) { for (b in a[s]) { printf "    \\addplot coordinates {"; PROCINFO["sorted_in"] = "@ind_num_asc"; for (r in a[s][b]) { printf "({-log2(%f)},%f) ", a[s][b][r]/100.0, r } printf "}; %%%s%d\n    \\addlegendentry{%s%d}\n", s, b, s, b } } }' > ribbon-results-par-${ID}-plot.tex
./combine-results.sh < ribbon-par-${ID}-raw.txt | grep 'BumpRibbon.*_int' | tr '_' ' ' | sed 's/Ribbon//;s/ int//' | awk '{ a[$1][$2][$11] = ($10 - $3)/$3*100.0; } END { for (s in a) { for (b in a[s]) { printf "    \\addplot coordinates {"; PROCINFO["sorted_in"] = "@ind_num_asc"; for (r in a[s][b]) { printf "({-log2(%f)},%f) ", a[s][b][r]/100.0, r } printf "}; %%%s%d\n    \\addlegendentry{%s%d}\n", s, b, s, b } } }' >> ribbon-results-par-${ID}-plot.tex
./combine-results.sh < ribbon-par-${ID}-raw.txt | egrep 'Rocks' | grep -v 'Compare' | tr 'K()' '   ' | sed 's/Bloom/Bloom /' | awk '{ a[$1][42][$11] = $12; } END { for (s in a) { for (b in a[s]) { printf "    \\addplot coordinates{"; PROCINFO["sorted_in"] = "@ind_num_asc"; for (r in a[s][b]) { printf "({-log2(%f)},%f) ", a[s][b][r]/100.0, r } printf "}; %%%s%d\n    \\addlegendentry{%s%d}\n", s, b, s, b } } }' >> ribbon-results-par-${ID}-plot.tex
