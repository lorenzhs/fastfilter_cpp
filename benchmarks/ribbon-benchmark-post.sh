#!/bin/bash

# only the post-computation bits of the benchmark script

export ID=${1:-$(hostname | cut -d 'c' -f 2)}

for MKEYS in 1 10 100; do
    PREF=""
    if [ $MKEYS == 10 ]; then
        PREF="D";
    elif [ $MKEYS == 100 ]; then
        PREF="C";
    fi
    echo "%\\name/\\timems/\\fprperc/\\wastedperc/\\queryms/\\addms
\\def\\data${PREF}M{" > ribbon-results-${ID}-${MKEYS}.tex
    cat ribbon-results-${ID}-${MKEYS}-raw.txt | egrep -v '(Using|Failed|FAILURE|Error)' | ./combine-results.sh | sed 's,1B_,OB_,;s,2B_,TB_,;s,_int,I,;s,_cls,C,;s,_SC,S,' | \
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
        sort -n -r | awk '{ print $2, $3 }' >> ribbon-results-${ID}-${MKEYS}.tex
    echo "\\DUMMY/10000/1/200/1000/1000}" >> ribbon-results-${ID}-${MKEYS}.tex
done
(./combine-results.sh < ribbon-results-${ID}-100-raw.txt; ./combine-results.sh < ribbon-results-${ID}-1-raw.txt) | ./summarize-results.sh  | \
    # input columns: 1 name, (2/7 size, 3/8 overhead, 4/9 construction, 5/10 pos query, 6/11 neg query)
    # -> if large overhead difference (1.7% absolute, 10% relative), use both, else average of the two
  awk '{ if ($11 != "") {
           if (($3 - $8 > 1.7 || $8 - $3 > 1.7) && ($3 / $8 > 1.1 || $8 / $3 > 1.1)) {
             ovr = int($3 + 0.5) ";" int($8 + 0.5);
           } else {
             ovr = int(($3 + $8) * 5 + 0.5)/10.0;
             if (ovr < 1.0) {
               ovr = sprintf("%.2f", ovr);
             } else {
               ovr = sprintf("%.1f", ovr);
             }
           }
printf "%28s & %7s & %4d & %3d & %3d & %4d & %3d & %3d \\\\", $1, ovr, $4, $5, $6, $9, $10, $11; print ""; }}' > ribbon-results-${ID}-table.tex
# columns: name, overhead, cons_1M, query_1M, query_1M_plusminus, cons_100M, query_100M, query_100M_plusminus

# a[name][L][optimal bits] = overhead%, print: x = -log2(overhead), y = optimal bits
# output of combine-results preserves original columns + as the last one how many reps
# 1 name | 2 add | 3 Query 0% | 4 Q 50% | 5 Q 100% | 6 add + 3Xfind | 7 ε% | 8 bits/item | 9 optimal bits/item | 10 wasted space% | 11 million keys | 12 #reps
# after processing through grep and tr, $1 is split into basename | ribbon width | filter bits (-> offset 2)
for MKEYS in 1 10 100; do
    # HomogRibbon: use measured fprate
    ./combine-results.sh < ribbon-results-${ID}-${MKEYS}-raw.txt | egrep 'HomogRibbon' | egrep '^[^_]*_1?[13579] ' | tr '_' ' ' | grep -v 128Pack | sed 's/Pack//;s/Ribbon/ /' | awk '{ a[$1][$2][$11] = $12; } END { for (s in a) { for (b in a[s]) { printf "    \\addplot coordinates {"; PROCINFO["sorted_in"] = "@ind_num_asc"; for (r in a[s][b]) { printf "({-log2(%f)},%f) ", a[s][b][r]/100.0, r } printf "}; %%%s%d\n    \\addlegendentry{%s%d}\n", s, b, s, b } } }' > ribbon-results-${ID}-${MKEYS}-plot.tex
    # BalancedRibbon (Bu^1RR) and BumpRibbon (BuRR): use claimed fprate
    ./combine-results.sh < ribbon-results-${ID}-${MKEYS}-raw.txt | egrep 'BalancedRibbon' | egrep '^[^_]*_1?[13579] ' | tr '_' ' ' | grep -v 128Pack | sed 's/Pack//;s/Ribbon/ /' | awk '{ a[$1][$2][$11] = ($10 - $3)/$3*100.0; } END { for (s in a) { for (b in a[s]) { printf "    \\addplot coordinates {"; PROCINFO["sorted_in"] = "@ind_num_asc"; for (r in a[s][b]) { printf "({-log2(%f)},%f) ", a[s][b][r]/100.0, r } printf "}; %%%s%d\n    \\addlegendentry{%s%d}\n", s, b, s, b } } }' >> ribbon-results-${ID}-${MKEYS}-plot.tex
    ./combine-results.sh < ribbon-results-${ID}-${MKEYS}-raw.txt | grep 'BumpRibbon.*_int' | tr '_' ' ' | sed 's/Ribbon//;s/ int//' | awk '{ a[$1][$2][$11] = ($10 - $3)/$3*100.0; } END { for (s in a) { for (b in a[s]) { printf "    \\addplot coordinates {"; PROCINFO["sorted_in"] = "@ind_num_asc"; for (r in a[s][b]) { printf "({-log2(%f)},%f) ", a[s][b][r]/100.0, r } printf "}; %%%s%d\n    \\addlegendentry{%s%d}\n", s, b, s, b } } }' >> ribbon-results-${ID}-${MKEYS}-plot.tex
    ./combine-results.sh < ribbon-results-${ID}-${MKEYS}-raw.txt | egrep 'Rocks' | grep -v 'Compare' | tr 'K()' '   ' | sed 's/Bloom/Bloom /' | awk '{ a[$1][42][$11] = $12; } END { for (s in a) { for (b in a[s]) { printf "    \\addplot coordinates{"; PROCINFO["sorted_in"] = "@ind_num_asc"; for (r in a[s][b]) { printf "({-log2(%f)},%f) ", a[s][b][r]/100.0, r } printf "}; %%%s%d\n    \\addlegendentry{%s%d}\n", s, b, s, b } } }' >> ribbon-results-${ID}-${MKEYS}-plot.tex
done
