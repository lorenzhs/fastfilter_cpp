#!/bin/bash

# input columns: original columns, i.e.
# 1 name | 2 add | 3 Query 0% | 4 Q 50% | 5 Q 100% | 6 add + 3Xfind |
# 7 ε% | 8 bits/item | 9 optimal bits/item | 10 wasted space% | 11 million keys
tr '\r' '\n' | awk '{
  if ($11 != "" && $11 != 0) {
    qneg[$1][$11] = $3;
    qpos[$1][$11] = $5;
    if ($1 ~ /((Balanced|Bump)Ribbon|Coupled|LMSS|GOV|TwoBlock[0-9]+k)/) {
       split($1,a,"_");
       bits = a[length(a)];
       ovr[$1][$11] = ($8-bits)/bits*100.0;
    } else {
        ovr[$1][$11] = $10;
    }
    const[$1][$11] = int($2 + 0.5);
  }
}
END {
  for (e in const) {
    printf "%s ", e;
    PROCINFO["sorted_in"] = "@ind_num_asc"
    for (i in const[e]) {
      printf "%d %g %d %d %d ", i, ovr[e][i], const[e][i], qpos[e][i], qneg[e][i];
    }
    print ""
  }
}' | sort

# output columns: name, size, overhead, construction, Qmid, QPM
# QPM (query plus/minus) = (pos - neg) / 2
# Qmid = negative query + QPM
