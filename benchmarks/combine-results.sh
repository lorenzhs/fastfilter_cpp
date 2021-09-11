#!/bin/bash

tr '\r' '\n' | awk '{
  if ($11 != "" && $11 != 0) {
    for (i = 2; i <= 11; i++) {
      a[$1][i] += $i;
    }
    count[$1]++;
  }
}
END {
  for (e in a) {
    printf "%s ", e;
    for (i = 2; i <= 11; i++) {
      printf "%g ", a[e][i] / count[e];
    }
    print count[e]
  }
}' | sort
# output: original columns, i.e.
# name | add | Query 0% | Q 50% | Q 100% | add + 3Xfind | ε% | bits/item | optimal bits/item | wasted space% | million keys
