#!/bin/bash

tr '\r' '\n' | awk '{
  if ($12 != "" && $12 != 0) {
    for (i = 2; i <= 12; i++) {
      a[$1][i] += $i;
    }
    count[$1]++;
  }
}
END {
  for (e in a) {
    printf "%s ", e;
    for (i = 2; i <= 12; i++) {
      printf "%g ", a[e][i] / count[e];
    }
    print count[e]
  }
}' | sort
