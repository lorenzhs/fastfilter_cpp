#!/bin/bash

tr '\r' '\n' | awk '{
  if ($12 != "" && $12 != 0) {
    qa = $4;
    qb = $5;
    qc = $6;
    if (qa > qb) {
      tmp = qa; qa = qb; qb = tmp;
    }
    if (qb > qc) {
      tmp = qb; qb = qc; qc = tmp;
    }
    if (qa > qb) {
      tmp = qa; qa = qb; qb = tmp;
    }
    my_qpm = (qc - qa) / 2;
    my_qmid = qa + my_qpm;
    qmid[$1][$12] = int(my_qmid + 0.5);
    qpm[$1][$12] = int(my_qpm + 0.5);
    ovr[$1][$12] = $11
    const[$1][$12] = int($2 + 0.5);
  }
}
END {
  for (e in const) {
    printf "%s ", e;
    PROCINFO["sorted_in"] = "@ind_num_asc"
    for (i in const[e]) {
      printf "%d %g %d %d %d ", i, ovr[e][i], const[e][i], qmid[e][i], qpm[e][i];
    }
    print ""
  }
}' | sort
