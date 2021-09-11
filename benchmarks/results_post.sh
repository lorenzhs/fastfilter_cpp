#!/bin/bash

# find quotient filter results
shopt -s nullglob
set -- results_qf_*/results_*.txt
if [ "$#" -gt 0 ]; then
    cat results_qf_*/results_{734003,786432,838860,891289,943718,996147}_*.txt > results_qf_1.txt
    cat results_qf_*/results_{11744051,12582912,13421772,14260633,15099494,15938355}_*.txt > results_qf_10.txt
    cat results_qf_*/results_{93952409,100663296,107374182,114085068,120795955,127506841}_*.txt > results_qf_100.txt
    cat results_qf_*/results_par_*t_*f_{11744051,12582912,13421772,14260633,15099494,15938355}_*.txt > results_qf_par.txt
fi

for MKEYS in 1 10 100 par; do
    if [ $MKEYS == 'par' ]; then
        PREFIX=par
    else
        PREFIX=${MKEYS}000000;
    fi
    cat results*/results_${PREFIX}_*.txt results_qf*_${MKEYS}.txt | \
        # 1 RESULT, 2 name, 3 n, 4 fpp, 5 bits, 6 minbits, 7 wasted, 8 tadd, 9 tfind0
        # 10 find50 11 tfind100 12 tcombined
        awk '{ if ($2 ~ /((Balanced|Bump)Ribbon|Coupled|LMSS|GOV|TwoBlock[0-9]+k)/) {
            if ($3 == "n=0") {
               ovr = "inf";
            } else {
               split($2,a,"_");
               minbits = a[length(a)];
               split($5,a,"=");
               bits = a[length(a)];
               ovr=(bits-minbits)/minbits*100.0;
            }
        } else {
            split($7,a,"=");
            ovr = a[length(a)];
        }
        print $0 " realovr=" ovr;
        }' > results_${MKEYS}M.txt
done

# lol hack
mv results_parM.txt results_par_10M.txt
