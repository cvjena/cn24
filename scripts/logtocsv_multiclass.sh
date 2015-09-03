#!/bin/bash
INPUTFILE=$1
#OUTPUTFILE=${INPUTFILE}.csv
#echo "Converting $INPUTFILE to $OUTPUTFILE"

echo -ne "Training,Epoch,AggregateLoss,OverallRR,AverageRR,AverageIOU"
cat $INPUTFILE |\
sed -r "s/\x1B\[([0-9]{1,2}(;[0-9]{1,2})?)?[mGK]//g" |\
grep -E "(intersection)|(recognition rate)|(aggregate lps)" |\
sed -r 's/INF.*Training \(Epoch ([0-9]+).*aggregate lps: /;1,\1,/' |\
sed -r 's/RESULT --- Training.*Epoch ([0-9]+) .*: /,/' |\
sed -r 's/INF.*Testing \(Epoch ([0-9]+).*aggregate lps: /;0,\1,/' |\
sed -r 's/RESULT --- Testing.*Epoch ([0-9]+) .*: /,/' |\
perl -p -e 's/\n//' |\
sed -r 's/;/\n/g' |\
sed -r 's/%//g'
echo
