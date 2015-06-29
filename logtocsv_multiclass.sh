#!/bin/bash
INPUTFILE=$1
#OUTPUTFILE=${INPUTFILE}.csv
#echo "Converting $INPUTFILE to $OUTPUTFILE"

echo -ne "Training,Epoch,AggrateLoss,OverallRR,AverageRR,AverageIOU"
cat $INPUTFILE |\
sed -r "s/\x1B\[([0-9]{1,2}(;[0-9]{1,2})?)?[mGK]//g" |\
grep -E "(intersection)|(recognition rate)|(aggregate lps)" |\
sed -r 's/DBG.*Training \(Epoch ([0-9]+).*aggregate lps: /;1,\1,/' |\
#sed -r 's/RESULT --- Training.*Epoch ([0-9]+) - Overall.*: /;1,\1,/' |\
sed -r 's/RESULT --- Training.*Epoch ([0-9]+) .*: /,/' |\
perl -p -e 's/\n//' |\
sed -r 's/;/\n/g' |\
sed -r 's/%//g'
echo
