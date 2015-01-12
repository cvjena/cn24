#!/bin/bash

SOURCE=$1
EPOCH=$2

function calc {
  STAT=$1
  FILTERED=`cat $SOURCE | grep "$EPOCH.*$STAT"`
  FILTERED=`echo "$FILTERED" | sed "s/#.*: //g"`
  FILTERED=`echo "$FILTERED" | sed "s/%.*//g"`
  AVERAGE=`echo "$FILTERED" | awk '{s+=$1}END{print s/NR}'`
  VARIANCE=`echo "$FILTERED" | awk '{s+=($1'"-$AVERAGE"')*($1'"-$AVERAGE"')}END{print s/NR}'`
  NR=`echo "$FILTERED" | awk '{s+=$1}END{print NR}'`
  STDDEV=$(echo "scale=2;sqrt($VARIANCE)" | bc)
  CONFID=$(echo "scale=2;1.96*$STDDEV/sqrt($NR)" | bc)

  echo "$STAT: $AVERAGE% +/- $CONFID% (95%)"
}

calc "F1"
calc "PRE"
calc "REC"
calc "FPR"
calc "FNR"
