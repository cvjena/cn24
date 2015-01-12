#!/bin/bash

SOURCE=$1

function calc {
  STAT=$1
  EPOCH=$2
  FILTERED=`cat $SOURCE | grep "$EPOCH.*$STAT"`
  FILTERED=`echo "$FILTERED" | sed "s/#.*: //g"`
  FILTERED=`echo "$FILTERED" | sed "s/%.*//g"`
  AVERAGE=`echo "$FILTERED" | awk '{s+=$1}END{print s/NR}'`
  VARIANCE=`echo "$FILTERED" | awk '{s+=($1'"-$AVERAGE"')*($1'"-$AVERAGE"')}END{print s/NR}'`
  NR=`echo "$FILTERED" | awk '{s+=$1}END{print NR}'`
  STDDEV=$(echo "scale=2;sqrt($VARIANCE)" | bc)
  CONFID=$(echo "scale=2;1.96*$STDDEV/sqrt($NR)" | bc)

  UPPER=$(echo "scale=2;$AVERAGE - $CONFID" | bc)
  LOWER=$(echo "scale=2;$AVERAGE + $CONFID" | bc)
  echo "$EPOCH,$AVERAGE,$UPPER,$LOWER"
}

calc "F1" 5 
calc "F1" 10
calc "F1" 15
calc "F1" 20
