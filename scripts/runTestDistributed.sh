#!/bin/bash

NET=$1
PARAMS=$2
TESTING=$3
SIZE=$4
EXPERIMENT=$5
MACHINES=$6

while shift && [ -n "$6" ]; do
  MACHINES="${MACHINES} $6"
done

COUNT=`echo $MACHINES | wc -w`
echo "Distributing to $COUNT machines."

MNUMBER=0
PERMACHINE=$(((SIZE/COUNT) + 1))
for machine in $MACHINES; do
  FIRST=$((MNUMBER * PERMACHINE))
  LAST=$(((MNUMBER+1) * PERMACHINE - 1))
  echo "Running on $machine. First: $FIRST, last: $LAST"

  MCOMMAND="~/builds/${machine}_cn24/testNetwork $NET $PARAMS $TESTING ${EXPERIMENT}_${machine}.log $FIRST $LAST 2&> ${EXPERIMENT}_${machine}.run &"

  echo "Running command: $MCOMMAND"
  ssh $machine $MCOMMAND
  MNUMBER=$((MNUMBER+1))
done
