#!/bin/bash
DATASET=$1
NETFILE=$2
EPOCHS=$3
ITERATIONS=$4
TIMESTAMP=`date +%s`

mkdir tmp 2&> /dev/null
mkdir logs 2&> /dev/null
mkdir csv 2&> /dev/null

echo "Running $ITERATIONS iterations ($EPOCHS epochs each) of network $NETFILE on dataset $DATASET..."

SIGNATURE=$(basename "$DATASET")_$(basename "$NETFILE")_${EPOCHS}_${ITERATIONS}_$TIMESTAMP
SCRFILE=tmp/scr_$SIGNATURE
LOGFILE=tmp/log_$SIGNATURE
OLOGFILE=logs/log_$SIGNATURE
CSVFILE=csv/csv_$SIGNATURE

echo "reset" > $SCRFILE
echo "set epoch=0" >> $SCRFILE

for i in $(seq 1 $ITERATIONS)
do
  for j in $(seq 1 $EPOCHS)
  do
    MODELFILE=tmp/model_${SIGNATURE}_i${i}_j$j
    echo "train" >> $SCRFILE
    echo "save file=$MODELFILE" >> $SCRFILE
    echo "test" >> $SCRFILE
  done
  echo "reset" >> $SCRFILE
  echo "set epoch=0" >> $SCRFILE
done


./trainNetwork $DATASET $NETFILE $SCRFILE 2&> $LOGFILE

mv $LOGFILE $OLOGFILE
echo "Training done, output log file: $OLOGFILE"

./logtocsv_multiclass.sh $OLOGFILE > $CSVFILE
echo "Output CSV file: $CSVFILE"
