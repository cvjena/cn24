#!/bin/bash
DATASET=$1
NETFILE=$2
EPOCHS=$3
ITERATIONS=$4
MODEL=$5
TIMESTAMP=`date +%s`

mkdir tmp 2&> /dev/null
mkdir logs 2&> /dev/null
mkdir csv 2&> /dev/null

echo "Running $ITERATIONS iterations ($EPOCHS * 10 epochs each, testing every 10th epoch) of network $NETFILE on dataset $DATASET, loading model $MODEL..."

SIGNATURE=$(basename "$DATASET")_$(basename "$NETFILE")_model${MODEL}_${EPOCHS}_${ITERATIONS}_$TIMESTAMP
SCRFILE=tmp/scr_$SIGNATURE
LOGFILE=tmp/log_$SIGNATURE
OLOGFILE=logs/log_$SIGNATURE
CSVFILE=csv/csv_$SIGNATURE

echo "set experiment name=$SIGNATURE" > $SCRFILE
echo "reset" >> $SCRFILE
echo "load file=$MODEL" >> $SCRFILE
echo "set epoch=0" >> $SCRFILE

for i in $(seq 1 $ITERATIONS)
do
  for j in $(seq 1 $EPOCHS)
  do
    MODELFILE=tmp/model_${SIGNATURE}_i${i}_j$j
    echo "tstat enable=0" >> $SCRFILE
    echo "train epochs=9" >> $SCRFILE
    echo "tstat enable=1" >> $SCRFILE
    echo "train" >> $SCRFILE
    echo "save file=$MODELFILE" >> $SCRFILE
    echo "test" >> $SCRFILE
  done
  echo "reset" >> $SCRFILE
  echo "load file=$MODEL" >> $SCRFILE
  echo "set epoch=0" >> $SCRFILE
done


./trainNetwork -v $DATASET $NETFILE $SCRFILE 2&> $LOGFILE

mv $LOGFILE $OLOGFILE
echo "Training done, output log file: $OLOGFILE"
