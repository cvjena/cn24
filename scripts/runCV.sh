#/bin/bash

NETFACTORY=$2
TENSOR=$1

echo "Running CV on $NETFACTORY"
echo "Using data tensor: $TENSOR"

date > logs/cv_$NETFACTORY

for s in {0..9}
do
  echo "Running subset $s..."
  ./trainNetwork $TENSOR $NETFACTORY$s 2&> logs/cv_$NETFACTORY$s
  cat logs/cv_$NETFACTORY$s | grep 'RESULT.*Testing' | sed -e "s/^/#$s#/" >> logs/cv_$NETFACTORY
done

echo 'Running complete training...'

./trainNetwork $TENSOR "$NETFACTORY"X 2&> logs/cv_"$NETFACTORY"X

echo 'Done.'
