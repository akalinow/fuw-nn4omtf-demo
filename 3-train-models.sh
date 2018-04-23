#!/bin/bash

. paths.sh
rm -fr ${TRAIN_LOG}
DATASET=$1
echo ${DATASET}

omtfrunner train -v --filter r2 -lim 5000 -a 100 -d 1000 --debug --sess_prefix demo --log txt --logdir ${TRAIN_LOG} --learning_rate 0.004 --batch_size 100 --epochs 5 ${DATASET} ${NETS}/*

