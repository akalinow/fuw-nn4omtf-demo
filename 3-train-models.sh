#!/bin/bash

. paths.sh
rm -fr ${TRAIN_LOG}

echo ${DATASET}

omtfrunner train -vv -lim 5000 -a 50 --sess_prefix demo --logdir ${TRAIN_LOG} --learning_rate 0.002 --batch_size 300 --epochs 5 ${DATASET} ${NETS}/*
