#!/bin/bash

. paths.sh
rm -fr ${TRAIN_LOG}

echo ${DATASET}

omtfrunner -v --sess_prefix demo --logdir ${TRAIN_LOG} --learning_rate 0.03 --batch_size 100 --epochs 1 ${DATASET} ${NETS}/*
