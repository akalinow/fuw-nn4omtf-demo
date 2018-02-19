#!/bin/bash

. paths.sh
rm -fr ${TRAIN_LOG}

omtfrunner -v --sess_prefix demo --logs ${TRAIN_LOG} --steps 101 --batch_size 1000 --reps 1 ${DATASET} ${NETS}/*
