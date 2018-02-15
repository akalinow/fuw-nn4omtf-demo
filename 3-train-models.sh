#!/bin/bash

. paths.sh

omtfrunner train -v --sess_prefix demo --logs ${TRAIN_LOG} --batch_size 1000 --reps 2 ${DATASET} ${NETS}/*
