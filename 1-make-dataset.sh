#!/bin/bash

. paths.sh

# Wanna more info, run: omtfdatasettool create -h
omtfdatasettool create -c -v --events_frac 0.01 ${DATASET} . ${NPZ_DATASET}/*
