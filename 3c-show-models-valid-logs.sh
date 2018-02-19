#!/bin/bash

. paths.sh

SINGLE_NETS=${NETS}/*/logs
SINGLE_NETS=$(echo ${SINGLE_NETS} | sed s/' '/','/g)
STORE_NETS=${NETS}/*/nets/*/logs
STORE_NETS=$(echo ${STORE_NETS} | sed s/' '/','/g)
LOGS=$SINGLE_NETS,$STORE_NETS

tensorboard --logdir $LOGS
