#!/bin/bash

. paths.sh

omtfrunner --test -v ${DATASET} ${NETS}/*

