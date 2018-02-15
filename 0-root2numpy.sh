#!/bin/bash

. paths.sh

# ROOT dataset should contain directories matching 'SingleMu_[0-9]+_[pm]'
# This command converts all ROOT's data into *.npz files with paths 
# matching "NPZ_DATASET/SingleMu_[pm]_[0-9]+.npz"
omtfdatasettool root2np -v ${NPZ_DATASET} ${ROOT_DATASET}/SingleMu*
