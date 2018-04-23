# nn4omtf demo & tutorial

## NOTICE
[23.04.18]
Due to many changes in main package, some scripts are outdated now.
Other features are also available.
This demo and tutorial scripts will be updated at the end.

---

Small demo for [nn4omtf](https://github.com/jlysiak/fuw-nn4omtf) package and its 
command line tools [fuw-nn4omtf-cli](https://github.com/jlysiak/fuw-nn4omtf-cli).
This demo doesn't contain original dataset in ROOT format nor any source `*.npz` 
file however repository contains apropriate scripts to perform conversion operations.

## Before you start...
... install `nn4omtf` and `nn4omtf-cli`.
TensorBoard is required to see all training statistics and graphs.
Some basic info and logs is saved along with model graph in OMTFNN class.
Graph topology preview is available in model directory as HTML file.

## Description
Run scripts one by one to reach the goal.

0. `0-root2numpy.sh` - convert ROOT dataset into Numpy dataset
1. `1-make-dataset.sh` - prepare demo OMTF Dataset (in fact set of TFRecords files) from Numpy dataset
2. `2-create-nets.sh` - create models using provided builder code
  * `2a-preview-arch.sh` - lookup created architectures in webbrowser (works with chrome)
3. `3-train-models.sh` - run training on selected networks
  * `3a-train-preview.sh` - run tensorboard to watch training results
  * `3b-show-models-log.sh` - print training logs on stdout
  * `3c-show-models-valid-logs.sh` - show logs from valid runs stored with model (those logs are always saved)
4. `4-compare-models.sh` - compare selected models accuracy on test dataset

