ROOT_DATASET ?="" # Not present
NPZ_DATASET ?="" # Not present

DATASET_DIR=demo-dataset
DATASET_PREF=test
DATASET_PATH=${DATASET_DIR}/${DATASET_PREF}-dataset.npz

BUILDERS=builder-code
MODEL=test-model

0-root-to-np:
	# ROOT dataset should contain directories matching 'SingleMu_[0-9]+_[pm]'
	# This command converts all ROOT's data into *.npz files with paths 
	# matching "NPZ_DATASET/SingleMu_[pm]_[0-9]+.npz"
	omtfdatasettool root2np -v ${NPZ_DATASET} ${ROOT_DATASET}/SingleMu*

1-create-dataset:
	omtftool create_dataset --outdir ${DATASET_DIR} ${DATASET_PREF} ${NPZ_DATASET}/* --transform 0 600

2-create-model:
	omtfrunner model ${BUILDERS}/fc.py ${MODEL} --batch_size 64 --ds_train ${DATASET_PATH} --ds_valid ${DATASET_PATH} \
		--ds_test ${DATASET_PATH} --lrate 0.0005

3-train-model:
	omtfrunner train ${MODEL} --epochs 100 --time_limit 0:01:00

4-test-model:
	omtfrunner test ${MODEL} --note 'test' --suffix 'first-test' 


clean:
	rm -rf ${MODEL}

