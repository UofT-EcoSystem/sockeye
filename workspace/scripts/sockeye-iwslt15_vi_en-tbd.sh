#!/bin/bash

# This file runs the sockeye toolkit and trains on the IWSLT15 English-Vietnamese dataset using the groundhog model.

SOCKEYE_ROOT=$(cd $(dirname $0)/../.. && pwd)
PROJECT_ROOT=${SOCKEYE_ROOT}/../..
CONFERENCE_SRC_TGT=iwslt15-vi_en
CONFERENCE_SRC_TGT_MODEL=${CONFERENCE_SRC_TGT}-tbd

PARTIAL_FORWARD_PROP=
if [ "$1" == "--legacy" ]
then
	echo "Backpropagation will be done using Legacy approach."
elif [ "$1" == "--partial-fw-prop" ]
then
	echo "Backpropagation will be done using Partial Forward Propagation."
	PARTIAL_FORWARD_PROP="--rnn-attention-partial-fw-prop"
else
	echo "Backpropagation will be done using Legacy approach."
fi
# ==================================================================================================
NVPROF_PREFIX=
if [ "$1" == "--nvprof" ] || [ "$2" == "--nvprof" ]
then
	echo "nvprof is enabled to profile the application."
	NVPROF_PREFIX="/usr/local/cuda/bin/nvprof --profile-from-start off"
fi
if [ "$1" == "--nvprof-runtime" ] || [ "$2" == "--nvprof-runtime" ]
then
	echo "nvprof is enabled to profile the runtime."
	NVPROF_PREFIX="/usr/local/cuda/bin/nvprof --profile-from-start off \
                --csv --log-file ${SOCKEYE_ROOT}/workspace/results/profile/runtime/${CONFERENCE_SRC_TGT_MODEL}.csv"
fi
# ==================================================================================================
BATCH_SIZE=256
INITIAL_LEARNING_RATE=0.0006
MAX_UPDATES=10000
CHECKPOINT_FREQUENCY=1000

cd ${SOCKEYE_ROOT} && rm -rf ${SOCKEYE_ROOT}/workspace/${CONFERENCE_SRC_TGT_MODEL} && \
PYTHONPATH=${SOCKEYE_ROOT} ${NVPROF_PREFIX} \
python3 -m sockeye.train --source ${SOCKEYE_ROOT}/workspace/data/${CONFERENCE_SRC_TGT}/train-preproc.en \
			 --target ${SOCKEYE_ROOT}/workspace/data/${CONFERENCE_SRC_TGT}/train-preproc.vi \
			 --validation-source ${SOCKEYE_ROOT}/workspace/data/${CONFERENCE_SRC_TGT}/tst2012.en \
			 --validation-target ${SOCKEYE_ROOT}/workspace/data/${CONFERENCE_SRC_TGT}/tst2012.vi \
			 --source-vocab ${SOCKEYE_ROOT}/workspace/data/${CONFERENCE_SRC_TGT}/vocab.en \
			 --target-vocab ${SOCKEYE_ROOT}/workspace/data/${CONFERENCE_SRC_TGT}/vocab.vi \
			 --output ${SOCKEYE_ROOT}/workspace/${CONFERENCE_SRC_TGT_MODEL} --seed=1 \
			 --encoder rnn --decoder rnn \
			 --num-layers 2:2 \
			 --rnn-cell-type lstm \
			 --rnn-num-hidden 512 \
			 --rnn-encoder-reverse-input \
			 --num-embed 512:512 \
			 --rnn-attention-type mlp --rnn-attention-num-hidden 512 \
			 --batch-size ${BATCH_SIZE} --initial-learning-rate ${INITIAL_LEARNING_RATE} \
			 --bucket-width 10 \
			 --metrics perplexity \
			 --optimized-metric bleu \
			 --checkpoint-frequency ${CHECKPOINT_FREQUENCY} \
			 --max-num-checkpoint-not-improved 5 \
			 --weight-init uniform --weight-init-scale 0.1 \
			 --monitor-bleu 500 --use-tensorboard \
			 --max-updates ${MAX_UPDATES} ${PARTIAL_FORWARD_PROP} 2>&1 | tee ${SOCKEYE_ROOT}/workspace/results/log/${CONFERENCE_SRC_TGT_MODEL}.log