#!/bin/bash

# This file runs the sockeye toolkit and trains on the IWSLT15 English-Vietnamese dataset using the groundhog model.

SOCKEYE_ROOT=$(cd $(dirname $0)/../.. && pwd)
PROJECT_ROOT=${SOCKEYE_ROOT}/../..
CONFERENCE_SRC_TGT=iwslt15-vi_en
CONFERENCE_SRC_TGT_MODEL_OPT=${CONFERENCE_SRC_TGT}-best-${USE_MLP_ATT_SCORING_FUNC}${USE_FUSED_LSTM_NONLIN_BLOCK}${USE_PAR_SEQUENCE_REVERSE}

MAX_UPDATES=
if [ "$1" == "--full-run" ]
then
        echo "Training will run until completion."
else
        echo "Training will stop early."
        MAX_UPDATES="--max-updates=500"
fi
# ==================================================================================================
NVPROF_PREFIX=
if [ "$1" == "--nvprof-runtime" ]
then
	echo "nvprof is enabled to profile the runtime."
        mkdir -p ${SOCKEYE_ROOT}/workspace/profile/runtime
	NVPROF_PREFIX="/usr/local/cuda/bin/nvprof --profile-from-start off \
                --csv --log-file ${SOCKEYE_ROOT}/workspace/profile/runtime/${CONFERENCE_SRC_TGT_MODEL_OPT}.csv"
        MAX_UPDATES="--max-updates=200"
fi
if [ "$1" == "--nvprof-dram-transactions" ]
then
	echo "nvprof is enabled to profile the DRAM transactions."
        mkdir -p ${SOCKEYE_ROOT}/workspace/profile/dram_transactions
	NVPROF_PREFIX="/usr/local/cuda/bin/nvprof --profile-from-start off \
                --metrics dram_read_transactions,dram_write_transactions \
                --csv --log-file ${SOCKEYE_ROOT}/workspace/profile/dram_transactions/${CONFERENCE_SRC_TGT_MODEL_OPT}.csv"
        MAX_UPDATES="--max-updates=200"
fi
# ==================================================================================================
BATCH_SIZE=32
INITIAL_LEARNING_RATE=0.0002
CHECKPOINT_FREQUENCY=4000

cd ${SOCKEYE_ROOT} && rm -rf ${SOCKEYE_ROOT}/workspace/${CONFERENCE_SRC_TGT_MODEL} && \
PYTHONPATH=${SOCKEYE_ROOT} ${NVPROF_PREFIX} \
python3 -m sockeye.train --source ${SOCKEYE_ROOT}/workspace/data/${CONFERENCE_SRC_TGT}/train-preproc.vi \
                         --target ${SOCKEYE_ROOT}/workspace/data/${CONFERENCE_SRC_TGT}/train-preproc.en \
                         --source-vocab ${SOCKEYE_ROOT}/workspace/data/${CONFERENCE_SRC_TGT}/vocab.vi \
                         --target-vocab ${SOCKEYE_ROOT}/workspace/data/${CONFERENCE_SRC_TGT}/vocab.en \
                         --validation-source ${SOCKEYE_ROOT}/workspace/data/${CONFERENCE_SRC_TGT}/tst2012.vi \
                         --validation-target ${SOCKEYE_ROOT}/workspace/data/${CONFERENCE_SRC_TGT}/tst2012.en \
                         --output ${SOCKEYE_ROOT}/workspace/${CONFERENCE_SRC_TGT_MODEL} --seed=1 \
                         --batch-size=${BATCH_SIZE} --checkpoint-frequency=${CHECKPOINT_FREQUENCY} \
                         --embed-dropout=0.1:0.1 --rnn-decoder-hidden-dropout=0.2 --layer-normalization \
                         --num-layers=4:4 --max-seq-len=100:100 --label-smoothing 0.1 \
                         --weight-tying --weight-tying-type=src_trg --num-embed 512:512 --num-words 50000:50000 \
                         --word-min-count 1:1 --optimizer=adam  --initial-learning-rate=${INITIAL_LEARNING_RATE} \
                         --learning-rate-reduce-num-not-improved=8 --learning-rate-reduce-factor=0.7 \
                         --learning-rate-scheduler-type=plateau-reduce --max-num-checkpoint-not-improved=32 --min-num-epochs=0 --rnn-attention-type mlp \
                         --monitor-bleu=500 --keep-last-params=60 --lock-dir /var/lock --use-tensorboard ${MAX_UPDATES}
