#!/bin/bash

# This file runs the sockeye toolkit and trains on the IWSLT15 English-Vietnamese dataset using the groundhog model.

SOCKEYE_ROOT=$(cd $(dirname $0)/../.. && pwd)
PROJECT_ROOT=${SOCKEYE_ROOT}/../..
CONFERENCE_SRC_TGT=iwslt15-vi_en
CONFERENCE_SRC_TGT_MODEL_OPT=${CONFERENCE_SRC_TGT}-groundhog-${USE_MLP_ATT_SCORING_FUNC}${USE_FUSED_LSTM_NONLIN_BLOCK}${USE_PAR_SEQUENCE_REVERSE}

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
BATCH_SIZE=80
INITIAL_LEARNING_RATE=0.0002
CHECKPOINT_FREQUENCY=4000

cd ${SOCKEYE_ROOT} && rm -rf ${SOCKEYE_ROOT}/workspace/${CONFERENCE_SRC_TGT_MODEL_OPT} && \
PYTHONPATH=${SOCKEYE_ROOT} ${NVPROF_PREFIX} \
python3 -m sockeye.train --source ${SOCKEYE_ROOT}/workspace/data/${CONFERENCE_SRC_TGT}/train-preproc.vi \
                         --target ${SOCKEYE_ROOT}/workspace/data/${CONFERENCE_SRC_TGT}/train-preproc.en \
                         --source-vocab ${SOCKEYE_ROOT}/workspace/data/${CONFERENCE_SRC_TGT}/vocab.vi \
                         --target-vocab ${SOCKEYE_ROOT}/workspace/data/${CONFERENCE_SRC_TGT}/vocab.en \
                         --validation-source ${SOCKEYE_ROOT}/workspace/data/${CONFERENCE_SRC_TGT}/tst2012.vi \
                         --validation-target ${SOCKEYE_ROOT}/workspace/data/${CONFERENCE_SRC_TGT}/tst2012.en \
                         --output ${SOCKEYE_ROOT}/workspace/${CONFERENCE_SRC_TGT_MODEL_OPT} --seed=1 \
                         --batch-type=sentence --batch-size=${BATCH_SIZE} --bucket-width=10 \
                         --checkpoint-frequency=${CHECKPOINT_FREQUENCY} --device-ids=0 --embed-dropout=0.3:0.3 \
                         --encoder=rnn --decoder=rnn --num-layers=1:1 --rnn-cell-type=lstm --rnn-num-hidden=1000 \
                         --rnn-residual-connections --layer-normalization \
                         --rnn-attention-type=mlp --rnn-attention-num-hidden=512 \
                         --rnn-attention-use-prev-word --rnn-attention-in-upper-layers --rnn-attention-coverage-num-hidden=1 \
                         --rnn-attention-coverage-type=count --rnn-decoder-state-init=zero \
                         --rnn-attention-use-prev-word --rnn-dropout-inputs=0:0 --rnn-dropout-states=0.0:0.0 \
                         --rnn-dropout-recurrent=0.0:0.0 --rnn-decoder-hidden-dropout=0.3 \
                         --fill-up=replicate --max-seq-len=50:50 --loss=cross-entropy \
                         --num-embed 500:500 --num-words 50000:50000 --word-min-count 1:1 \
                         --optimizer=adam --optimized-metric=perplexity --clip-gradient=1.0 \
                         --initial-learning-rate=${INITIAL_LEARNING_RATE} --learning-rate-reduce-num-not-improved=8 --learning-rate-reduce-factor=0.7 \
                         --learning-rate-scheduler-type=plateau-reduce --learning-rate-warmup=0 \
                         --max-num-checkpoint-not-improved=16 --min-num-epochs=1 \
                         --monitor-bleu=500 --keep-last-params=60 --lock-dir /var/lock --use-tensorboard ${MAX_UPDATES} \
                         2>&1 | tee /tmp/${CONFERENCE_SRC_TGT_MODEL_OPT}.log
