#!/bin/bash

# This file runs the sockeye toolkit and trains on the IWSLT15 English-Vietnamese dataset using the groundhog model.
# ==============================================================================
CONFERENCE_SRC_TGT=
TRAIN_SRC=
TRAIN_SRC=
VOCABULARY_SRC=
VOCABULARY_TGT=
VALIDATION_SRC=
VALIDATION_TGT=

if [ "$1" == "wmt16-de_en" ]
then
        echo "Running with WMT 16 English-German dataset."
        CONFERENCE_SRC_TGT=wmt16-de_en
        TRAIN_SRC=train.tok.clean.bpe.32000.de
        TRAIN_TGT=train.tok.clean.bpe.32000.en
        VOCABULARY_SRC=vocab.bpe.32000.de
        VOCABULARY_TGT=vocab.bpe.32000.en
        VALIDATION_SRC=newstest2015.tok.bpe.32000.de
        VALIDATION_TGT=newstest2015.tok.bpe.32000.en
else
        echo "Running with IWSLT 15 English-Vietnamese dataset."
        CONFERENCE_SRC_TGT=iwslt15-vi_en
        TRAIN_SRC=train-preproc.vi
        TRAIN_TGT=train-preproc.en
        VOCABULARY_SRC=vocab.vi
        VOCABULARY_TGT=vocab.en
        VALIDATION_SRC=tst2013.vi
        VALIDATION_TGT=tst2013.en
fi
# ==============================================================================
MODEL=
HPARAM_SETTINGS=

if [ "$2" == '--best' ]
then 
        echo "Running with best hyperparameter settings."
        MODEL=best
        HPARAM_SETTINGS="--device-ids=0 --embed-dropout=0.1:0.1 --rnn-decoder-hidden-dropout=0.2 --layer-normalization \
                         --num-layers=4:4 --max-seq-len=100:100 --label-smoothing 0.1 \
                         --weight-tying --weight-tying-type=src_trg --num-embed 512:512 --num-words 50000:50000 \
                         --word-min-count 1:1 --optimizer=adam  --initial-learning-rate=${INITIAL_LEARNING_RATE} \
                         --learning-rate-reduce-num-not-improved=8 --learning-rate-reduce-factor=0.7 \
                         --learning-rate-scheduler-type=plateau-reduce --max-num-checkpoint-not-improved=32 --min-num-epochs=0 --rnn-attention-type mlp \
                         --monitor-bleu=500 --keep-last-params=60 --lock-dir /var/lock --use-tensorboard"
        BATCH_SIZE=32
        CHECKPOINT_FREQUENCY=4000
else 
        echo "Running with groundhog hyperparameter settings."
        MODEL=groundhog
        HPARAM_SETTINGS="--bucket-width=10 --device-ids=0 --embed-dropout=0.3:0.3 \
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
                         --monitor-bleu=500 --keep-last-params=60 --lock-dir /var/lock --use-tensorboard"
        BATCH_SIZE=80
        CHECKPOINT_FREQUENCY=2000
fi

SOCKEYE_ROOT=$(cd $(dirname $0)/../.. && pwd)
PROJECT_ROOT=${SOCKEYE_ROOT}/../..
CONFERENCE_SRC_TGT_MODEL_OPT=${CONFERENCE_SRC_TGT}-${MODEL}-
CONFERENCE_SRC_TGT_MODEL_OPT=${CONFERENCE_SRC_TGT_MODEL_OPT}${USE_MLP_ATT_SCORING_FUNC}
CONFERENCE_SRC_TGT_MODEL_OPT=${CONFERENCE_SRC_TGT_MODEL_OPT}${USE_ECO_LSTM_CELL}
CONFERENCE_SRC_TGT_MODEL_OPT=${CONFERENCE_SRC_TGT_MODEL_OPT}${USE_LSTM_NONLIN_BLOCK}
CONFERENCE_SRC_TGT_MODEL_OPT=${CONFERENCE_SRC_TGT_MODEL_OPT}${USE_PAR_SEQUENCE_REVERSE}
# ==============================================================================
MAX_UPDATES=
NVPROF_PREFIX=

if [ "$3" == "--full-run" ]
then
        echo "Training will run until completion."
else
        echo "Training will stop early."
        MAX_UPDATES="--max-updates=500"
fi

if [ "$3" == "--nvprof-runtime" ]
then
	echo "nvprof is enabled to profile the runtime."
        mkdir -p ${SOCKEYE_ROOT}/workspace/profile/runtime
	NVPROF_PREFIX="/usr/local/cuda/bin/nvprof --profile-from-start off \
                       --csv --log-file ${SOCKEYE_ROOT}/workspace/profile/runtime/${CONFERENCE_SRC_TGT_MODEL_OPT}.csv"
        MAX_UPDATES="--max-updates=200"
fi
if [ "$3" == "--nvprof-dram-transactions" ]
then
	echo "nvprof is enabled to profile the DRAM transactions."
        mkdir -p ${SOCKEYE_ROOT}/workspace/profile/dram_transactions
	NVPROF_PREFIX="/usr/local/cuda/bin/nvprof --profile-from-start off \
                       --metrics dram_read_transactions,dram_write_transactions \
                       --csv --log-file ${SOCKEYE_ROOT}/workspace/profile/dram_transactions/${CONFERENCE_SRC_TGT_MODEL_OPT}.csv"
        MAX_UPDATES="--max-updates=200"
fi
# ==============================================================================
BATCH_SIZE=32

cd ${SOCKEYE_ROOT} && rm -rf ${SOCKEYE_ROOT}/workspace/${CONFERENCE_SRC_TGT_MODEL_OPT} && \
PYTHONPATH=${SOCKEYE_ROOT} ${NVPROF_PREFIX} \
python3 -m sockeye.train --source ${SOCKEYE_ROOT}/workspace/data/${CONFERENCE_SRC_TGT}/${TRAIN_SRC} \
                         --target ${SOCKEYE_ROOT}/workspace/data/${CONFERENCE_SRC_TGT}/${TRAIN_TGT} \
                         --source-vocab ${SOCKEYE_ROOT}/workspace/data/${CONFERENCE_SRC_TGT}/${VOCABULARY_SRC} \
                         --target-vocab ${SOCKEYE_ROOT}/workspace/data/${CONFERENCE_SRC_TGT}/${VOCABULARY_TGT} \
                         --validation-source ${SOCKEYE_ROOT}/workspace/data/${CONFERENCE_SRC_TGT}/${VALIDATION_SRC} \
                         --validation-target ${SOCKEYE_ROOT}/workspace/data/${CONFERENCE_SRC_TGT}/${VALIDATION_TGT} \
                         --output ${SOCKEYE_ROOT}/workspace/${CONFERENCE_SRC_TGT_MODEL_OPT} --seed=1 \
                         --batch-size=${BATCH_SIZE} --checkpoint-frequency=${CHECKPOINT_FREQUENCY} \
                         ${HPARAM_SETTINGS} ${MAX_UPDATES} \
                         2>&1 | tee /tmp/${CONFERENCE_SRC_TGT_MODEL_OPT}.log
