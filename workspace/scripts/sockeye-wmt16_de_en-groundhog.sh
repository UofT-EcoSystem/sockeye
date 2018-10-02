#!/bin/bash

# This file runs the sockeye toolkit and trains on the WMT16 English-German dataset using the groundhog model.

SOCKEYE_ROOT=$(cd $(dirname $0)/../.. && pwd)
PROJECT_ROOT=${SOCKEYE_ROOT}/../..
CONFERENCE_SRC_TGT=wmt16-de_en

# Activate the virtual environment with MXNet and MXBoard.
# cd ${PROJECT_ROOT}/virtualenv/bin && source activate

cd ${SOCKEYE_ROOT} && rm -rf ${SOCKEYE_ROOT}/workspace/${CONFERENCE_SRC_TGT} && \
PYTHONPATH=${SOCKEYE_ROOT} python3 -m sockeye.train --source ${SOCKEYE_ROOT}/workspace/data/${CONFERENCE_SRC_TGT}/train.tok.clean.bpe.32000.de \
                                                    --target ${SOCKEYE_ROOT}/workspace/data/${CONFERENCE_SRC_TGT}/train.tok.clean.bpe.32000.en \
                                              --source-vocab ${SOCKEYE_ROOT}/workspace/data/${CONFERENCE_SRC_TGT}/vocab.bpe.32000.de \
                                              --target-vocab ${SOCKEYE_ROOT}/workspace/data/${CONFERENCE_SRC_TGT}/vocab.bpe.32000.en \
                                         --validation-source ${SOCKEYE_ROOT}/workspace/data/${CONFERENCE_SRC_TGT}/newstest2013.tok.bpe.32000.de \
                                         --validation-target ${SOCKEYE_ROOT}/workspace/data/${CONFERENCE_SRC_TGT}/newstest2013.tok.bpe.32000.en \
--output ${SOCKEYE_ROOT}/workspace/${CONFERENCE_SRC_TGT} --seed=1 --batch-type=sentence --batch-size=80 --bucket-width=10 --checkpoint-frequency=2000 --device-ids=0 --embed-dropout=0.3:0.3 --encoder=rnn --decoder=rnn --num-layers=1:1 --rnn-cell-type=lstm --rnn-num-hidden=1000 --rnn-residual-connections --layer-normalization --rnn-attention-type=mlp --rnn-attention-num-hidden=512 --rnn-attention-use-prev-word --rnn-attention-in-upper-layers --rnn-attention-coverage-num-hidden=1 --rnn-attention-coverage-type=count --rnn-decoder-state-init=zero --rnn-attention-use-prev-word --rnn-dropout-inputs=0:0 --rnn-dropout-states=0.0:0.0 --rnn-dropout-recurrent=0.0:0.0 --rnn-decoder-hidden-dropout=0.3 --fill-up=replicate --max-seq-len=50:50 --loss=cross-entropy --num-embed 500:500 --num-words 50000:50000 --word-min-count 1:1 --optimizer=adam --optimized-metric=perplexity --clip-gradient=1.0 --initial-learning-rate=0.0002 --learning-rate-reduce-num-not-improved=8 --learning-rate-reduce-factor=0.7 --learning-rate-scheduler-type=plateau-reduce --learning-rate-warmup=0 --max-num-checkpoint-not-improved=16 --min-num-epochs=1 --monitor-bleu=500 --keep-last-params=60 --lock-dir /var/lock --use-tensorboard
