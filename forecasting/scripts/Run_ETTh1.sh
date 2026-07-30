#!/bin/bash
DATASET_FILENAME=ETTh1.csv
ROOT_PATH=./dataset/
DATA_NAME=ETTh1
MODEL_NAME=DLinear

SEQ_LEN=96
PRED_LEN=96
CHANNELS=7
C_OUT=7
CHANNEL_INDEPENDENCE=0
CYCLE_PARAM=24
FREQ=h
NUM_TRIALS=2

python3 -u ./run_longExp.py \
    --model $MODEL_NAME \
    --data $DATA_NAME \
    --root_path $ROOT_PATH \
    --data_path $DATASET_FILENAME \
    --seq_len $SEQ_LEN \
    --pred_len $PRED_LEN \
    --enc_in $CHANNELS \
    --c_out $C_OUT \
    --cycle $CYCLE_PARAM \
    --freq $FREQ \
    --patience 1 \
    --use_optuna True \
    --n_trials $NUM_TRIALS> logs/$MODEL_NAME'_'$SEQ_LEN'_'$PRED_LEN'_'$DATASET_FILENAME.log
