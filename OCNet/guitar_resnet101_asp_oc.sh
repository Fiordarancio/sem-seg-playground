#!/bin/bash
# Simple modifications to apply this network to the guitars-augmented dataset. 
# NOTE: since by default they list 4 GPUS, it is likely that we encounter the
# same problems we had in semseg_mit training.
#-----------------------------------------------------------------------------------------------------------------------

# Check the enviroment info.
nvidia-smi
# pytorch 04
PYTHON="/home/ilaria/workspace/ocnenv/bin/python3.6"

# Load CUDA.
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.1/lib64
echo "Loaded CUDA lybrary: ${LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES="1"
echo "Cuda visible devices = ${CUDA_VISIBLE_DEVICES}"
echo "(device = 0 : using GeForce; device = 1 : using Quadro)"
GPUS=0,1

# Network configuration. 
NETWORK="resnet101"
METHOD="asp_oc_dsn"
TRAINDATASET="guitars_train"

# Training settings.
LEARNING_RATE=1e-2
WEIGHT_DECAY=5e-4
START_ITERS=0
MAX_ITERS=20000 # 20k-40k-80k
BATCHSIZE=8
INPUT_SIZE="960,720" # DEFINE A FIXED SIZE!
USE_CLASS_BALANCE=True
# # Apply OHEM criterion to boost performances (...but at which cost?)
# USE_OHEM=True
# OHEMTHRES=0.7
# OHEMKEEP=100000
# Don't apply OHEM criterion
USE_OHEM=False
OHEMTHRES=0.7 
OHEMKEEP=0
d
USE_VAL_SET=False
USE_EXTRA_SET=False

# Replace the DATA_DIR with your folder path to the dataset.
DATA_DIR="./dataset/guitars/"
DATA_LIST_PATH="./dataset/list/guitars/train.lst"
RESTORE_FROM="./pretrained_model/resnet101-imagenet.pth"

# Set directories
TRAIN_LOGDIR="log/log_train"
EVAL_LOGDIR="log/log_eval"
TEST_LOGDIR="log/log_test"
mkdir -p ${TRAIN_LOGDIR}
mkdir -p ${EVAL_LOGDIR}
mkdir -p ${TEST_LOGDIR}

# Set the output path of checkpoints, training log.
TRAIN_LOG_FILE="./${TRAIN_LOGDIR}/log_${NETWORK}_${METHOD}_${LEARNING_RATE}_${WEIGHT_DECAY}_${BATCHSIZE}_${MAX_ITERS}.txt"
SNAPSHOT_DIR="./checkpoint/snapshots_${NETWORK}_${METHOD}_${LEARNING_RATE}_${WEIGHT_DECAY}_${BATCHSIZE}_${MAX_ITERS}/"
touch TRAIN_LOG_FILE

#-----------------------------------------------------------------------------------------------------------------------
#  Training
#-----------------------------------------------------------------------------------------------------------------------
echo "----------------------------------------------------------------------------"
echo "Training started on $(date)"
$PYTHON -u train.py \
  --network $NETWORK \
  --method $METHOD \
  --random-mirror \
  --random-scale \
  --gpu $GPUS \
  --batch-size $BATCHSIZE \
  --snapshot-dir $SNAPSHOT_DIR \
  --num-steps $MAX_ITERS \
  --ohem $USE_OHEM \
  --data-list $DATA_LIST_PATH \
  --weight-decay $WEIGHT_DECAY \
  --input-size $INPUT_SIZE \
  --ohem-thres $OHEMTHRES \
  --ohem-keep $OHEMKEEP \
  --use-val $USE_VAL_SET \
  --use-weight $USE_CLASS_BALANCE \
  --snapshot-dir $SNAPSHOT_DIR \
  --restore-from $RESTORE_FROM \
  --start-iters $START_ITERS \
  --learning-rate $LEARNING_RATE  \
  --use-extra $USE_EXTRA_SET \
  --dataset $TRAINDATASET \
  --data-dir $DATA_DIR \
  # > $TRAIN_LOG_FILE 2>&1

#-----------------------------------------------------------------------------------------------------------------------
# Evaluation settings.
EVAL_USE_FLIP=False
EVAL_USE_MS=False
EVAL_STORE_RESULT=False
EVAL_BATCHSIZE=4
PREDICT_CHOICE="whole"
WHOLE_SCALE="1"
EVAL_RESTORE_FROM="${SNAPSHOT_DIR}CS_scenes_${MAX_ITERS}.pth"

#-----------------------------------------------------------------------------------------------------------------------
#  Evaluation
#-----------------------------------------------------------------------------------------------------------------------
# on the validation set
EVALDATASET="guitars_eval"
EVAL_SET="eval"
EVAL_DATA_LIST_PATH="./dataset/list/cityscapes/eval.lst"
EVAL_LOG_FILE="./${EVAL_LOGDIR}/log_result_${NETWORK}_${METHOD}_${EVAL_SET}_${LEARNING_RATE}_${WEIGHT_DECAY}_${BATCHSIZE}_${MAX_ITERS}_${PREDICT_CHOICE}"
EVAL_OUTPUT_PATH="./visualize/${NETWORK}_${METHOD}_${EVAL_SET}_${LEARNING_RATE}_${WEIGHT_DECAY}_${BATCHSIZE}_${MAX_ITERS}/"

echo "----------------------------------------------------------------------------"
echo "Evaluation on ${EVALDATASET} started on $(date)"
$PYTHON -u eval.py \
  --network=$NETWORK \
  --method=$METHOD \
  --batch-size=$EVAL_BATCHSIZE \
  --data-list $EVAL_DATA_LIST_PATH \
  --dataset $EVALDATASET \
  --restore-from=$EVAL_RESTORE_FROM \
  --store-output=$EVAL_STORE_RESULT \
  --output-path=$EVAL_OUTPUT_PATH \
  --input-size $INPUT_SIZE \
  --use-flip=$EVAL_USE_FLIP  \
  --use-ms=$EVAL_USE_MS \
  --gpu $GPUS \
  --predict-choice $PREDICT_CHOICE \
  --whole-scale ${WHOLE_SCALE} \
  # > $EVAL_LOG_FILE 2>&1

# on the training set
EVALDATASET="guitars_train"
EVAL_SET="train"
EVAL_DATA_LIST_PATH="./dataset/list/cityscapes/train.lst"
EVAL_LOG_FILE="./${EVAL_LOGDIR}/log_result_${NETWORK}_${METHOD}_${EVAL_SET}_${LEARNING_RATE}_${WEIGHT_DECAY}_${BATCHSIZE}_${MAX_ITERS}_${PREDICT_CHOICE}"
EVAL_OUTPUT_PATH="./visualize/${NETWORK}_${METHOD}_${EVAL_SET}_${LEARNING_RATE}_${WEIGHT_DECAY}_${BATCHSIZE}_${MAX_ITERS}/"

echo "----------------------------------------------------------------------------"
echo "Evaluation on ${EVALDATASET} started on $(date)"
$PYTHON -u eval.py \
  --network=$NETWORK \
  --method=$METHOD \
  --batch-size=$EVAL_BATCHSIZE \
  --data-list $EVAL_DATA_LIST_PATH \
  --dataset $EVALDATASET \
  --restore-from=$EVAL_RESTORE_FROM \
  --store-output=$EVAL_STORE_RESULT \
  --output-path=$EVAL_OUTPUT_PATH \
  --input-size $INPUT_SIZE \
  --use-flip=$EVAL_USE_FLIP  \
  --use-ms=$EVAL_USE_MS \
  --gpu $GPUS \
  --predict-choice $PREDICT_CHOICE \
  --whole-scale ${WHOLE_SCALE} \
  # > $EVAL_LOG_FILE 2>&1


#-----------------------------------------------------------------------------------------------------------------------
# Test settings.
TEST_STORE_RESULT=True
TESTDATASET="guitars_vis" 
TEST_SET="vis" 
TEST_DATA_LIST_PATH="./dataset/list/cityscapes/vis.lst"
TEST_LOG_FILE="./${TEST_LOGDIR}/log_result_${NETWORK}_${METHOD}_${TEST_SET}_${LEARNING_RATE}_${WEIGHT_DECAY}_${BATCHSIZE}_${MAX_ITERS}_${PREDICT_CHOICE}"
TEST_OUTPUT_PATH="./visualize/${NETWORK}_${METHOD}_${TEST_SET}_${LEARNING_RATE}_${WEIGHT_DECAY}_${BATCHSIZE}_${MAX_ITERS}/"

#-----------------------------------------------------------------------------------------------------------------------
# Testing
#-----------------------------------------------------------------------------------------------------------------------
$PYTHON -u generate_submit.py \
  --network=$NETWORK \
  --method=$METHOD \
  --batch-size=$TEST_BATCHSIZE \
  --data-list $TEST_DATA_LIST_PATH \
  --dataset $TESTDATASET \
  --restore-from=$TEST_RESTORE_FROM \
  --store-output=$TEST_STORE_RESULT \
  --output-path=$TEST_OUTPUT_PATH \
  --input-size $INPUT_SIZE \
  --use-flip=$TEST_USE_FLIP \
  --use-ms=$TEST_USE_MS \
  --gpu 0,1,2,3 \
  --predict-choice $PREDICT_CHOICE \
  --whole-scale ${WHOLE_SCALE} \
  # > $TEST_LOG_FILE 2>&1

echo "-------------------------------"
echo "OCRNet got all steps completed."
