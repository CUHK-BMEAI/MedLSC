#!/bin/bash
export WANDB_MODE=disabled
export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"
source ./scripts/base/port_generator.sh
PORT=$(generate_random_port)
echo "Generated master port: $PORT"

DATASET_SPLIT=$1
DEVICES=$2
DATA_ROOT="./data/MedLSC"

EPOCH=3
if [[ "$DATASET_SPLIT" == "pathvqa" ]]; then
    TRAIN_DATA_PATH="./data/MedLSC/PathVQA/train.json"
    IMAGE_FOLDER="./data/MedLSC/PathVQA"
elif [[ "$DATASET_SPLIT" == "slake-vqarad" ]]; then
    TRAIN_DATA_PATH="./data/MedLSC/Slake-VQARad/train.json"
    IMAGE_FOLDER="./data/MedLSC/Slake-VQARad"
elif [[ "$DATASET_SPLIT" == "derm" ]]; then
    TRAIN_DATA_PATH="./data/MedLSC/Fitzpatrick/train.json"
    IMAGE_FOLDER="./data/MedLSC/Fitzpatrick"
elif [[ "$DATASET_SPLIT" == "CXP" ]]; then
    TRAIN_DATA_PATH="./data/MedLSC/CXP/train.json"
    IMAGE_FOLDER="./data/MedLSC/CXP"
    EPOCH=1
elif [[ "$DATASET_SPLIT" == "HAM" ]]; then
    TRAIN_DATA_PATH="./data/MedLSC/HAM/train.json"
    IMAGE_FOLDER="./data/MedLSC/HAM"
    EPOCH=1
elif [[ "$DATASET_SPLIT" == "PCAM" ]]; then
    TRAIN_DATA_PATH="./data/MedLSC/PCam/train.json"
    IMAGE_FOLDER="./data/MedLSC/PCam"
    EPOCH=1
elif [[ "$DATASET_SPLIT" == "iu-x-ray" ]]; then
    TRAIN_DATA_PATH="./data/MedLSC/IU-X-Ray/train.json"
    IMAGE_FOLDER="./data/MedLSC/IU-X-Ray"
    EPOCH=20
elif [[ "$DATASET_SPLIT" == "covid" ]]; then
    TRAIN_DATA_PATH="./data/MedLSC/COVID/train.json"
    IMAGE_FOLDER="./data/MedLSC/COVID"
    EPOCH=1
elif [[ "$DATASET_SPLIT" == "covid-CXP" ]]; then
    TRAIN_DATA_PATH="${DATA_ROOT}/COVID_CXP/train.json"
    IMAGE_FOLDER="${DATA_ROOT}"
    EPOCH=1
elif [[ "$DATASET_SPLIT" == "skin8" ]]; then
    TRAIN_DATA_PATH="./data/MedLSC/skin_8/train.json"
    IMAGE_FOLDER="./data/MedLSC/skin_8"
    EPOCH=1
elif [[ "$DATASET_SPLIT" == "HAM_skin8" ]]; then
    TRAIN_DATA_PATH="${DATA_ROOT}/HAM_skin8/train.json"
    IMAGE_FOLDER="${DATA_ROOT}"
    EPOCH=1
elif [[ "$DATASET_SPLIT" == "kvasir" ]]; then
    TRAIN_DATA_PATH="./data/MedLSC/Kvasir-VQA/train.json"
    IMAGE_FOLDER="./data/MedLSC/Kvasir-VQA"
elif [[ "$DATASET_SPLIT" == "radimagenet-vqa" ]]; then
    TRAIN_DATA_PATH="./data/MedLSC/RadImageNet-VQA/train.json"
    IMAGE_FOLDER="./data/MedLSC/RadImageNet-VQA"
elif [[ "$DATASET_SPLIT" == "Yangxi" ]]; then
    TRAIN_DATA_PATH="./data/MedLSC/Yangxi/train.json"
    IMAGE_FOLDER="./data/MedLSC/Yangxi"
elif [[ "$DATASET_SPLIT" == "oct-c8" ]]; then
    TRAIN_DATA_PATH="./data/MedLSC/Retinal_OCT_C8/train.json"
    IMAGE_FOLDER="./data/MedLSC/Retinal_OCT_C8"
elif [[ "$DATASET_SPLIT" == "cervical" ]]; then
    TRAIN_DATA_PATH="./data/MedLSC/cervical/train.json"
    IMAGE_FOLDER="./data/MedLSC/cervical"
elif [[ "$DATASET_SPLIT" == "hyperkvasir" ]]; then
    TRAIN_DATA_PATH="./data/MedLSC/HyperKvasir/train.json"
    IMAGE_FOLDER="./data/MedLSC/HyperKvasir"
    EPOCH=6
elif [[ "$DATASET_SPLIT" == "slake-ctxr" ]]; then
    TRAIN_DATA_PATH="${DATA_ROOT}/Slake-VQARad/train_ct_xray.json"
    IMAGE_FOLDER="${DATA_ROOT}/Slake-VQARad"
elif [[ "$DATASET_SPLIT" == "slake-mri" ]]; then
    TRAIN_DATA_PATH="${DATA_ROOT}/Slake-VQARad/train_mri.json"
    IMAGE_FOLDER="${DATA_ROOT}/Slake-VQARad"
fi

echo "========================================================"
echo "DATASET_SPLIT: $DATASET_SPLIT"
echo "TRAIN_DATA_PATH: $TRAIN_DATA_PATH"
echo "IMAGE_FOLDER: $IMAGE_FOLDER"
echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++"

#######################################################
# Important parameter about the version of training
#######################################################
TRAIN_VERSION="finetune_lora_each"
PRETRAINED_MODEL_PATH="./pretrained_models/llava_med_v1.5"
MODEL_NAME=$(basename "$PRETRAINED_MODEL_PATH")
#######################################################
# parameter about the hyper-parameters of training
#######################################################
LR=2e-4
BATCH_SIZE=16
GRADIENT_ACC_STEPS=1
LORA_RANK=64
LORA_ALPHA=64

MAX_TASK=1
#######################################################
# The following content does not need modification.
#######################################################
OUTPUT_MODEL_NAME="${TRAIN_VERSION}-${LORA_RANK}-${LORA_ALPHA}_${MODEL_NAME}/${DATASET_SPLIT}"

CUDA_VISIBLE_DEVICES=${DEVICES} \
python llava/train/train.py \
    --model_path $PRETRAINED_MODEL_PATH \
    --lora_enable True \
    --lora_rank $LORA_RANK \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout 0.0 \
    --max_task $MAX_TASK \
    --data_path $TRAIN_DATA_PATH \
    --image_folder $IMAGE_FOLDER \
    --bf16 True \
    --output_dir ./checkpoints/$OUTPUT_MODEL_NAME \
    --num_train_epochs $EPOCH \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --save_steps 100 \
    --save_total_limit 2 \
    --learning_rate $LR \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --report_to wandb


