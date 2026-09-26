DATASET_SPLITS=("covid-CXP" "slake-ctxr" "iu-x-ray" "slake-mri" "PCAM" "pathvqa" "HAM_skin8" "derm" "Yangxi" "oct-c8" "cervical" "kvasir" "hyperkvasir")
DEVICES=0
for dataset in "${DATASET_SPLITS[@]}"; do
    bash ./scripts/finetune/main_each.sh "$dataset" "$DEVICES"
done
