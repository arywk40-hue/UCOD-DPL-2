#!/bin/bash
# Train PURE baseline (no APM, no agents) for 20 epochs, then evaluate on all 4 datasets

echo "=========================================="
echo "STEP 1: Training PURE baseline - 20 epochs"
echo "  (no APM, no agents, fixed pseudo-labels)"
echo "=========================================="

python train.py --config configs/uscod/UCOD-Pure_dinov2.py --opts train_cfg.max_epoch 20

echo ""
echo "=========================================="
echo "STEP 2: Check saved checkpoints"
echo "=========================================="

ls -lh work_dir/uscod/UCOD-Pure_dinov2/UCOD-Pure/ckp/

echo ""
echo "=========================================="
echo "STEP 3: Evaluate on all 4 datasets"
echo "=========================================="

# Find latest checkpoint
CKP_DIR="work_dir/uscod/UCOD-Pure_dinov2/UCOD-Pure/ckp"
CHECKPOINT=$(ls -t "$CKP_DIR"/*.pth 2>/dev/null | head -1)

if [ -z "$CHECKPOINT" ]; then
    echo "ERROR: No .pth checkpoint found in $CKP_DIR"
    exit 1
fi

echo "Using checkpoint: $CHECKPOINT"
echo ""

DATASETS=("CHAMELEON" "TE-CAMO" "TE-COD10K" "NC4K")

for DATASET in "${DATASETS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Evaluating on: $DATASET"
    echo "=========================================="
    
    python eval.py \
        --config configs/uscod/UCOD-Pure_dinov2.py \
        --load_from "$CHECKPOINT" \
        --opts dataset_cfg.valset_cfg.DATASET "$DATASET"
    
    if [ $? -ne 0 ]; then
        echo "ERROR: Evaluation failed for $DATASET"
    else
        echo "SUCCESS: Evaluation completed for $DATASET"
    fi
done

echo ""
echo "=========================================="
echo "ALL DONE — Compare with ADF and APM results"
echo "=========================================="
