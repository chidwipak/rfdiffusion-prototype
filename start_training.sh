#!/bin/bash
# =============================================================================
# Start Production Training for RFDiffusion
# =============================================================================
# This script starts the training with nohup so it survives SSH disconnection.
#
# Usage: ./start_training.sh [epochs] [--resume]
# Example: ./start_training.sh 500
# Example: ./start_training.sh 500 --resume
# =============================================================================

EPOCHS=${1:-500}
RESUME_FLAG=""

if [ "$2" == "--resume" ]; then
    RESUME_FLAG="--resume"
    echo "Will resume from last checkpoint"
fi

# Configuration
GPUS="1,2,3"
NUM_GPUS=3
MASTER_PORT=29501
BATCH_SIZE=1
ACCUM_STEPS=4
LR=0.0001
LAYERS=6
EMBED_DIM=128

# Directories
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SAVE_DIR="${SCRIPT_DIR}/checkpoints"
LOG_DIR="${SCRIPT_DIR}/logs"
VENV_PYTHON="/home/chidwipak/Gsoc2026/venv/bin/python"

# Create directories
mkdir -p "$SAVE_DIR"
mkdir -p "$LOG_DIR"

# Generate log filename with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
NOHUP_LOG="${LOG_DIR}/nohup_${TIMESTAMP}.log"

echo "=============================================================="
echo "  STARTING RFDIFFUSION PRODUCTION TRAINING"
echo "=============================================================="
echo "  Timestamp: $(date)"
echo "  GPUs: $GPUS ($NUM_GPUS GPUs)"
echo "  Epochs: $EPOCHS"
echo "  Batch Size: $BATCH_SIZE per GPU"
echo "  Effective Batch: $((BATCH_SIZE * NUM_GPUS * ACCUM_STEPS))"
echo "  Learning Rate: $LR"
echo "  Layers: $LAYERS"
echo "  Embed Dim: $EMBED_DIM"
echo "  Save Dir: $SAVE_DIR"
echo "  Log Dir: $LOG_DIR"
echo "  Nohup Log: $NOHUP_LOG"
echo "=============================================================="

# Kill any existing training processes
pkill -f "train_production.py" 2>/dev/null
sleep 2

# Start training with nohup
cd "$SCRIPT_DIR"

nohup bash -c "CUDA_VISIBLE_DEVICES=$GPUS $VENV_PYTHON -m torch.distributed.launch \
    --nproc_per_node=$NUM_GPUS \
    --use_env \
    --master_port=$MASTER_PORT \
    train_production.py \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --accum-steps $ACCUM_STEPS \
    --lr $LR \
    --layers $LAYERS \
    --embed-dim $EMBED_DIM \
    --save-dir $SAVE_DIR \
    --log-dir $LOG_DIR \
    $RESUME_FLAG" > "$NOHUP_LOG" 2>&1 &

TRAINING_PID=$!

echo ""
echo "  Training started with PID: $TRAINING_PID"
echo ""
echo "  To monitor: ./monitor_training.sh"
echo "  To follow logs: tail -f $NOHUP_LOG"
echo "  To stop: pkill -f train_production.py"
echo "=============================================================="

# Wait a moment and check if it started successfully
sleep 5

if ps -p $TRAINING_PID > /dev/null 2>&1; then
    echo ""
    echo "  ✅ Training is running successfully!"
    echo ""
    echo "  Initial output:"
    head -20 "$NOHUP_LOG" 2>/dev/null
else
    echo ""
    echo "  ❌ Training may have failed to start. Check logs:"
    cat "$NOHUP_LOG"
fi
