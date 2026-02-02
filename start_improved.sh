#!/bin/bash
# =============================================================================
# Start Improved Training (v2) for RFDiffusion
# =============================================================================
# Key improvements over v1:
# 1. Per-atom noise prediction (not broadcasting)
# 2. Cosine noise schedule (better than linear)
# 3. Transformer architecture with proper attention
# 4. Learning rate warmup
# 5. Normalized data
#
# Expected: Loss should drop from ~1.0 to <0.1
# =============================================================================

EPOCHS=${1:-1000}
RESUME_FLAG=""

if [ "$2" == "--resume" ]; then
    RESUME_FLAG="--resume"
    echo "Will resume from last checkpoint"
fi

# Configuration - Improved settings
GPUS="1,2,3"
NUM_GPUS=3
MASTER_PORT=29502
BATCH_SIZE=2           # Larger batch for stability
LR=0.0003              # 3e-4, higher LR with warmup
WARMUP_EPOCHS=50       # Warmup period
EMBED_DIM=256          # Larger model
NUM_LAYERS=8           # More layers
NUM_HEADS=8            # Multi-head attention
SCHEDULE="cosine"      # Cosine schedule

# Directories
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SAVE_DIR="${SCRIPT_DIR}/checkpoints_v2"
LOG_DIR="${SCRIPT_DIR}/logs_v2"
VENV_PYTHON="/home/chidwipak/Gsoc2026/venv/bin/python"

# Create directories
mkdir -p "$SAVE_DIR"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
NOHUP_LOG="${LOG_DIR}/nohup_v2_${TIMESTAMP}.log"

echo "=============================================================="
echo "  STARTING IMPROVED RFDIFFUSION TRAINING (v2)"
echo "=============================================================="
echo "  Timestamp: $(date)"
echo ""
echo "  Key Improvements:"
echo "    - Per-atom noise prediction"
echo "    - Cosine noise schedule"
echo "    - Transformer with attention"
echo "    - LR warmup (${WARMUP_EPOCHS} epochs)"
echo "    - Normalized data"
echo ""
echo "  Configuration:"
echo "    GPUs: $GPUS ($NUM_GPUS GPUs)"
echo "    Epochs: $EPOCHS"
echo "    Batch Size: $BATCH_SIZE per GPU (total: $((BATCH_SIZE * NUM_GPUS)))"
echo "    Learning Rate: $LR (with warmup)"
echo "    Model: ${EMBED_DIM}d, ${NUM_LAYERS} layers, ${NUM_HEADS} heads"
echo "    Schedule: $SCHEDULE"
echo ""
echo "  Output:"
echo "    Save Dir: $SAVE_DIR"
echo "    Log Dir: $LOG_DIR"
echo "    Nohup Log: $NOHUP_LOG"
echo "=============================================================="

# Kill any existing training processes
pkill -f "train_improved.py" 2>/dev/null
sleep 2

# Start training with nohup
cd "$SCRIPT_DIR"

nohup bash -c "CUDA_VISIBLE_DEVICES=$GPUS $VENV_PYTHON -m torch.distributed.launch \
    --nproc_per_node=$NUM_GPUS \
    --use_env \
    --master_port=$MASTER_PORT \
    train_improved.py \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --lr $LR \
    --warmup-epochs $WARMUP_EPOCHS \
    --embed-dim $EMBED_DIM \
    --num-layers $NUM_LAYERS \
    --num-heads $NUM_HEADS \
    --schedule $SCHEDULE \
    --save-dir $SAVE_DIR \
    --log-dir $LOG_DIR \
    $RESUME_FLAG" > "$NOHUP_LOG" 2>&1 &

TRAINING_PID=$!

echo ""
echo "  Training started with PID: $TRAINING_PID"
echo ""
echo "  Commands:"
echo "    Monitor: ./monitor_v2.sh"
echo "    Follow logs: tail -f $NOHUP_LOG"
echo "    Stop: pkill -f train_improved.py"
echo "=============================================================="

# Wait and check if started successfully
sleep 5

if ps -p $TRAINING_PID > /dev/null 2>&1; then
    echo ""
    echo "  ✅ Training is running successfully!"
    echo ""
    echo "  Initial output:"
    head -30 "$NOHUP_LOG" 2>/dev/null
else
    echo ""
    echo "  ❌ Training may have failed to start. Check logs:"
    cat "$NOHUP_LOG"
fi
