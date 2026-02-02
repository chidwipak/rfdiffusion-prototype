#!/bin/bash
# =============================================================================
# Training Monitor Script for RFDiffusion
# =============================================================================
# This script monitors the training progress, shows recent logs, 
# and displays GPU utilization.
#
# Usage: ./monitor_training.sh
# =============================================================================

echo "=============================================================="
echo "  RFDIFFUSION TRAINING MONITOR"
echo "  $(date)"
echo "=============================================================="

# Check if training is running
TRAINING_PIDS=$(pgrep -f "train_production.py" 2>/dev/null)
if [ -z "$TRAINING_PIDS" ]; then
    echo ""
    echo "[STATUS] ❌ No training process detected"
    echo ""
else
    echo ""
    echo "[STATUS] ✅ Training is running (PIDs: $TRAINING_PIDS)"
    echo ""
fi

# GPU Status
echo "--------------------------------------------------------------"
echo "  GPU STATUS"
echo "--------------------------------------------------------------"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits | \
while IFS=, read -r idx name mem_used mem_total util temp; do
    mem_pct=$((100 * mem_used / mem_total))
    echo "  GPU $idx: ${util}% util | ${mem_used}/${mem_total} MB (${mem_pct}%) | ${temp}°C"
done

# Latest metrics
echo ""
echo "--------------------------------------------------------------"
echo "  TRAINING METRICS"
echo "--------------------------------------------------------------"
METRICS_FILE="./checkpoints/metrics.json"
if [ -f "$METRICS_FILE" ]; then
    # Get last 5 losses
    echo "  Recent losses (last 5 epochs):"
    python3 -c "
import json
with open('$METRICS_FILE', 'r') as f:
    m = json.load(f)
losses = m.get('train_loss', [])
epochs = list(range(len(losses) - 4, len(losses) + 1)) if len(losses) >= 5 else list(range(1, len(losses) + 1))
for i, (e, l) in enumerate(zip(epochs[-5:], losses[-5:])):
    print(f'    Epoch {e:4d}: {l:.6f}')
print(f\"  Best: {m.get('best_loss', 'N/A'):.6f} (Epoch {m.get('best_epoch', 'N/A')})\")
print(f\"  Total Epochs: {m.get('total_epochs', 0)}\")
" 2>/dev/null || echo "  [Unable to parse metrics]"
else
    echo "  [No metrics file found yet]"
fi

# Latest log entries
echo ""
echo "--------------------------------------------------------------"
echo "  RECENT LOG ENTRIES (last 10 lines)"
echo "--------------------------------------------------------------"
LATEST_LOG=$(ls -t ./logs/training_*.log 2>/dev/null | head -1)
if [ -n "$LATEST_LOG" ]; then
    echo "  Log file: $LATEST_LOG"
    echo ""
    tail -10 "$LATEST_LOG" | sed 's/^/  /'
else
    echo "  [No log files found]"
fi

# Checkpoint status
echo ""
echo "--------------------------------------------------------------"
echo "  CHECKPOINTS"
echo "--------------------------------------------------------------"
if [ -d "./checkpoints" ]; then
    ls -lh ./checkpoints/*.pt 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}' || echo "  [No checkpoints yet]"
else
    echo "  [Checkpoint directory not found]"
fi

echo ""
echo "=============================================================="
echo "  To follow live logs: tail -f $LATEST_LOG"
echo "  To stop training: pkill -f train_production.py"
echo "=============================================================="
