#!/bin/bash
# Monitor script for improved training (v2)

echo "=============================================================="
echo "  RFDIFFUSION IMPROVED TRAINING MONITOR (v2)"
echo "  $(date)"
echo "=============================================================="

# Check if training is running
TRAINING_PIDS=$(pgrep -f "train_improved.py" 2>/dev/null)
if [ -z "$TRAINING_PIDS" ]; then
    echo ""
    echo "[STATUS] ❌ No training process detected"
else
    echo ""
    echo "[STATUS] ✅ Training is running"
fi

# GPU Status
echo ""
echo "--------------------------------------------------------------"
echo "  GPU STATUS"
echo "--------------------------------------------------------------"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits | \
while IFS=, read -r idx mem_used mem_total util temp; do
    mem_pct=$((100 * mem_used / mem_total))
    echo "  GPU $idx: ${util}% util | ${mem_used}/${mem_total} MB (${mem_pct}%) | ${temp}°C"
done

# Latest metrics
echo ""
echo "--------------------------------------------------------------"
echo "  TRAINING METRICS (v2)"
echo "--------------------------------------------------------------"
METRICS_FILE="./checkpoints_v2/metrics_v2.json"
if [ -f "$METRICS_FILE" ]; then
    python3 -c "
import json
with open('$METRICS_FILE', 'r') as f:
    m = json.load(f)

losses = m.get('train_loss', [])
if len(losses) == 0:
    print('  [No data yet]')
else:
    print('  Recent losses (last 5 epochs):')
    for i, loss in enumerate(losses[-5:]):
        epoch = len(losses) - 5 + i + 1
        if epoch > 0:
            print(f'    Epoch {epoch:4d}: {loss:.6f}')
    
    print(f\"  Best: {m.get('best_loss', 'N/A'):.6f} (Epoch {m.get('best_epoch', 'N/A')})\")
    print(f\"  Total Epochs: {m.get('total_epochs', 0)}\")
    
    # Show improvement
    if len(losses) > 10:
        first_10_avg = sum(losses[:10]) / 10
        last_10_avg = sum(losses[-10:]) / 10
        improvement = (first_10_avg - last_10_avg) / first_10_avg * 100
        print(f'  Improvement: {improvement:.1f}% (first 10 vs last 10 epochs)')
" 2>/dev/null || echo "  [Unable to parse metrics]"
else
    echo "  [No metrics file found yet]"
fi

# Latest log entries
echo ""
echo "--------------------------------------------------------------"
echo "  RECENT LOG ENTRIES (last 10 lines)"
echo "--------------------------------------------------------------"
LATEST_LOG=$(ls -t ./logs_v2/training_v2_*.log 2>/dev/null | head -1)
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
echo "  CHECKPOINTS (v2)"
echo "--------------------------------------------------------------"
if [ -d "./checkpoints_v2" ]; then
    ls -lh ./checkpoints_v2/*.pt 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}' | head -10 || echo "  [No checkpoints yet]"
else
    echo "  [Checkpoint directory not found]"
fi

echo ""
echo "=============================================================="
echo "  To follow live logs: tail -f $LATEST_LOG"
echo "  To stop training: pkill -f train_improved.py"
echo "=============================================================="
