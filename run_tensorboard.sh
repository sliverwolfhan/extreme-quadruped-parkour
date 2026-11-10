#!/usr/bin/env bash
echo "Ensuring TensorBoard is running on port 6006..."
docker exec -d -e TERM=xterm robot-lab /workspace/isaaclab/isaaclab.sh -p -m tensorboard.main \
  --logdir /workspace/logs/rsl_rl/atdog2_meihua_direct_abl3_5 \
  --port 6006 --bind_all 2>/dev/null || true
echo "=========================================================="
echo "  TensorBoard is running!"
echo "  Open in your browser: http://localhost:6006"
echo "=========================================================="
