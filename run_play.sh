#!/usr/bin/env bash
set -e

# Grant local X11 access to Docker container
xhost +local:root 2>/dev/null || true

CHECKPOINT="/workspace/logs/rsl_rl/atdog2_meihua_direct_abl3_5/2026-11-10_18-59-28/model_3000.pt"

echo "=========================================================="
echo "  Launching ATDog2 Plum Blossom (梅花桩) 3D Simulation Play"
echo "  Checkpoint: ${CHECKPOINT}"
echo "=========================================================="

docker exec -it -e DISPLAY="${DISPLAY:-:0}" -e TERM="${TERM:-xterm}" robot-lab /workspace/isaaclab/isaaclab.sh -p \
  /workspace/isaaclab_extension_template/scripts/reinforcement_learning/rsl_rl/play.py \
  --task RobotLab-Isaac-Direct-Meihua-ATDog-Dog2-Play-v0 \
  --num_envs 4 \
  --checkpoint "${CHECKPOINT}" "$@"
