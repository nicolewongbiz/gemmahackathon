#!/usr/bin/env bash
# All EdgeHand nodes (real Gemma, vision, handover, pi_bridge) against fake_cell.py instead of MoveIt + hardware.
#   source /opt/ros/humble/setup.bash   (or any env with rclpy, moveit_msgs, control_msgs)
#   EDGEHAND_MODEL=/path/to/gemma-4-E4B-it.litertlm bash test/run_integration.sh
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:$PWD/test:${PYTHONPATH:-}"
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-77}
export EDGEHAND_CALIBRATION=none      # fake_cell draws config.py's tags, not the real cell's data/calibration.json
PY=${PYTHON:-python3}
LOG=$(mktemp -d)
pids=()
run() { $PY -c "from $1 import main; main()" >"$LOG/$2.log" 2>&1 & pids+=($!); }
trap 'kill ${pids[@]} 2>/dev/null; wait 2>/dev/null' EXIT

$PY test/fake_cell.py >"$LOG/fake_cell.log" 2>&1 & pids+=($!)
$PY -c "import sys; sys.argv += ['--ros-args', '-p', 'camera:=topic']; from edgehand.nodes.vision_node import main; main()" >"$LOG/vision.log" 2>&1 & pids+=($!)
run edgehand.nodes.handover_node handover
run edgehand.nodes.intent_node intent
$PY -c "import sys; sys.argv += ['--ros-args', '-p', 'require_pi:=false']; from edgehand.nodes.pi_bridge import main; main()" >"$LOG/pi_bridge.log" 2>&1 & pids+=($!)

# wait for Gemma to load
for _ in $(seq 120); do grep -q "Gemma ready" "$LOG/intent.log" 2>/dev/null && break; sleep 1; done
$PY test/integration_check.py
rc=$?
echo "node logs: $LOG"
exit $rc
