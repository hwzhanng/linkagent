#!/usr/bin/env bash
set -euo pipefail

workspace_dir="$(cd -- "$(dirname -- "$0")/.." && pwd)"
ros_install="$workspace_dir/ros2_official/install"
driver="$ros_install/linker_hand_ros2_sdk/lib/linker_hand_ros2_sdk/linker_hand_sdk"
gui="$ros_install/gui_control/lib/gui_control/gui_control"
driver_pid=""

set +u
source /opt/ros/jazzy/setup.bash
source "$ros_install/setup.bash"
set -u

if ! ip link show can0 >/dev/null 2>&1 || ! ip link show can0 | rg -q "UP"; then
    echo "can0 is missing or not UP; run the CAN preflight first." >&2
    exit 1
fi

if pgrep -f "[l]inker_hand_sdk|[g]ui_control" >/dev/null; then
    echo "A LinkerHand ROS driver or official GUI is already running." >&2
    exit 1
fi

cleanup() {
    if [[ -n "$driver_pid" ]] && kill -0 "$driver_pid" 2>/dev/null; then
        kill -INT "$driver_pid" 2>/dev/null || true
        wait "$driver_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

"$driver" --ros-args \
    -p hand_type:=left \
    -p hand_joint:=G20 \
    -p is_touch:=true \
    -p can:=can0 \
    -p modbus:=None &
driver_pid="$!"

ready=false
for _ in $(seq 1 50); do
    if ros2 node list 2>/dev/null | rg -q "^/linker_hand_sdk$"; then
        ready=true
        break
    fi
    if ! kill -0 "$driver_pid" 2>/dev/null; then
        wait "$driver_pid"
    fi
    sleep 0.1
done
if [[ "$ready" != true ]]; then
    echo "G20 driver did not become ready." >&2
    exit 1
fi

ros2 topic pub --once /cb_hand_setting_cmd std_msgs/msg/String \
    "data: '{\"setting_cmd\":\"set_speed\",\"params\":{\"hand_type\":\"left\",\"speed\":[80,80,80,80,80]}}'"

"$gui" --ros-args \
    -p hand_type:=left \
    -p hand_joint:=G20 \
    -p topic_hz:=30 \
    -p is_touch:=true \
    -p is_arc:=false
