#!/usr/bin/env bash
set -eo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
workspace="$script_dir/ros_ws"

source /opt/ros/jazzy/setup.bash
set -u
cd "$workspace"
colcon --log-base log build \
  --base-paths src \
  --packages-select lbot_arm_interfaces \
  --build-base build \
  --install-base install \
  --cmake-clean-cache \
  --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3 \
  --event-handlers console_direct+
