#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../.." && pwd)
runtime_dir="${XDG_RUNTIME_DIR:-/tmp}/lbot-local-arm-can"
local_ros_domain_id="${LBOT_LOCAL_ROS_DOMAIN_ID:-121}"
mkdir -p "$runtime_dir"

if curl -fsS --max-time 1 http://127.0.0.1:8767/state >/dev/null 2>&1 \
  && curl -fsS --max-time 1 http://127.0.0.1:8766/state >/dev/null 2>&1; then
  left_pid=$(pgrep -f '^python3 -u -m tools\.local_arm_can\.service serve --interface can0 --arm left ' | head -n 1 || true)
  right_pid=$(pgrep -f '^python3 -u -m tools\.local_arm_can\.service serve --interface can1 --arm right ' | head -n 1 || true)
  ros_pid=$(pgrep -f 'python3 -u tools/local_arm_can/ros_node\.py --interface can1$' | head -n 1 || true)
  [[ -n "$left_pid" ]] && echo "$left_pid" >"$runtime_dir/left-service.pid"
  [[ -n "$right_pid" ]] && echo "$right_pid" >"$runtime_dir/right-service.pid"
  [[ -n "$ros_pid" ]] && echo "$ros_pid" >"$runtime_dir/right-ros.pid"
  echo "local arm CAN services are already running"
  exec "$script_dir/status.sh"
fi

if [[ ! -f "$script_dir/ros_ws/install/setup.bash" ]]; then
  "$script_dir/build_ros_interfaces.sh"
fi

pkexec /usr/bin/python3 "$script_dir/service.py" configure can0 --allow-can0
pkexec /usr/bin/python3 "$script_dir/service.py" configure can1

cd "$repo_root"
nohup /usr/bin/python3 -u -m tools.local_arm_can.service serve \
  --interface can0 --arm left --allow-can0 --allow-tx --poll-telemetry \
  --bind 127.0.0.1 --port 8767 \
  >"$runtime_dir/left-service.log" 2>&1 &
left_pid=$!
echo "$left_pid" >"$runtime_dir/left-service.pid"

nohup /usr/bin/python3 -u -m tools.local_arm_can.service serve \
  --interface can1 --arm right --allow-tx --poll-telemetry \
  --bind 127.0.0.1 --port 8766 \
  >"$runtime_dir/right-service.log" 2>&1 &
right_pid=$!
echo "$right_pid" >"$runtime_dir/right-service.pid"

nohup bash -c "
  source /opt/ros/jazzy/setup.bash
  source '$script_dir/ros_ws/install/setup.bash'
  export ROS_DOMAIN_ID='$local_ros_domain_id'
  export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
  exec /usr/bin/python3 -u '$script_dir/ros_node.py' --interface can1
" >"$runtime_dir/right-ros.log" 2>&1 &
ros_pid=$!
echo "$ros_pid" >"$runtime_dir/right-ros.pid"

cleanup_start_failure() {
  kill "$left_pid" "$right_pid" "$ros_pid" 2>/dev/null || true
}
trap cleanup_start_failure ERR

for _attempt in {1..30}; do
  if curl -fsS --max-time 1 http://127.0.0.1:8767/state >/dev/null 2>&1 \
    && curl -fsS --max-time 1 http://127.0.0.1:8766/state >/dev/null 2>&1; then
    trap - ERR
    "$script_dir/status.sh"
    exit 0
  fi
  sleep 0.2
done

echo "services failed to become ready; see $runtime_dir/*.log" >&2
exit 1
