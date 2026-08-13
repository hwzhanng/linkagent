#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DRIVER_DIR="${PROJECT_DIR}/vendor/pyorbbecsdk/install/lib"

echo "USB 设备"
lsusb | grep -E '2bc5:(0511|0614)' || true
echo
echo "视频节点"
for node in /sys/class/video4linux/video*; do
  [[ -e "${node}" ]] || continue
  printf '%s  %s\n' "/dev/$(basename "${node}")" "$(cat "${node}/name")"
done
echo
echo "Orbbec Python 驱动"
if [[ -f "${DRIVER_DIR}/pyorbbecsdk.cpython-311-x86_64-linux-gnu.so" ]]; then
  echo "已编译：${DRIVER_DIR}"
else
  echo "未编译，请运行 ./scripts/install.sh"
fi
