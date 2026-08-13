#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${PROJECT_DIR}/.venv"
DRIVER_DIR="${PROJECT_DIR}/vendor/pyorbbecsdk/install/lib"
PORT="${PORT:-8080}"

cd "${PROJECT_DIR}"

if [[ ! -x "${VENV_DIR}/bin/python" || ! -f "${DRIVER_DIR}/libOrbbecSDK.so" ]]; then
  echo "首次运行，正在安装项目依赖并编译 Orbbec 驱动…"
  "${PROJECT_DIR}/scripts/install.sh"
fi

export ORBBEC_DRIVER_DIR="${DRIVER_DIR}"
export LD_LIBRARY_PATH="${DRIVER_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONUNBUFFERED=1

echo "Gemini Pro 视觉台已启动："
echo "  本机访问  http://127.0.0.1:${PORT}"
echo "  局域网访问 http://$(hostname -I | awk '{print $1}'):${PORT}"
echo "按 Ctrl+C 停止服务。"

exec "${VENV_DIR}/bin/gunicorn" \
  --bind "0.0.0.0:${PORT}" \
  --workers 1 \
  --threads 8 \
  --timeout 0 \
  --access-logfile - \
  --error-logfile - \
  "server:app"
