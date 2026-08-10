#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${PROJECT_DIR}/.venv"
SDK_SOURCE="${PROJECT_DIR}/vendor/pyorbbecsdk"
SDK_TAG="v1.3.4"

cd "${PROJECT_DIR}"

if command -v uv >/dev/null 2>&1; then
  uv venv --python 3.11 "${VENV_DIR}"
  uv pip install --python "${VENV_DIR}/bin/python" -r requirements.txt
else
  PYTHON311="$(command -v python3.11 || true)"
  if [[ -z "${PYTHON311}" ]]; then
    echo "需要 Python 3.11。请先安装 Python 3.11，或安装 uv 后重试。" >&2
    exit 1
  fi
  "${PYTHON311}" -m venv "${VENV_DIR}"
  "${VENV_DIR}/bin/pip" install -r requirements.txt
fi

mkdir -p "${PROJECT_DIR}/vendor"
if [[ ! -f "${SDK_SOURCE}/CMakeLists.txt" ]]; then
  git clone --branch "${SDK_TAG}" --depth 1 \
    https://github.com/orbbec/pyorbbecsdk.git "${SDK_SOURCE}"
fi

cmake -S "${SDK_SOURCE}" -B "${SDK_SOURCE}/build" \
  -DPython3_EXECUTABLE="${VENV_DIR}/bin/python" \
  -Dpybind11_DIR="${VENV_DIR}/lib/python3.11/site-packages/pybind11/share/cmake/pybind11" \
  -DBUILD_TESTING=OFF
cmake --build "${SDK_SOURCE}/build" --parallel
cmake --install "${SDK_SOURCE}/build"

echo
echo "安装完成。运行：./start.sh"
