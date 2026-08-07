#!/usr/bin/env bash

# Source this file; executing it cannot activate the caller's shell.
_linkerhand_workspace="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
_linkerhand_env="${_linkerhand_workspace}/.conda/envs/linkerhand"
_linkerhand_conda="${LINKERHAND_CONDA_EXE:-}"

if [[ -f "${_linkerhand_env}/bin/activate" ]]; then
    # A conda-pack environment is self-contained after conda-unpack and does
    # not require conda to be installed on the destination machine.
    source "${_linkerhand_env}/bin/activate"
elif [[ -x "${_linkerhand_env}/bin/python" ]]; then
    # Some copied/packed conda environments do not contain bin/activate.
    # Activating them only requires selecting their executables and prefix.
    export CONDA_PREFIX="${_linkerhand_env}"
    export PATH="${_linkerhand_env}/bin:${PATH}"
else
    if [[ -z "${_linkerhand_conda}" ]]; then
        _linkerhand_conda="$(command -v conda 2>/dev/null || true)"
    fi
    if [[ ! -x "${_linkerhand_conda}" ]]; then
        echo "Neither a packed LinkerHand environment nor conda was found." >&2
        return 1 2>/dev/null || exit 1
    fi
    eval "$("${_linkerhand_conda}" shell.bash hook)"
    conda activate "${_linkerhand_env}"
fi

export LINKERHAND_WORKSPACE="${_linkerhand_workspace}"
export LINKERHAND_SDK_ROOT="${_linkerhand_workspace}/../linkerhand-ros2-sdk-main"
export PYTHONPATH="${LINKERHAND_SDK_ROOT}/linker_hand_ros2_sdk${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1

unset _linkerhand_conda
unset _linkerhand_env
unset _linkerhand_workspace
