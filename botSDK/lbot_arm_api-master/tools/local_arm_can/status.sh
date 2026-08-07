#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
/usr/bin/python3 "$script_dir/commission.py" status --arm both
ip -brief -details link show can0
ip -brief -details link show can1
