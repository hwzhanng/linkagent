#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
runtime_dir="${XDG_RUNTIME_DIR:-/tmp}/lbot-local-arm-can"

# Fail-safe first: disabling is allowed without an area-clear token.
/usr/bin/python3 "$script_dir/commission.py" disable --arm both || true

for name in right-ros right-service left-service; do
  pid_file="$runtime_dir/$name.pid"
  if [[ -f "$pid_file" ]]; then
    pid=$(<"$pid_file")
    if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
      kill -TERM "$pid"
    fi
    rm -f "$pid_file"
  fi
done

echo "local arm processes stopped; CAN interfaces remain UP for diagnostics"
