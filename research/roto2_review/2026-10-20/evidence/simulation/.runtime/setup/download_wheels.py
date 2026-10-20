"""Resumable ranged downloads; verify upstream wheel SHA256 before use."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import threading
import time
import urllib.request

BASE = Path(__file__).parent / "wheels"
CHUNK = 32 * 1024 * 1024


def download(item):
    final = BASE / item["file"]
    if final.exists():
        with final.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() == item["sha256"]:
                print(f"REUSED {final.name}", flush=True)
                return
    partial = final.with_suffix(final.suffix + ".part")
    state_path = final.with_suffix(final.suffix + ".progress.json")
    req = urllib.request.Request(item["url"], headers={"Range": "bytes=0-0"})
    with urllib.request.urlopen(req, timeout=60) as response:
        if response.status != 206:
            raise RuntimeError("Server does not support byte ranges")
        size = int(response.headers["Content-Range"].split("/")[-1])
    state = json.loads(state_path.read_text()) if state_path.exists() else {"size": size, "done": []}
    assert state["size"] == size
    if not partial.exists():
        state["done"] = []
    fd = os.open(partial, os.O_CREAT | os.O_RDWR, 0o644)
    os.ftruncate(fd, size)
    lock = threading.Lock()
    start_time = time.monotonic()
    initial_done = len(state["done"])

    def fetch(index):
        start = index * CHUNK
        end = min(size, start + CHUNK) - 1
        for attempt in range(5):
            try:
                req = urllib.request.Request(item["url"], headers={"Range": f"bytes={start}-{end}"})
                with urllib.request.urlopen(req, timeout=90) as response:
                    expected_range = f"bytes {start}-{end}/{size}"
                    if response.status != 206 or response.headers.get("Content-Range") != expected_range:
                        raise RuntimeError("Unexpected byte range response")
                    offset = start
                    while block := response.read(1024 * 1024):
                        if offset + len(block) > end + 1:
                            raise RuntimeError("Oversized range response")
                        consumed = 0
                        while consumed < len(block):
                            consumed += os.pwrite(fd, block[consumed:], offset + consumed)
                        offset += len(block)
                    if offset != end + 1:
                        raise RuntimeError("Incomplete byte range")
                with lock:
                    state["done"].append(index)
                    temporary = state_path.with_suffix(".tmp")
                    temporary.write_text(json.dumps(state))
                    temporary.replace(state_path)
                    count = len(state["done"])
                    rate = (count - initial_done) * CHUNK / max(1, time.monotonic() - start_time) / 1024**2
                    print(f'{item["file"]}: {count}/{(size + CHUNK - 1)//CHUNK} chunks, {rate:.1f} MiB/s', flush=True)
                return
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt)

    try:
        pending = [i for i in range((size + CHUNK - 1) // CHUNK) if i not in state["done"]]
        with concurrent.futures.ThreadPoolExecutor(max_workers=int(os.environ.get("WHEEL_CONNECTIONS", "16"))) as pool:
            list(pool.map(fetch, pending))
        os.fsync(fd)
    finally:
        os.close(fd)
    digest = hashlib.file_digest(partial.open("rb"), "sha256").hexdigest()
    if digest != item["sha256"]:
        raise RuntimeError(f"SHA256 mismatch for {partial}: {digest}")
    partial.replace(final)
    print(f"VERIFIED {final.name} {digest}", flush=True)


manifest = Path(sys.argv[1]) if len(sys.argv) > 1 else BASE / "downloads.json"
with concurrent.futures.ThreadPoolExecutor(max_workers=int(os.environ.get("WHEEL_PARALLEL_FILES", "1"))) as pool:
    list(pool.map(download, json.loads(manifest.read_text())))
