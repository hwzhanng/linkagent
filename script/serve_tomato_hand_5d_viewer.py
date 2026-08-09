#!/usr/bin/env python3
"""Serve the hand-only URDF and exact production 5D mapping on localhost."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import mimetypes
from pathlib import Path
import subprocess
import sys
from urllib.parse import parse_qs, unquote, urlparse
import xml.etree.ElementTree as ET


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from module.skills.tomato_pick import G20_JOINT_ORDER, HAND_SYNERGY_ORDER  # noqa: E402
from module.skills.tomato_pick.hand_synergy import (  # noqa: E402
    HAND_JOINT_LIMITS,
    PREGRASP_SYNERGY,
    g20_synergy_to_joint_targets,
)
from module.skills.tomato_pick.hand_synergy_calibrated_v2 import (  # noqa: E402
    MAPPING_ID as CALIBRATED_V2_MAPPING_ID,
    g20_synergy_to_joint_targets_v2,
)


VIEWER_ROOT = PROJECT_ROOT / "tools/tomato_hand_5d_viewer"
PACKAGE_ROOT = (
    PROJECT_ROOT
    / "botSDK/urdf/02_project_derived/ros2_packages/linkbot_bimanual_sdk_description"
)
SOURCE_URDF = PACKAGE_ROOT / "urdf/linkbot_bimanual_sdk_description.urdf"
MAPPING_SOURCE = PROJECT_ROOT / "module/skills/tomato_pick/hand_synergy.py"
OUTPUT_ROOT = PROJECT_ROOT / "logs/hand_5d_viewer"
PACKAGE_URI_PREFIX = "package://linkbot_bimanual_sdk_description/"
MODEL_HTTP_PREFIX = "/model/package/"
MODEL_URDF_MESH_PREFIX = "package/"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=1)
def build_hand_only_urdf() -> bytes:
    """Extract the formal right-hand subtree without copying model meshes."""

    source = ET.parse(SOURCE_URDF).getroot()
    target = ET.Element("robot", {"name": "linkbot_tomato_right_hand_only"})
    for material in source.findall("material"):
        target.append(deepcopy(material))

    hand_links = {
        link.attrib["name"]
        for link in source.findall("link")
        if link.attrib.get("name", "").startswith("right_hand_")
    }
    if "right_hand_base_link" not in hand_links:
        raise RuntimeError("formal URDF has no right_hand_base_link")
    for link in source.findall("link"):
        if link.attrib.get("name") in hand_links:
            copied = deepcopy(link)
            for mesh in copied.iter("mesh"):
                filename = mesh.attrib.get("filename", "")
                if filename.startswith(PACKAGE_URI_PREFIX):
                    mesh.set(
                        "filename",
                        MODEL_URDF_MESH_PREFIX
                        + filename.removeprefix(PACKAGE_URI_PREFIX),
                    )
            target.append(copied)

    for joint in source.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if (
            parent is not None
            and child is not None
            and parent.attrib.get("link") in hand_links
            and child.attrib.get("link") in hand_links
        ):
            target.append(deepcopy(joint))

    return ET.tostring(target, encoding="utf-8", xml_declaration=True)


@lru_cache(maxsize=1)
def _formal_hand_metadata() -> tuple[list[dict[str, object]], int, int]:
    root = ET.fromstring(build_hand_only_urdf())
    mimic_joints: list[dict[str, object]] = []
    for joint in root.findall("joint"):
        mimic = joint.find("mimic")
        limit = joint.find("limit")
        if mimic is None or limit is None:
            continue
        mimic_joints.append(
            {
                "name": joint.attrib["name"],
                "driver": mimic.attrib["joint"],
                "multiplier": float(mimic.attrib.get("multiplier", "1")),
                "offset": float(mimic.attrib.get("offset", "0")),
                "lower": float(limit.attrib["lower"]),
                "upper": float(limit.attrib["upper"]),
            }
        )
    return mimic_joints, len(root.findall("link")), len(root.findall("joint"))


@lru_cache(maxsize=1)
def viewer_config() -> dict[str, object]:
    mimic_joints, link_count, urdf_joint_count = _formal_hand_metadata()
    active_joints = [
        {
            "name": f"right_hand_{name}",
            "lower": limits[0],
            "neutral": limits[1],
            "upper": limits[2],
        }
        for name, limits in zip(G20_JOINT_ORDER, HAND_JOINT_LIMITS, strict=True)
    ]
    return {
        "schema_version": 1,
        "synergy_order": list(HAND_SYNERGY_ORDER),
        "pregrasp_synergy": list(PREGRASP_SYNERGY),
        "active_joints": active_joints,
        "mimic_joints": mimic_joints,
        "urdf_link_count": link_count,
        "urdf_joint_count": urdf_joint_count,
        "source": {
            "urdf": str(SOURCE_URDF.relative_to(PROJECT_ROOT)),
            "urdf_sha256": _sha256(SOURCE_URDF),
            "mapping": str(MAPPING_SOURCE.relative_to(PROJECT_ROOT)),
            "mapping_sha256": _sha256(MAPPING_SOURCE),
            "mapper": "g20_synergy_to_joint_targets",
        },
        "safety": {
            "hardware_connected": False,
            "production_config_mutated": False,
            "draft_only": True,
        },
    }


def map_synergy(
    values: list[float], *, calibrated_v2: bool = False
) -> dict[str, object]:
    if len(values) != 5 or any(not math.isfinite(value) for value in values):
        raise ValueError("values must contain exactly five finite numbers")
    if any(value < -1.0 or value > 1.0 for value in values):
        raise ValueError("every synergy coordinate must lie in [-1, 1]")
    targets = (
        g20_synergy_to_joint_targets_v2(values)
        if calibrated_v2
        else g20_synergy_to_joint_targets(values)
    )
    names = [f"right_hand_{name}" for name in G20_JOINT_ORDER]
    active = dict(zip(names, targets, strict=True))
    config = viewer_config()
    all_targets = dict(active)
    for item in config["mimic_joints"]:  # type: ignore[index]
        mimic = item  # type: ignore[assignment]
        value = (
            float(mimic["multiplier"]) * active[str(mimic["driver"])]
            + float(mimic["offset"])
        )
        all_targets[str(mimic["name"])] = max(
            float(mimic["lower"]), min(float(mimic["upper"]), value)
        )
    return {
        "synergy": values,
        "mapping_id": (
            CALIBRATED_V2_MAPPING_ID if calibrated_v2 else "clean_v5_legacy_5d"
        ),
        "active_joint_targets": active,
        "all_revolute_joint_targets": all_targets,
    }


def _safe_child(root: Path, relative: str) -> Path | None:
    candidate = (root / unquote(relative).lstrip("/")).resolve()
    return candidate if candidate.is_relative_to(root.resolve()) else None


class ViewerHandler(BaseHTTPRequestHandler):
    server_version = "TomatoHand5DViewer/1.0"

    def _json(self, payload: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _bytes(self, body: bytes, content_type: str, *, cache: bool = False) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "public, max-age=3600" if cache else "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _file(self, root: Path, relative: str) -> None:
        path = _safe_child(root, relative)
        if path is None or not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if path.suffix.lower() == ".js":
            content_type = "text/javascript; charset=utf-8"
        elif path.suffix.lower() == ".stl":
            content_type = "model/stl"
        self._bytes(path.read_bytes(), content_type, cache=True)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        try:
            if path == "/api/health":
                self._json({"ok": True, "hardware_connected": False})
            elif path == "/api/config.json":
                self._json(viewer_config())
            elif path == "/api/latest-draft":
                latest = OUTPUT_ROOT / "latest_draft.json"
                if not latest.is_file():
                    self._json(
                        {"error": "no saved hand draft exists"},
                        HTTPStatus.NOT_FOUND,
                    )
                else:
                    self._bytes(
                        latest.read_bytes(),
                        "application/json; charset=utf-8",
                    )
            elif path in {"/api/map", "/api/map-v2"}:
                raw = parse_qs(parsed.query).get("values", [""])[0]
                values = [float(value) for value in raw.split(",") if value != ""]
                self._json(map_synergy(values, calibrated_v2=path.endswith("-v2")))
            elif path == "/model/right_hand_only.urdf":
                self._bytes(build_hand_only_urdf(), "application/xml; charset=utf-8")
            elif path.startswith(MODEL_HTTP_PREFIX):
                self._file(PACKAGE_ROOT, path.removeprefix(MODEL_HTTP_PREFIX))
            elif path.startswith("/node_modules/"):
                self._file(VIEWER_ROOT / "node_modules", path.removeprefix("/node_modules/"))
            elif path == "/":
                self._file(VIEWER_ROOT, "index.html")
            else:
                self._file(VIEWER_ROOT, path)
        except (ValueError, KeyError) as error:
            self._json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
        except Exception as error:  # pragma: no cover - defensive server boundary
            self._json({"error": f"{type(error).__name__}: {error}"}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_POST(self) -> None:  # noqa: N802
        if urlparse(self.path).path != "/api/save":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 2 or length > 1_000_000:
                raise ValueError("invalid draft size")
            payload = json.loads(self.rfile.read(length))
            if payload.get("schema_version") != 1:
                raise ValueError("unsupported draft schema")
            values_by_name = payload.get("synergy_values")
            if not isinstance(values_by_name, dict):
                raise ValueError("synergy_values must be an object")
            values = [float(values_by_name[name]) for name in HAND_SYNERGY_ORDER]
            map_synergy(values)
            OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
            destination = OUTPUT_ROOT / f"tomato_hand_5d_draft_{timestamp}.json"
            encoded = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
            destination.write_text(encoded, encoding="utf-8")
            latest = OUTPUT_ROOT / "latest_draft.json"
            latest.write_text(encoded, encoding="utf-8")
            self._json({"saved": True, "path": str(destination), "latest": str(latest)})
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
            self._json({"error": str(error)}, HTTPStatus.BAD_REQUEST)

    def log_message(self, format_string: str, *args: object) -> None:
        print(f"[{self.log_date_time_string()}] {format_string % args}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open", action="store_true")
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "localhost"}:
        raise ValueError("the viewer is intentionally restricted to localhost")
    if not (VIEWER_ROOT / "node_modules/three").is_dir():
        raise RuntimeError("viewer dependencies are missing; run npm install in tools/tomato_hand_5d_viewer")
    server = ThreadingHTTPServer((args.host, args.port), ViewerHandler)
    url = f"http://{args.host}:{args.port}/?mapping=v2&load=latest"
    print(f"TOMATO_HAND_5D_VIEWER={url}", flush=True)
    print(f"URDF_SOURCE={SOURCE_URDF}", flush=True)
    print(f"MAPPING_SOURCE={MAPPING_SOURCE}", flush=True)
    if args.open:
        subprocess.Popen(
            ["xdg-open", url],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
