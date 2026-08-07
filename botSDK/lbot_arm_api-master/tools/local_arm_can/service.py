#!/usr/bin/env python3
"""Standalone local Linkbot arm CAN state/HTTP service.

Listening is the default.  Transmit support must be explicitly enabled, and
commands capable of changing mechanical state require a per-request physical
area confirmation.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import signal
import threading
import time
from typing import Any

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from tools.local_arm_can.protocol import (  # type: ignore
        ArmState,
        LEFT_BASE_ID,
        RIGHT_BASE_ID,
        ProtocolError,
        control_mode_frame,
        decode_frame,
        disable_frame,
        enable_frame,
        position_frame,
        set_auto_report_frame,
        state_query_frame,
        temperature_query_frame,
    )
    from tools.local_arm_can.socketcan import (  # type: ignore
        SocketCan,
        configure_interface,
        interface_info,
        list_can_interfaces,
    )
else:
    from .protocol import (
        ArmState,
        LEFT_BASE_ID,
        RIGHT_BASE_ID,
        ProtocolError,
        control_mode_frame,
        decode_frame,
        disable_frame,
        enable_frame,
        position_frame,
        set_auto_report_frame,
        state_query_frame,
        temperature_query_frame,
    )
    from .socketcan import SocketCan, configure_interface, interface_info, list_can_interfaces


PHYSICAL_CONFIRMATIONS = {
    "right": "RIGHT_ARM_AREA_CLEAR",
    "left": "LEFT_ARM_AREA_CLEAR",
}


class ArmCanRuntime:
    def __init__(
        self,
        interface: str,
        *,
        arm: str = "right",
        allow_tx: bool = False,
        allow_can0: bool = False,
        poll_telemetry: bool = False,
    ) -> None:
        if arm not in ("right", "left"):
            raise ProtocolError("arm must be right or left")
        self.interface = interface
        self.arm = arm
        self.base_id = RIGHT_BASE_ID if arm == "right" else LEFT_BASE_ID
        with Path(__file__).with_name(f"a7_v2_{arm}.json").open(encoding="utf-8") as stream:
            config = json.load(stream)
        limits = config["joint_limits"]["current"]
        self._joint_lower = [float(value) for value in limits["lower"]]
        self._joint_upper = [float(value) for value in limits["upper"]]
        self.allow_tx = allow_tx
        if poll_telemetry and not allow_tx:
            raise ProtocolError("telemetry polling requires explicit TX authorization")
        self.poll_telemetry = poll_telemetry
        self.state = ArmState(self.base_id)
        self._bus = SocketCan(
            interface,
            tx_enabled=allow_tx,
            allow_can0=allow_can0,
            timeout=0.1,
        )
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._telemetry_thread: threading.Thread | None = None
        self._write_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._condition = threading.Condition()
        self._sequence = 0
        self._last_error: str | None = None
        self._software_enabled = False

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._receive_loop, name="arm-can-rx", daemon=True)
        self._thread.start()
        if self.poll_telemetry:
            self._telemetry_thread = threading.Thread(
                target=self._telemetry_loop,
                name="arm-can-telemetry",
                daemon=True,
            )
            self._telemetry_thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._telemetry_thread is not None:
            self._telemetry_thread.join(timeout=1.0)
        self._bus.close()

    def _receive_loop(self) -> None:
        while not self._stop.is_set():
            try:
                frame = self._bus.receive()
                if frame is None:
                    continue
                event = decode_frame(frame, self.base_id)
                with self._state_lock:
                    self.state.rx_frames += 1
                    if event is None:
                        self.state.unknown_frames += 1
                    else:
                        self.state.update(event)
                with self._condition:
                    self._sequence += 1
                    self._condition.notify_all()
            except Exception as error:  # keep diagnostics alive after transient CAN faults
                self._last_error = f"{type(error).__name__}: {error}"
                time.sleep(0.05)

    def _telemetry_loop(self) -> None:
        """Reproduce the official HAL's no-motion telemetry schedule."""
        # Classic J1-J5 support automatic reports.  Keep those enabled, but
        # also query all seven joints at 50 Hz.  Real-hardware acceptance found
        # that a classic joint can occasionally pause its automatic reports
        # for longer than the 250 ms motion watchdog while the CAN bus itself
        # remains healthy.  Explicit queries preserve the watchdog threshold
        # instead of hiding that gap by making stale_after more permissive.
        for index in range(5):
            try:
                frame = set_auto_report_frame(self.base_id + index, 20.0)
                if frame is not None:
                    self._send(frame)
            except Exception as error:
                self._last_error = f"{type(error).__name__}: {error}"
            if self._stop.wait(0.01):
                return

        next_temperature = time.monotonic()
        period = 0.02
        while not self._stop.is_set():
            started = time.monotonic()
            try:
                for index in range(7):
                    self._send(state_query_frame(self.base_id + index))
                if started >= next_temperature:
                    for index in range(5):
                        frame = temperature_query_frame(self.base_id + index)
                        if frame is not None:
                            self._send(frame)
                    next_temperature = started + 1.0
            except Exception as error:
                self._last_error = f"{type(error).__name__}: {error}"
            self._stop.wait(max(0.0, period - (time.monotonic() - started)))

    def snapshot(self) -> dict[str, Any]:
        with self._state_lock:
            result = self.state.snapshot()
        result.update(
            interface=self.interface,
            arm=self.arm,
            tx_enabled=self.allow_tx,
            telemetry_polling=self.poll_telemetry,
            software_enabled=self._software_enabled,
            last_error=self._last_error,
        )
        return result

    def wait_for_update(self, sequence: int, timeout: float) -> int:
        with self._condition:
            if self._sequence == sequence:
                self._condition.wait(timeout=timeout)
            return self._sequence

    def sequence(self) -> int:
        with self._condition:
            return self._sequence

    def _send(self, frame: Any) -> None:
        with self._write_lock:
            self._bus.send(frame)
            with self._state_lock:
                self.state.tx_frames += 1

    def probe(self) -> None:
        if not self.allow_tx:
            raise PermissionError("service is read-only; restart with --allow-tx for a no-motion probe")
        for index in range(7):
            can_id = self.base_id + index
            self._send(state_query_frame(can_id))
            temperature = temperature_query_frame(can_id)
            if temperature is not None:
                self._send(temperature)

    def require_physical_confirmation(self, value: str | None) -> None:
        expected = PHYSICAL_CONFIRMATIONS[self.arm]
        if value != expected:
            raise PermissionError(
                f"physical state change requires confirmation {expected!r}"
            )

    def set_enabled(self, enabled: bool, confirmation: str | None) -> None:
        if not self.allow_tx:
            raise PermissionError("service is read-only")
        if enabled:
            self.require_physical_confirmation(confirmation)
            snapshot = self.snapshot()
            if not snapshot["all_joints_healthy"]:
                raise RuntimeError(
                    f"refusing enable: only {snapshot['healthy_joint_count']}/7 joints are healthy"
                )
            # Match the official Pi startup sequence: disable first, then set
            # control mode, and only then enable.  This prevents changing mode
            # while a joint is carrying an active command.
            for index in range(7):
                self._send(disable_frame(self.base_id + index))
            time.sleep(0.1)
            for index in range(7):
                self._send(control_mode_frame(self.base_id + index, 4))
                time.sleep(0.1)
            for index in range(7):
                self._send(enable_frame(self.base_id + index))
            self._software_enabled = True
        else:
            # A disable/emergency-stop path must never depend on an area-clear
            # token.  It is the fail-safe action and remains available whenever
            # CAN transmission itself was explicitly authorized at startup.
            for index in range(7):
                self._send(disable_frame(self.base_id + index))
            self._software_enabled = False

    def command_positions(
        self,
        positions: list[float],
        confirmation: str | None,
        evo_parameters: list[float] | None,
    ) -> None:
        self.require_physical_confirmation(confirmation)
        if not self.allow_tx or not self._software_enabled:
            raise PermissionError("arm transmission is not enabled")
        if len(positions) != 7 or not all(math.isfinite(value) for value in positions):
            raise ProtocolError("positions must contain seven finite radians")
        for index, (position, lower, upper) in enumerate(
            zip(positions, self._joint_lower, self._joint_upper, strict=True), start=1
        ):
            if not lower <= position <= upper:
                raise ProtocolError(
                    f"J{index} target {position:.6f} rad exceeds software limit "
                    f"[{lower:.6f}, {upper:.6f}]"
                )
        snapshot = self.snapshot()
        if not snapshot["all_joints_healthy"]:
            stale = []
            for row in snapshot["joints"]:
                if row["healthy"]:
                    continue
                age = row["state_age"]
                age_text = "missing" if age is None else f"{age:.3f}s"
                stale.append(f"J{row['joint_number']} age={age_text}")
            details = ", ".join(stale) or "unknown joint"
            raise RuntimeError(
                "refusing motion while joint feedback is stale or missing: " + details
            )
        for index, position in enumerate(positions):
            can_id = self.base_id + index
            frame = position_frame(can_id, position, evo_parameters=evo_parameters)
            self._send(frame)


class ApiHandler(BaseHTTPRequestHandler):
    server: "ArmHttpServer"

    def log_message(self, format_string: str, *args: object) -> None:
        if args and "/api/joint_positions" in str(args[0]):
            return
        print(f"[http] {self.address_string()} {format_string % args}")

    def _json(self, status: int, payload: Any) -> None:
        encoded = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def _body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length > 64 * 1024:
            raise ProtocolError("request body is too large")
        return json.loads(self.rfile.read(length) or b"{}")

    def do_GET(self) -> None:  # noqa: N802
        if self.path in ("/", "/health", "/state"):
            self._json(HTTPStatus.OK, self.server.runtime.snapshot())
            return
        if self.path == "/events":
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            sequence = self.server.runtime.sequence()
            try:
                while not self.server.stopping.is_set():
                    sequence = self.server.runtime.wait_for_update(sequence, 1.0)
                    payload = json.dumps(self.server.runtime.snapshot(), ensure_ascii=False)
                    self.wfile.write(f"event: state\ndata: {payload}\n\n".encode("utf-8"))
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            return
        self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        try:
            body = self._body()
            if self.path == "/api/probe":
                self.server.runtime.probe()
            elif self.path == "/api/enable":
                self.server.runtime.set_enabled(True, body.get("confirmation"))
            elif self.path == "/api/disable":
                self.server.runtime.set_enabled(False, body.get("confirmation"))
            elif self.path == "/api/joint_positions":
                self.server.runtime.command_positions(
                    [float(value) for value in body.get("positions", [])],
                    body.get("confirmation"),
                    body.get("evo_parameters"),
                )
            else:
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            self._json(HTTPStatus.OK, {"ok": True, "state": self.server.runtime.snapshot()})
        except (PermissionError, ProtocolError, RuntimeError, ValueError, json.JSONDecodeError) as error:
            self._json(HTTPStatus.CONFLICT, {"ok": False, "error": str(error)})


class ArmHttpServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], runtime: ArmCanRuntime) -> None:
        self.runtime = runtime
        self.stopping = threading.Event()
        super().__init__(address, ApiHandler)

    def server_close(self) -> None:
        self.stopping.set()
        super().server_close()


def doctor() -> int:
    interfaces = [asdict(item) for item in list_can_interfaces()]
    print(
        json.dumps(
            {
                "can_interfaces": interfaces,
                "protected_interface": "can0",
                "required_bitrate": 1_000_000,
                "ready": bool(interfaces),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if interfaces else 2


def monitor(args: argparse.Namespace) -> int:
    runtime = ArmCanRuntime(
        args.interface,
        arm=args.arm,
        allow_tx=False,
        allow_can0=args.allow_can0,
    )
    runtime.start()
    try:
        deadline = time.monotonic() + args.seconds if args.seconds else None
        while deadline is None or time.monotonic() < deadline:
            print(json.dumps(runtime.snapshot(), ensure_ascii=False))
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        runtime.close()
    return 0


def serve(args: argparse.Namespace) -> int:
    runtime = ArmCanRuntime(
        args.interface,
        arm=args.arm,
        allow_tx=args.allow_tx,
        allow_can0=args.allow_can0,
        poll_telemetry=args.poll_telemetry,
    )
    runtime.start()
    server = ArmHttpServer((args.bind, args.port), runtime)

    def stop(_signum: int, _frame: object) -> None:
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    print(
        f"local arm CAN service: http://{args.bind}:{args.port} "
        f"arm={args.arm} interface={args.interface} tx={'enabled' if args.allow_tx else 'locked'}"
    )
    try:
        server.serve_forever(poll_interval=0.2)
    finally:
        server.server_close()
        runtime.close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("doctor", help="list locally detected CAN interfaces")

    configure = subparsers.add_parser("configure", help="explicitly configure one interface at 1 Mbit/s")
    configure.add_argument("interface")
    configure.add_argument(
        "--allow-can0",
        action="store_true",
        help="override can0 protection after confirming it is the left mechanical arm",
    )

    for name in ("monitor", "serve"):
        child = subparsers.add_parser(name)
        child.add_argument("--interface", required=True)
        child.add_argument("--arm", choices=("right", "left"), default="right")
        child.add_argument(
            "--allow-can0",
            action="store_true",
            help="override local can0 protection only after its physical cable has been identified",
        )
        if name == "monitor":
            child.add_argument("--seconds", type=float, default=0.0)
        else:
            child.add_argument("--bind", default="127.0.0.1")
            child.add_argument("--port", type=int, default=8766)
            child.add_argument("--allow-tx", action="store_true")
            child.add_argument(
                "--poll-telemetry",
                action="store_true",
                help="initialize classic auto-report and poll Motorevo state at 50 Hz",
            )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "doctor":
        return doctor()
    if args.command == "configure":
        configure_interface(args.interface, allow_can0=args.allow_can0)
        print(json.dumps(asdict(interface_info(args.interface)), ensure_ascii=False, indent=2))
        return 0
    if args.command == "monitor":
        return monitor(args)
    if args.command == "serve":
        return serve(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
