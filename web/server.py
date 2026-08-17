from __future__ import annotations

import json
import mimetypes
import threading
import time
from collections import deque
from copy import deepcopy
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

from config import SimulationConfig
from simulation.engine import SimulationEngine


class WebSimulationController:
    """Runs one real engine tick at the user-selected visual pace."""

    def __init__(
        self,
        engine: SimulationEngine,
        ticks_per_second: float = 4.0,
        next_engine_factory: Callable[[Path, SimulationConfig, int], SimulationEngine] | None = None,
        new_engine_factory: Callable[[SimulationConfig, int], SimulationEngine] | None = None,
    ) -> None:
        self.engine = engine
        self._next_engine_factory = next_engine_factory
        self._new_engine_factory = new_engine_factory
        self.ticks_per_second = ticks_per_second
        self.status = "paused"
        self.error: str | None = None
        self.result: dict[str, str] | None = None
        self.continued_from = str(engine.source_checkpoint) if engine.source_checkpoint else None
        self._events: deque[dict[str, object]] = deque(maxlen=80)
        self._lock = threading.RLock()
        self._wake = threading.Event()
        self._stopped = threading.Event()
        self._cached_snapshot = engine.snapshot()
        self._thread = threading.Thread(target=self._loop, name="lifesim-engine", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        self._stopped.set()
        self._wake.set()
        self._thread.join(timeout=2.0)

    def control(self, action: str, value: object = None) -> dict[str, object]:
        if action == "play":
            with self._lock:
                if self.status not in {"completed", "finalizing", "error"}:
                    self.status = "running"
            self._wake.set()
        elif action == "pause":
            with self._lock:
                if self.status == "running":
                    self.status = "paused"
        elif action == "step":
            with self._lock:
                allowed = self.status == "paused"
            if allowed:
                self._advance_one_tick()
        elif action == "speed":
            if not isinstance(value, (int, float)) or not 0.5 <= value <= 60.0:
                raise ValueError("Speed must be between 0.5 and 60 ticks per second")
            with self._lock:
                self.ticks_per_second = float(value)
            self._wake.set()
        elif action == "next_run":
            self._start_next_run()
        elif action == "new_experiment":
            self._start_new_experiment(value)
        else:
            raise ValueError(f"Unknown control action: {action}")
        return self.state()

    def state(self) -> dict[str, object]:
        with self._lock:
            snapshot = dict(self._cached_snapshot)
            snapshot.update({
                "status": self.status,
                "ticks_per_second": self.ticks_per_second,
                "error": self.error,
                "recent_events": list(self._events),
                "result": self.result,
                "can_start_next_run": self.status == "completed" and self.result is not None,
                "can_configure_experiment": (
                    self.status == "completed"
                    or (
                        self.status == "paused"
                        and self.engine.current_tick == 0
                        and self.engine.source_checkpoint is None
                    )
                ),
                "continued_from": self.continued_from,
            })
            return snapshot

    def _loop(self) -> None:
        while not self._stopped.is_set():
            with self._lock:
                running = self.status == "running"
                speed = self.ticks_per_second
            if not running:
                self._wake.wait(timeout=0.25)
                self._wake.clear()
                continue
            started = time.monotonic()
            self._advance_one_tick()
            elapsed = time.monotonic() - started
            self._wake.wait(timeout=max(0.0, 1.0 / speed - elapsed))
            self._wake.clear()

    def _advance_one_tick(self) -> None:
        should_finalize = False
        try:
            with self._lock:
                if self.engine.is_complete or self.status in {"completed", "finalizing", "error"}:
                    return
                advanced = self.engine.step()
                if advanced:
                    self._cached_snapshot = self.engine.snapshot()
                    self._capture_events()
                if self.engine.is_complete:
                    self.status = "finalizing"
                    should_finalize = True
            if should_finalize:
                self._finalize()
        except Exception as error:  # Surface background failures to the browser.
            with self._lock:
                self.status = "error"
                self.error = f"{type(error).__name__}: {error}"

    def _capture_events(self) -> None:
        for event in self.engine.last_agent_events.values():
            self._events.appendleft(dict(event))

    def _finalize(self) -> None:
        try:
            result = self.engine.finalize()
            with self._lock:
                self.result = {
                    "checkpoint_dir": str(result.checkpoint_dir),
                    "results_dir": str(result.results_dir),
                }
                self._cached_snapshot = self.engine.snapshot()
                self.status = "completed"
        except Exception as error:
            with self._lock:
                self.status = "error"
                self.error = f"{type(error).__name__}: {error}"

    def _start_next_run(self) -> None:
        """Rebuild agents from the completed checkpoint and immediately continue."""
        with self._lock:
            if self.status != "completed" or self.result is None:
                raise ValueError("The current run must finish before starting the next one")
            if self._next_engine_factory is None:
                raise ValueError("This web session cannot create a subsequent run")
            checkpoint_dir = Path(self.result["checkpoint_dir"])
            config = self.engine.config
            next_seed = self.engine.seed + 1
            self.status = "preparing"
        try:
            new_engine = self._next_engine_factory(checkpoint_dir, config, next_seed)
        except Exception as error:
            with self._lock:
                self.status = "completed"
                self.error = f"{type(error).__name__}: {error}"
            return
        with self._lock:
            self.engine = new_engine
            self.error = None
            self.result = None
            self.continued_from = str(checkpoint_dir)
            self._events.clear()
            self._cached_snapshot = new_engine.snapshot()
            self.status = "running"
        self._wake.set()

    def _start_new_experiment(self, value: object) -> None:
        """Create a fresh paused experiment from validated web controls."""
        if not isinstance(value, dict):
            raise ValueError("New experiment settings must be an object")
        try:
            humans = int(value["num_humans"])
            animals = int(value["num_animals"])
            human_width = int(value["human_brain_width"])
            animal_width = int(value["animal_brain_width"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Incomplete new experiment settings") from error
        if not 1 <= humans <= 30:
            raise ValueError("Humans must be between 1 and 30")
        if not 1 <= animals <= 50:
            raise ValueError("Animals must be between 1 and 50")
        if human_width not in range(8, 65, 8):
            raise ValueError("Human brain width must be 8..64 in steps of 8")
        if animal_width not in range(8, 65, 8):
            raise ValueError("Animal brain width must be 8..64 in steps of 8")
        with self._lock:
            configurable = self.status == "completed" or (
                self.status == "paused"
                and self.engine.current_tick == 0
                and self.engine.source_checkpoint is None
            )
            if not configurable:
                raise ValueError(
                    "Configure population before the first tick or after the run completes"
                )
            if self._new_engine_factory is None:
                raise ValueError("This web session cannot create a new experiment")
            config = deepcopy(self.engine.config)
            config.num_humans = humans
            config.num_animals = animals
            config.human_brain.hidden_sizes = [
                max(8, human_width // 2), human_width, human_width
            ]
            config.animal_brain.hidden_sizes = [
                max(8, animal_width // 2), animal_width, animal_width
            ]
            seed = self.engine.seed if self.engine.current_tick == 0 else self.engine.seed + 1
            previous_status = self.status
            self.status = "preparing"
        try:
            new_engine = self._new_engine_factory(config, seed)
        except Exception as error:
            with self._lock:
                self.status = previous_status
                self.error = f"{type(error).__name__}: {error}"
            return
        with self._lock:
            self.engine = new_engine
            self.error = None
            self.result = None
            self.continued_from = None
            self._events.clear()
            self._cached_snapshot = new_engine.snapshot()
            self.status = "paused"
        self._wake.set()


class LifeSimRequestHandler(BaseHTTPRequestHandler):
    controller: WebSimulationController
    static_dir: Path

    def do_GET(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        path = urlparse(self.path).path
        if path == "/api/state":
            self._send_json(self.controller.state())
            return
        if path == "/api/health":
            self._send_json({"ok": True, "status": self.controller.status})
            return
        relative = "index.html" if path == "/" else path.removeprefix("/")
        target = (self.static_dir / relative).resolve()
        if self.static_dir.resolve() not in target.parents and target != self.static_dir.resolve():
            self.send_error(HTTPStatus.FORBIDDEN)
            return
        if not target.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        content_type, _ = mimetypes.guess_type(target.name)
        body = target.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type or "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        if urlparse(self.path).path != "/api/control":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length > 10_000:
                raise ValueError("Request body is too large")
            payload = json.loads(self.rfile.read(length) or b"{}")
            state = self.controller.control(payload.get("action", ""), payload.get("value"))
            self._send_json(state)
        except (ValueError, json.JSONDecodeError) as error:
            self._send_json({"error": str(error)}, status=HTTPStatus.BAD_REQUEST)

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _send_json(self, payload: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def serve_web(engine: SimulationEngine, host: str, port: int, root: Path) -> None:
    from simulation.experiment import build_new_engine, build_resumed_engine

    def next_engine_factory(
        checkpoint_dir: Path, config_object: SimulationConfig, seed: int
    ) -> SimulationEngine:
        return build_resumed_engine(root, checkpoint_dir, config_object, seed)

    def new_engine_factory(
        config_object: SimulationConfig, seed: int
    ) -> SimulationEngine:
        return build_new_engine(root, config_object, seed)

    controller = WebSimulationController(
        engine,
        next_engine_factory=next_engine_factory,
        new_engine_factory=new_engine_factory,
    )
    static_dir = root / "web" / "static"
    handler_class = type(
        "ConfiguredLifeSimHandler",
        (LifeSimRequestHandler,),
        {"controller": controller, "static_dir": static_dir},
    )
    server = ThreadingHTTPServer((host, port), handler_class)
    controller.start()
    print(f"LifeSim web laboratory: http://{host}:{port}")
    print("The simulation is paused. Press Iniciar in the browser.")
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        print("\nStopping LifeSim web laboratory.")
    finally:
        controller.close()
        server.server_close()
