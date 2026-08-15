"""
================================================================================
PROJECT     : FXP - Forex Autonomous Trading Bot
REPOSITORY  : https://github.com/raidvoltus/FX
MODULE      : main.py
VERSION     : 2026.Q3.v4.6.0 (HALTED State Enforcement Supervisor)
PYTHON      : 3.11+
================================================================================
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import socket
import sys
import time
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional
from urllib.parse import urlparse

from config import Config
from autonomous_engine import AutonomousEngine

logger = logging.getLogger("FXP.Main")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] [FXP.Main] %(message)s"))
    logger.addHandler(handler)

logger.setLevel(getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO))
logger.propagate = False


def mask_account_id(account_id: str) -> str:
    if not account_id or len(account_id) < 6:
        return "***"
    return f"{account_id[:3]}-***-***-{account_id[-3:]}"


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


class RuntimeMode(str, Enum):
    AUTO = "auto"
    PERSISTENT = "persistent"
    SCHEDULED = "scheduled"


class OandaEnvironment(str, Enum):
    PRACTICE = "PRACTICE"
    LIVE = "LIVE"
    UNKNOWN = "UNKNOWN"


class SupervisorState(str, Enum):
    STARTING = "STARTING"
    VALIDATING = "VALIDATING"
    RECONCILING = "RECONCILING"
    RUNNING = "RUNNING"
    DEGRADED = "DEGRADED"
    HALTED = "HALTED"
    STOPPING = "STOPPING"


def detect_oanda_environment(api_url: str) -> OandaEnvironment:
    if not api_url:
        return OandaEnvironment.UNKNOWN
    try:
        parsed = urlparse(api_url.strip().rstrip("/"))
        hostname = (parsed.hostname or "").lower().rstrip(".")
    except Exception:
        return OandaEnvironment.UNKNOWN

    if hostname == "api-fxpractice.oanda.com":
        return OandaEnvironment.PRACTICE
    if hostname == "api-fxtrade.oanda.com":
        return OandaEnvironment.LIVE
    return OandaEnvironment.UNKNOWN


def detect_runtime() -> RuntimeMode:
    if os.getenv("GITHUB_ACTIONS", "").lower() == "true":
        return RuntimeMode.SCHEDULED
    if os.getenv("FXP_FORCE_SCHEDULED", "").lower() == "true":
        return RuntimeMode.SCHEDULED
    return RuntimeMode.PERSISTENT


class AtomicInstanceLock:

    def __init__(self, path: str) -> None:
        self.path = path
        self.acquired = False

    def acquire(self) -> None:
        directory = os.path.dirname(self.path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        if os.path.exists(self.path):
            previous_pid = None
            try:
                with open(self.path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                    previous_pid = int(data.get("pid", 0))
            except Exception:
                previous_pid = None

            if previous_pid and _pid_alive(previous_pid):
                raise RuntimeError(f"FXP instance lock aktif (PID={previous_pid}). Eksekusi dibatalkan.")

            logger.warning("Stale lock file terdeteksi (PID=%s). Membersihkan lock file lama...", previous_pid)
            try:
                os.remove(self.path)
            except OSError as exc:
                raise RuntimeError(f"Gagal menghapus stale lock file: {exc}") from exc

        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            payload = {
                "pid": os.getpid(),
                "hostname": socket.gethostname(),
                "started_at": datetime.now(timezone.utc).isoformat()
            }
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh)
            self.acquired = True
        except FileExistsError:
            raise RuntimeError("Race condition terdeteksi: Instance lock dibuat oleh proses lain.") from None

    def release(self) -> None:
        if not self.acquired:
            return
        try:
            if os.path.exists(self.path):
                os.remove(self.path)
        except Exception as exc:
            logger.warning("Gagal melepaskan instance lock file: %s", exc)
        self.acquired = False


class FXPAutonomousOrchestrator:

    def __init__(self, runtime: RuntimeMode = RuntimeMode.AUTO, interval: Optional[int] = None) -> None:
        self.state = SupervisorState.STARTING
        self.is_running = False
        self.cycle_count = 0
        self.cleanup_done = False
        self.consecutive_failures = 0

        self.runtime = detect_runtime() if runtime == RuntimeMode.AUTO else runtime
        self.config = Config.load()

        raw_api_url = getattr(self.config, "OANDA_API_URL", "")
        self.oanda_env = detect_oanda_environment(raw_api_url)
        if self.oanda_env == OandaEnvironment.UNKNOWN:
            logger.critical("CRITICAL: Hostname OANDA_API_URL '%s' tidak terdaftar pada Whitelist!", raw_api_url)
            sys.exit(1)

        default_lock = "/tmp/fxp.lock" if os.name != "nt" else os.path.join(os.getenv("TEMP", "."), "fxp.lock")
        self.lock = AtomicInstanceLock(os.getenv("FXP_LOCK_FILE", default_lock))
        self.lock.acquire()

        self.base_interval = max(1, int(interval if interval is not None else getattr(self.config, "LOOP_INTERVAL_SECONDS", 60)))
        self.max_backoff_interval = 300

        logger.info("Menginisialisasi AutonomousEngine...")
        self.engine = AutonomousEngine()

    def setup_and_validate(self) -> None:
        self.state = SupervisorState.VALIDATING

        logger.info("Mengeksekusi otentikasi REST API OANDA...")
        is_valid, reason = self.engine.validate_account_credentials()
        if not is_valid:
            self.state = SupervisorState.HALTED
            logger.critical("VALIDASI KREDENSIAL GAGAL: %s", reason)
            self.lock.release()
            sys.exit(1)

        self.state = SupervisorState.RECONCILING
        logger.info("Mengeksekusi rekonsiliasi total remote state (Positions, Trades, Orders)...")
        rec_ok, rec_msg = self.engine.reconcile_remote_state()
        if not rec_ok:
            self.state = SupervisorState.HALTED
            logger.critical("REKONSILIASI STARTUP GAGAL: %s", rec_msg)
            self.lock.release()
            sys.exit(1)

        logger.info("✅ Remote State Sync Berhasil: %s", rec_msg)
        self.state = SupervisorState.RUNNING

        self._install_signal_handlers()
        self._startup_log()

    def _startup_log(self) -> None:
        masked_acc = mask_account_id(getattr(self.config, "OANDA_ACCOUNT_ID", ""))
        logger.info("=" * 78)
        logger.info("FXP AUTONOMOUS TRADING ENGINE v4.6.0 (STRICT FAIL-CLOSED)")
        logger.info("=" * 78)
        logger.info("Runtime Mode     : %s", self.runtime.value.upper())
        logger.info("OANDA Environment: %s", self.oanda_env.value)
        logger.info("OANDA Endpoint   : %s", self.config.OANDA_API_URL)
        logger.info("Account ID       : %s", masked_acc)
        logger.info("Process PID      : %s", os.getpid())
        logger.info("Base Interval    : %s sec", self.base_interval)
        logger.info("=" * 78)

    def _install_signal_handlers(self) -> None:
        try:
            signal.signal(signal.SIGINT, self._shutdown_signal)
            signal.signal(signal.SIGTERM, self._shutdown_signal)
        except ValueError:
            logger.warning("OS signal handler tidak dapat dipasang.")

    def _shutdown_signal(self, signum: int, frame: Any) -> None:
        logger.warning("Sinyal shutdown OS (%s) diterima. Mengubah supervisor state ke STOPPING...", signum)
        self.state = SupervisorState.STOPPING
        self.is_running = False

    def run_cycle(self) -> Dict[str, Any]:
        cycle_id = self.cycle_count + 1
        started = time.perf_counter()

        logger.info("--------------------------------------------------")
        logger.info("CYCLE #%s START | state=%s | env=%s", cycle_id, self.state.value, self.oanda_env.value)

        try:
            result = self.engine.run_cycle()
            elapsed = (time.perf_counter() - started) * 1000.0

            if not isinstance(result, dict):
                result = {"status": "ERROR", "reason": f"Tipe respon invalid: {type(result).__name__}"}

            status = str(result.get("status", "UNKNOWN")).upper()

            if status == "HALTED":
                # P0 HALT ENFORCEMENT: Matikan loop supervisor seketika jika engine mengembalikan status HALTED
                self.state = SupervisorState.HALTED
                self.is_running = False
                logger.critical("🚨 ENGINE HALTED: Loop supervisor dihentikan demi keamanan dana.")
            elif status in ["SUCCESS", "PARTIAL"]:
                self.consecutive_failures = 0
                if self.state == SupervisorState.DEGRADED:
                    self.state = SupervisorState.RUNNING
                    logger.info("Sistem pulih dari kondisi DEGRADED ke RUNNING.")
            else:
                self.consecutive_failures += 1
                self.state = SupervisorState.DEGRADED
                logger.warning("Cycle #%s menghasilkan status non-success: %s", cycle_id, status)

            self.cycle_count = cycle_id
            result.update({
                "cycle_id": cycle_id,
                "runtime": self.runtime.value,
                "supervisor_state": self.state.value,
                "orchestrator_latency_ms": elapsed,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            return result

        except Exception as exc:
            elapsed = (time.perf_counter() - started) * 1000.0
            self.consecutive_failures += 1
            self.state = SupervisorState.DEGRADED
            self.cycle_count = cycle_id
            logger.warning("CYCLE #%s DEGRADED (API/Network Error) | %.2f ms | %s", cycle_id, elapsed, exc)
            return {
                "status": "ERROR",
                "cycle_id": cycle_id,
                "supervisor_state": self.state.value,
                "reason": str(exc),
                "orchestrator_latency_ms": elapsed,
            }

    def start(self) -> None:
        try:
            self.setup_and_validate()

            if self.runtime == RuntimeMode.SCHEDULED:
                logger.info("SCHEDULED MODE: Menjalankan 1 siklus eksekusi.")
                self.run_cycle()
                return

            self.is_running = True
            logger.info("PERSISTENT MODE: Daemon supervisor aktif.")
            while self.is_running:
                self.run_cycle()
                if not self.is_running:
                    break
                
                if self.consecutive_failures > 0:
                    backoff = min(self.max_backoff_interval, self.base_interval * (2 ** (self.consecutive_failures - 1)))
                    logger.warning("Kondisi DEGRADED (Failures=%d). Backoff interval: %d detik.", self.consecutive_failures, backoff)
                    self._sleep_interruptible(backoff)
                else:
                    self._sleep_interruptible(self.base_interval)

        finally:
            self.cleanup()

    def _sleep_interruptible(self, seconds: int) -> None:
        deadline = time.monotonic() + seconds
        while self.is_running:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            time.sleep(min(1.0, remaining))

    def cleanup(self) -> None:
        if self.cleanup_done:
            return
        self.cleanup_done = True
        self.is_running = False

        logger.info("Menutup supervisor dan membersihkan resource...")
        for method in ("shutdown", "close", "stop"):
            fn = getattr(self.engine, method, None)
            if callable(fn):
                try:
                    fn()
                except Exception as exc:
                    logger.error("Gagal membersihkan engine: %s", exc)
                break

        self.lock.release()
        logger.info("FXP Engine dihentikan bersih. Total cycle=%s.", self.cycle_count)


def main() -> int:
    parser = argparse.ArgumentParser(description="FXP Multi-Environment Autonomous Forex Trading Bot")
    parser.add_argument("--runtime", choices=[x.value for x in RuntimeMode], default="auto")
    parser.add_argument("--interval", type=int, default=None)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    try:
        runtime = RuntimeMode.SCHEDULED if args.once else RuntimeMode(args.runtime)
        orchestrator = FXPAutonomousOrchestrator(runtime=runtime, interval=args.interval)
        orchestrator.start()
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        logger.critical("FXP Anomali Kritis Startup: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
