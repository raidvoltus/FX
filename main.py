"""
================================================================================
PROJECT     : FXP - Forex Autonomous Trading Bot
REPOSITORY  : https://github.com/raidvoltus/FX
MODULE      : main.py
VERSION     : 2026.Q3.v4.0.0
PYTHON      : 3.11+
================================================================================

MULTI-RUNTIME PRODUCTION ENTRY POINT

Supported environments
----------------------
1. PC / Laptop
   - Windows
   - Linux
   - macOS
   - Persistent loop

2. VPS / Server
   - Linux
   - Persistent loop
   - Compatible with systemd / Docker / supervisor

3. GitHub Actions
   - Scheduled execution
   - One cycle per workflow run
   - Never keeps a hosted runner alive indefinitely

Runtime detection
-----------------
AUTO
 ├── GITHUB_ACTIONS=true → SCHEDULED
 └── otherwise            → PERSISTENT

Execution authority
-------------------
main.py
   ↓
AutonomousEngine
   ↓
market → features → ML → risk → execution → portfolio

main.py DOES NOT execute trades directly.
main.py is lifecycle/supervisor only.
================================================================================
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import socket
import sys
import time
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional

from config import Config
from autonomous_engine import AutonomousEngine


# =============================================================================
# LOGGING
# =============================================================================

logger = logging.getLogger("FXP.Main")

if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [FXP.Main] %(message)s"
        )
    )
    logger.addHandler(handler)

logger.setLevel(
    getattr(
        logging,
        os.getenv("LOG_LEVEL", "INFO").upper(),
        logging.INFO,
    )
)

logger.propagate = False


# =============================================================================
# ENUMS
# =============================================================================

class RuntimeMode(str, Enum):
    AUTO = "auto"
    PERSISTENT = "persistent"
    SCHEDULED = "scheduled"


VALID_EXECUTION_MODES = {
    "dry-run",
    "live",
    "force-rebalance",
}


# =============================================================================
# ENVIRONMENT DETECTION
# =============================================================================

def detect_runtime() -> RuntimeMode:
    """
    Detect execution environment.

    GitHub Actions:
        GITHUB_ACTIONS=true

    Everything else:
        persistent

    Rationale:
        Hosted GitHub runners are ephemeral and should not be used
        as permanent trading daemons.
    """

    github_actions = (
        os.getenv("GITHUB_ACTIONS", "").lower()
        == "true"
    )

    if github_actions:
        return RuntimeMode.SCHEDULED

    return RuntimeMode.PERSISTENT


# =============================================================================
# INSTANCE LOCK
# =============================================================================

class InstanceLock:
    """
    Lightweight local process lock.

    Tujuan:
        mencegah dua bot live berjalan bersamaan pada mesin yang sama.

    Catatan:
        Ini bukan distributed lock.
        Untuk multi-server deployment gunakan external lock/database.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        self.acquired = False

    def acquire(self) -> None:
        directory = os.path.dirname(self.path)

        if directory:
            os.makedirs(
                directory,
                exist_ok=True,
            )

        if os.path.exists(self.path):
            try:
                with open(
                    self.path,
                    "r",
                    encoding="utf-8",
                ) as fh:
                    previous_pid = fh.read().strip()
            except Exception:
                previous_pid = "unknown"

            raise RuntimeError(
                "Instance lock already exists: "
                f"{self.path}; previous PID={previous_pid}"
            )

        with open(
            self.path,
            "w",
            encoding="utf-8",
        ) as fh:
            fh.write(str(os.getpid()))

        self.acquired = True

    def release(self) -> None:
        if not self.acquired:
            return

        try:
            if os.path.exists(self.path):
                os.remove(self.path)
        except Exception as exc:
            logger.warning(
                "Gagal menghapus instance lock: %s",
                exc,
            )

        self.acquired = False


# =============================================================================
# ORCHESTRATOR
# =============================================================================

class FXPAutonomousOrchestrator:

    def __init__(
        self,
        runtime: RuntimeMode = RuntimeMode.AUTO,
        execution_mode: Optional[str] = None,
        interval: Optional[int] = None,
    ) -> None:

        self.is_running = False
        self.cycle_count = 0
        self.cleanup_done = False

        # ---------------------------------------------------------------------
        # Runtime
        # ---------------------------------------------------------------------

        if runtime == RuntimeMode.AUTO:
            self.runtime = detect_runtime()
        else:
            self.runtime = runtime

        # ---------------------------------------------------------------------
        # Config
        # ---------------------------------------------------------------------

        self.config = Config.load()

        self.execution_mode = (
            execution_mode
            or os.getenv("EXECUTION_MODE")
            or "dry-run"
        ).strip().lower()

        if self.execution_mode not in VALID_EXECUTION_MODES:
            raise ValueError(
                f"Invalid EXECUTION_MODE={self.execution_mode}. "
                f"Allowed={sorted(VALID_EXECUTION_MODES)}"
            )

        self.interval = max(
            1,
            int(
                interval
                if interval is not None
                else getattr(
                    self.config,
                    "LOOP_INTERVAL_SECONDS",
                    60,
                )
            ),
        )

        # ---------------------------------------------------------------------
        # Lock
        # ---------------------------------------------------------------------

        default_lock = (
            "/tmp/fxp_live.lock"
            if os.name != "nt"
            else os.path.join(
                os.getenv("TEMP", "."),
                "fxp_live.lock",
            )
        )

        self.lock_path = os.getenv(
            "FXP_LOCK_FILE",
            default_lock,
        )

        self.lock = InstanceLock(
            self.lock_path
        )

        # ---------------------------------------------------------------------
        # Engine
        # ---------------------------------------------------------------------

        logger.info(
            "Initializing AutonomousEngine..."
        )

        self.engine = AutonomousEngine(
            mode=self.execution_mode
        )

        # ---------------------------------------------------------------------
        # Signal handlers
        # ---------------------------------------------------------------------

        self._install_signal_handlers()

        self._startup_log()

    # =========================================================================
    # STARTUP
    # =========================================================================

    def _startup_log(self) -> None:

        logger.info("=" * 78)
        logger.info(
            "FXP AUTONOMOUS TRADING ENGINE"
        )
        logger.info("=" * 78)

        logger.info(
            "Runtime          : %s",
            self.runtime.value.upper(),
        )

        logger.info(
            "Execution Mode   : %s",
            self.execution_mode.upper(),
        )

        logger.info(
            "Hostname         : %s",
            socket.gethostname(),
        )

        logger.info(
            "PID              : %s",
            os.getpid(),
        )

        logger.info(
            "Python           : %s.%s",
            sys.version_info.major,
            sys.version_info.minor,
        )

        logger.info(
            "Loop Interval    : %s sec",
            self.interval,
        )

        logger.info(
            "GitHub Actions   : %s",
            os.getenv(
                "GITHUB_ACTIONS",
                "false",
            ),
        )

        logger.info(
            "UTC Start        : %s",
            datetime.now(
                timezone.utc
            ).isoformat(),
        )

        logger.info("=" * 78)

    # =========================================================================
    # SIGNAL
    # =========================================================================

    def _install_signal_handlers(self) -> None:

        try:
            signal.signal(
                signal.SIGINT,
                self._shutdown_signal,
            )

            signal.signal(
                signal.SIGTERM,
                self._shutdown_signal,
            )

        except ValueError:
            logger.warning(
                "OS signal handler unavailable."
            )

    def _shutdown_signal(
        self,
        signum: int,
        frame: Any,
    ) -> None:

        try:
            name = signal.Signals(
                signum
            ).name
        except Exception:
            name = str(signum)

        logger.warning(
            "Shutdown signal received: %s",
            name,
        )

        self.is_running = False

    # =========================================================================
    # SINGLE CYCLE
    # =========================================================================

    def run_cycle(self) -> Dict[str, Any]:

        cycle_id = self.cycle_count + 1

        started = time.perf_counter()

        logger.info(
            "--------------------------------------------------"
        )

        logger.info(
            "CYCLE #%s START | runtime=%s | mode=%s",
            cycle_id,
            self.runtime.value,
            self.execution_mode,
        )

        try:

            result = self.engine.run_cycle()

            elapsed = (
                time.perf_counter()
                - started
            ) * 1000.0

            if not isinstance(
                result,
                dict,
            ):
                result = {
                    "status": "ERROR",
                    "reason": (
                        "AutonomousEngine returned "
                        f"{type(result).__name__}"
                    ),
                }

            status = str(
                result.get(
                    "status",
                    "UNKNOWN",
                )
            ).upper()

            orders = result.get(
                "executed_orders",
                [],
            )

            if not isinstance(
                orders,
                list,
            ):
                orders = []

            logger.info(
                "CYCLE #%s END | status=%s | "
                "orders=%s | %.2f ms",
                cycle_id,
                status,
                len(orders),
                elapsed,
            )

            self.cycle_count = cycle_id

            result.update(
                {
                    "cycle_id": cycle_id,
                    "runtime": self.runtime.value,
                    "execution_mode": self.execution_mode,
                    "orchestrator_latency_ms": elapsed,
                    "timestamp": datetime.now(
                        timezone.utc
                    ).isoformat(),
                }
            )

            return result

        except Exception as exc:

            elapsed = (
                time.perf_counter()
                - started
            ) * 1000.0

            self.cycle_count = cycle_id

            logger.error(
                "CYCLE #%s FAILED | %.2f ms | %s",
                cycle_id,
                elapsed,
                exc,
                exc_info=True,
            )

            return {
                "status": "ERROR",
                "cycle_id": cycle_id,
                "runtime": self.runtime.value,
                "execution_mode": self.execution_mode,
                "reason": str(exc),
                "orchestrator_latency_ms": elapsed,
            }

    # =========================================================================
    # START
    # =========================================================================

    def start(self) -> None:

        self.lock.acquire()

        try:

            if self.runtime == RuntimeMode.SCHEDULED:

                logger.info(
                    "SCHEDULED MODE: running exactly one cycle."
                )

                self.run_cycle()

                return

            self.is_running = True

            logger.info(
                "PERSISTENT MODE: autonomous loop active."
            )

            while self.is_running:

                self.run_cycle()

                if not self.is_running:
                    break

                self._sleep_interruptible(
                    self.interval
                )

        finally:

            self.cleanup()

    # =========================================================================
    # INTERRUPTIBLE SLEEP
    # =========================================================================

    def _sleep_interruptible(
        self,
        seconds: int,
    ) -> None:

        deadline = (
            time.monotonic()
            + seconds
        )

        while self.is_running:

            remaining = (
                deadline
                - time.monotonic()
            )

            if remaining <= 0:
                return

            time.sleep(
                min(
                    1.0,
                    remaining,
                )
            )

    # =========================================================================
    # CLEANUP
    # =========================================================================

    def cleanup(self) -> None:

        if self.cleanup_done:
            return

        self.cleanup_done = True
        self.is_running = False

        logger.info(
            "Starting FXP cleanup..."
        )

        # Future compatible shutdown API.
        for method_name in (
            "shutdown",
            "close",
            "stop",
        ):

            method = getattr(
                self.engine,
                method_name,
                None,
            )

            if callable(method):

                try:
                    logger.info(
                        "Calling engine.%s()",
                        method_name,
                    )

                    method()

                except Exception as exc:

                    logger.error(
                        "Engine cleanup failed: %s",
                        exc,
                        exc_info=True,
                    )

                break

        self.lock.release()

        logger.info(
            "FXP stopped. cycles=%s",
            self.cycle_count,
        )


# =============================================================================
# CLI
# =============================================================================

def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description=(
            "FXP Multi-Environment Autonomous "
            "Forex Trading Bot"
        )
    )

    parser.add_argument(
        "--runtime",
        choices=[
            x.value
            for x in RuntimeMode
        ],
        default="auto",
        help=(
            "Runtime environment. "
            "auto detects GitHub Actions."
        ),
    )

    parser.add_argument(
        "--mode",
        choices=sorted(
            VALID_EXECUTION_MODES
        ),
        default=None,
        help="Trading execution mode.",
    )

    parser.add_argument(
        "--interval",
        type=int,
        default=None,
        help=(
            "Persistent loop interval "
            "in seconds."
        ),
    )

    parser.add_argument(
        "--once",
        action="store_true",
        help=(
            "Run one cycle and exit."
        ),
    )

    return parser


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:

    parser = build_parser()

    args = parser.parse_args()

    try:

        runtime = RuntimeMode(
            args.runtime
        )

        # --once always forces scheduled behavior.
        if args.once:
            runtime = RuntimeMode.SCHEDULED

        orchestrator = (
            FXPAutonomousOrchestrator(
                runtime=runtime,
                execution_mode=args.mode,
                interval=args.interval,
            )
        )

        orchestrator.start()

        return 0

    except KeyboardInterrupt:

        logger.warning(
            "Interrupted by user."
        )

        return 130

    except Exception as exc:

        logger.critical(
            "FXP fatal startup/runtime error: %s",
            exc,
            exc_info=True,
        )

        return 1


if __name__ == "__main__":
    sys.exit(main())