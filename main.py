"""
================================================================================
PROJECT     : FXP (Forex Autonomous Trading Bot)
REPOSITORY  : https://github.com/raidvoltus/FX
MODULE      : main.py
DESCRIPTION : Production Entry Point & Autonomous Loop Orchestrator.
VERSION     : 2026.Q3.v3.1.0
PYTHON      : 3.11+ / 3.12+
ARCHITECTURE:
    Config
      ↓
    AutonomousEngine
      ├── Broker / OANDA
      ├── Market Data
      ├── Feature Engineering
      ├── ML Strategy Selection
      ├── ML Direction Signal
      ├── Risk Engine
      ├── Position Sizing
      ├── Order Execution
      ├── Portfolio Management
      ├── State Persistence
      └── Health Monitoring
      ↓
    Main Orchestrator
      ├── Cycle supervision
      ├── Audit logging
      ├── Failure isolation
      └── Graceful shutdown

IMPORTANT:
- AutonomousEngine adalah execution authority.
- main.py TIDAK melakukan re-validation setelah order dieksekusi.
- 9-Gateway signal engine tidak dipanggil sebagai post-execution filter.
- Evaluation / Validation / Self-Learning dijalankan oleh pipeline terpisah
  ketika dataset yang diperlukan tersedia.
================================================================================
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from config import Config
from autonomous_engine import AutonomousEngine


# ==============================================================================
# LOGGER
# ==============================================================================

LOGGER_NAME = "FXP.MainOrchestrator"

logger = logging.getLogger(LOGGER_NAME)

if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [FXP.Main]: %(message)s"
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


# ==============================================================================
# CONSTANTS
# ==============================================================================

VALID_EXECUTION_MODES = {
    "dry-run",
    "live",
    "force-rebalance",
}

DEFAULT_EXECUTION_MODE = "dry-run"
DEFAULT_LOOP_INTERVAL_SECONDS = 60


# ==============================================================================
# ORCHESTRATOR
# ==============================================================================

class FXPAutonomousOrchestrator:
    """
    Production supervisor untuk AutonomousEngine.

    Tanggung jawab:
    1. Load configuration.
    2. Menentukan execution mode.
    3. Membuat AutonomousEngine.
    4. Menjalankan autonomous cycle.
    5. Menangani shutdown.
    6. Menjaga loop tetap hidup jika terjadi exception.
    7. Menghasilkan audit log yang konsisten.

    AutonomousEngine tetap menjadi satu-satunya authority untuk:
        market → signal → risk → sizing → execution → portfolio.
    """

    def __init__(self, execution_mode: Optional[str] = None) -> None:
        self.is_running = False
        self.cycle_count = 0

        # ------------------------------------------------------------------
        # Configuration
        # ------------------------------------------------------------------

        try:
            self.config = Config.load()
        except Exception as exc:
            logger.critical(
                "Gagal memuat Config: %s",
                exc,
                exc_info=True,
            )
            raise

        configured_mode = (
            execution_mode
            or os.getenv("EXECUTION_MODE")
            or os.getenv("TRADING_MODE")
            or DEFAULT_EXECUTION_MODE
        )

        self.execution_mode = str(configured_mode).strip().lower()

        if self.execution_mode not in VALID_EXECUTION_MODES:
            raise ValueError(
                f"EXECUTION_MODE tidak valid: '{self.execution_mode}'. "
                f"Pilihan: {sorted(VALID_EXECUTION_MODES)}"
            )

        # Config aktual repository menggunakan LOOP_INTERVAL_SECONDS.
        self.loop_interval = max(
            1,
            int(
                getattr(
                    self.config,
                    "LOOP_INTERVAL_SECONDS",
                    DEFAULT_LOOP_INTERVAL_SECONDS,
                )
            ),
        )

        self.graceful_shutdown_timeout = max(
            1,
            int(os.getenv("SHUTDOWN_TIMEOUT_SECONDS", "10")),
        )

        # ------------------------------------------------------------------
        # Runtime metadata
        # ------------------------------------------------------------------

        self.started_at = datetime.now(timezone.utc)

        # ------------------------------------------------------------------
        # Autonomous Engine
        # ------------------------------------------------------------------

        try:
            self.autonomous_engine = AutonomousEngine(
                mode=self.execution_mode
            )
        except Exception as exc:
            logger.critical(
                "Gagal menginisialisasi AutonomousEngine: %s",
                exc,
                exc_info=True,
            )
            raise

        # ------------------------------------------------------------------
        # OS signal handlers
        # ------------------------------------------------------------------

        self._install_signal_handlers()

        self._log_startup_banner()

    # ==========================================================================
    # SIGNAL HANDLING
    # ==========================================================================

    def _install_signal_handlers(self) -> None:
        """
        Install SIGINT/SIGTERM handler.

        SIGTERM:
            Docker / GitHub runner / process manager shutdown.

        SIGINT:
            Ctrl+C / manual shutdown.
        """

        try:
            signal.signal(signal.SIGINT, self._handle_shutdown)
            signal.signal(signal.SIGTERM, self._handle_shutdown)
        except ValueError:
            # Signal handlers can only be installed from the main thread.
            logger.warning(
                "Signal handler tidak dapat dipasang karena proses "
                "bukan berjalan pada main thread."
            )

    def _handle_shutdown(
        self,
        signum: int,
        frame: Any,
    ) -> None:
        """
        Request graceful shutdown.

        Handler tidak melakukan cleanup berat secara langsung.
        Handler hanya mengubah state sehingga loop dapat berhenti
        pada safe boundary.
        """

        signal_name = getattr(
            signal.Signals(signum),
            "name",
            str(signum),
        )

        logger.warning(
            "Shutdown signal diterima: %s (%s). "
            "Menghentikan autonomous loop pada safe boundary.",
            signal_name,
            signum,
        )

        self.is_running = False

    # ==========================================================================
    # LOGGING
    # ==========================================================================

    def _log_startup_banner(self) -> None:
        mode_label = self.execution_mode.upper()

        logger.info("=" * 78)
        logger.info("FXP FOREX AUTONOMOUS TRADING ENGINE")
        logger.info("=" * 78)
        logger.info(
            "Repository       : https://github.com/raidvoltus/FX"
        )
        logger.info(
            "Execution Mode   : %s",
            mode_label,
        )
        logger.info(
            "Loop Interval    : %s seconds",
            self.loop_interval,
        )
        logger.info(
            "Default TF       : %s",
            getattr(
                self.config,
                "DEFAULT_GRANULARITY",
                "M5",
            ),
        )
        logger.info(
            "Historical Bars  : %s",
            getattr(
                self.config,
                "HISTORICAL_CANDLES_COUNT",
                500,
            ),
        )
        logger.info(
            "Max Risk/Trade   : %.4f%%",
            float(
                getattr(
                    self.config,
                    "MAX_RISK_PER_TRADE_PCT",
                    0.01,
                )
            )
            * 100.0,
        )
        logger.info(
            "Max Portfolio Risk: %.4f%%",
            float(
                getattr(
                    self.config,
                    "MAX_PORTFOLIO_RISK_PCT",
                    0.03,
                )
            )
            * 100.0,
        )
        logger.info(
            "Max Open Positions: %s",
            getattr(
                self.config,
                "MAX_OPEN_POSITIONS",
                3,
            ),
        )
        logger.info(
            "Started At       : %s",
            self.started_at.isoformat(),
        )
        logger.info("=" * 78)

    # ==========================================================================
    # SINGLE CYCLE
    # ==========================================================================

    def run_single_cycle(self) -> Dict[str, Any]:
        """
        Execute satu autonomous cycle.

        AutonomousEngine merupakan authority utama.

        Return schema dinormalisasi agar caller tidak bergantung
        pada implementation detail internal engine.
        """

        cycle_id = self.cycle_count + 1
        cycle_started = time.perf_counter()

        timestamp = datetime.now(timezone.utc).isoformat()

        logger.info(
            "[Cycle %s] START | mode=%s | timestamp=%s",
            cycle_id,
            self.execution_mode.upper(),
            timestamp,
        )

        try:
            result = self.autonomous_engine.run_cycle()

            if not isinstance(result, dict):
                logger.error(
                    "[Cycle %s] AutonomousEngine mengembalikan "
                    "object non-dict: %s",
                    cycle_id,
                    type(result).__name__,
                )

                return {
                    "status": "ERROR",
                    "cycle_id": cycle_id,
                    "reason": "INVALID_ENGINE_RESPONSE",
                }

            elapsed_ms = (
                time.perf_counter() - cycle_started
            ) * 1000.0

            engine_status = str(
                result.get("status", "UNKNOWN")
            ).upper()

            executed_orders = result.get(
                "executed_orders",
                [],
            )

            if not isinstance(executed_orders, list):
                executed_orders = []

            executed_count = len(executed_orders)

            if engine_status == "SUCCESS":
                logger.info(
                    "[Cycle %s] SUCCESS | orders=%s | latency=%.2f ms",
                    cycle_id,
                    executed_count,
                    elapsed_ms,
                )

            elif engine_status in {
                "SKIPPED",
                "HALTED",
            }:
                logger.warning(
                    "[Cycle %s] %s | reason=%s | latency=%.2f ms",
                    cycle_id,
                    engine_status,
                    result.get("reason", "UNKNOWN"),
                    elapsed_ms,
                )

            else:
                logger.error(
                    "[Cycle %s] Engine status=%s | reason=%s | "
                    "latency=%.2f ms",
                    cycle_id,
                    engine_status,
                    result.get("reason", "UNKNOWN"),
                    elapsed_ms,
                )

            normalized_result = dict(result)

            normalized_result.update(
                {
                    "cycle_id": cycle_id,
                    "orchestrator_latency_ms": elapsed_ms,
                    "orchestrator_timestamp": timestamp,
                    "execution_mode": self.execution_mode,
                }
            )

            self.cycle_count = cycle_id

            return normalized_result

        except Exception as exc:
            elapsed_ms = (
                time.perf_counter() - cycle_started
            ) * 1000.0

            logger.error(
                "[Cycle %s] UNHANDLED EXCEPTION | %.2f ms | %s",
                cycle_id,
                elapsed_ms,
                exc,
                exc_info=True,
            )

            self.cycle_count = cycle_id

            return {
                "status": "ERROR",
                "cycle_id": cycle_id,
                "execution_mode": self.execution_mode,
                "reason": str(exc),
                "orchestrator_latency_ms": elapsed_ms,
                "orchestrator_timestamp": timestamp,
            }

    # ==========================================================================
    # LOOP
    # ==========================================================================

    def start_loop(self, run_once: bool = False) -> None:
        """
        Main autonomous supervisor loop.

        --once:
            Satu cycle → cleanup → exit.

        Normal:
            cycle → interruptible wait → cycle → ...
        """

        if run_once:
            logger.info(
                "Run-once mode aktif. Hanya satu autonomous cycle."
            )

            try:
                self.run_single_cycle()
            finally:
                self._cleanup()

            return

        self.is_running = True

        logger.info(
            "Autonomous loop aktif. Interval=%s seconds.",
            self.loop_interval,
        )

        try:
            while self.is_running:
                self.run_single_cycle()

                if not self.is_running:
                    break

                self._interruptible_sleep(
                    self.loop_interval
                )

        except KeyboardInterrupt:
            logger.warning(
                "KeyboardInterrupt diterima."
            )
            self.is_running = False

        except Exception as exc:
            logger.critical(
                "Supervisor loop mengalami exception fatal: %s",
                exc,
                exc_info=True,
            )
            self.is_running = False

        finally:
            self._cleanup()

    # ==========================================================================
    # INTERRUPTIBLE SLEEP
    # ==========================================================================

    def _interruptible_sleep(self, seconds: int) -> None:
        """
        Sleep yang tetap responsif terhadap SIGINT/SIGTERM.
        """

        deadline = time.monotonic() + max(0, seconds)

        while self.is_running:
            remaining = deadline - time.monotonic()

            if remaining <= 0:
                break

            time.sleep(min(1.0, remaining))

    # ==========================================================================
    # CLEANUP
    # ==========================================================================

    def _cleanup(self) -> None:
        """
        Graceful shutdown.

        AutonomousEngine pada repository saat ini tidak memiliki
        public shutdown()/close() contract. Karena itu cleanup utama
        berada pada supervisor state, tanpa memanggil method yang
        belum tersedia.
        """

        if getattr(self, "_cleanup_done", False):
            return

        self._cleanup_done = True
        self.is_running = False

        logger.info(
            "Memulai graceful shutdown..."
        )

        # ------------------------------------------------------------------
        # Future-compatible cleanup hooks.
        #
        # Jika AutonomousEngine nantinya memiliki close()/shutdown(),
        # method tersebut otomatis akan dipanggil.
        # ------------------------------------------------------------------

        for method_name in (
            "shutdown",
            "close",
            "stop",
        ):
            method = getattr(
                self.autonomous_engine,
                method_name,
                None,
            )

            if callable(method):
                try:
                    logger.info(
                        "Memanggil AutonomousEngine.%s()",
                        method_name,
                    )
                    method()
                except Exception as exc:
                    logger.error(
                        "Cleanup %s() gagal: %s",
                        method_name,
                        exc,
                        exc_info=True,
                    )
                break

        logger.info(
            "FXP Autonomous Trading Engine stopped. "
            "Total cycles=%s.",
            self.cycle_count,
        )


# ==============================================================================
# CLI
# ==============================================================================

def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "FXP OANDA Forex Autonomous Trading Bot "
            "Production Orchestrator"
        )
    )

    parser.add_argument(
        "--once",
        action="store_true",
        help=(
            "Run exactly one autonomous cycle and exit."
        ),
    )

    parser.add_argument(
        "--mode",
        choices=sorted(VALID_EXECUTION_MODES),
        default=None,
        help=(
            "Override EXECUTION_MODE for this process."
        ),
    )

    parser.add_argument(
        "--interval",
        type=int,
        default=None,
        help=(
            "Override LOOP_INTERVAL_SECONDS for this process."
        ),
    )

    return parser


# ==============================================================================
# MAIN
# ==============================================================================

def main() -> int:
    parser = build_argument_parser()
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # CLI interval override
    # ------------------------------------------------------------------

    if args.interval is not None:
        if args.interval < 1:
            parser.error(
                "--interval harus >= 1 detik."
            )

        os.environ["LOOP_INTERVAL_SECONDS"] = str(
            args.interval
        )

    # ------------------------------------------------------------------
    # CLI mode override
    # ------------------------------------------------------------------

    if args.mode is not None:
        os.environ["EXECUTION_MODE"] = args.mode

    try:
        orchestrator = FXPAutonomousOrchestrator(
            execution_mode=args.mode
        )

        orchestrator.start_loop(
            run_once=args.once
        )

        return 0

    except KeyboardInterrupt:
        logger.warning(
            "Process interrupted by user."
        )
        return 130

    except Exception as exc:
        logger.critical(
            "FXP startup failure: %s",
            exc,
            exc_info=True,
        )
        return 1


# ==============================================================================
# PROCESS ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    sys.exit(main())