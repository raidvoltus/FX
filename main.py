"""
================================================================================
MODULE: main.py
DESCRIPTION: Production Orchestrator for OANDA Autonomous Forex Trading Bot
VERSION: 2026.1.0 (Forex Engine Architecture Edition)
PYTHON VERSION: 3.11+ / 3.12+
ARCHITECTURE: Clean Architecture, Decoupled Orchestration Pipeline
================================================================================
"""

import os
import sys
import time
import logging
import gc
import json
import traceback
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type

# =============================================================================
# 1. CORE SYSTEM CONFIGURATION & ARGUMENT PARSER
# =============================================================================
def _determine_execution_mode() -> str:
    if "--dry-run" in sys.argv:
        return "dry-run"
    if "--live" in sys.argv:
        return "live"
    if "--backtest-validation" in sys.argv:
        return "backtest-validation"
    
    return os.getenv("EXECUTION_MODE", "dry-run").lower().strip()


def _get_mode_suffix() -> str:
    mode = _determine_execution_mode()
    return "live" if mode == "live" else "dryrun"


def _get_mode_file_path(default_prefix: str, ext: str) -> str:
    suffix = _get_mode_suffix()
    return f"{default_prefix}_{suffix}.{ext}"


@dataclass
class ForexOrchestratorConfig:
    EXECUTION_MODE: str = field(default_factory=_determine_execution_mode)
    STATE_SUFFIX: str = field(default_factory=_get_mode_suffix)
    
    LOCK_FILE: str = field(default_factory=lambda: os.getenv("BOT_LOCK_FILE", _get_mode_file_path("forex_bot", "lock")))
    LOG_FILE: str = field(default_factory=lambda: os.getenv("BOT_LOG_FILE", _get_mode_file_path("forex_orchestrator", "log")))
    CHECKPOINT_FILE: str = field(default_factory=lambda: os.getenv("BOT_CHECKPOINT_FILE", _get_mode_file_path("forex_checkpoint", "json")))
    
    OANDA_CFG_PATH: str = os.getenv("OANDA_CFG_PATH", "oanda.cfg")
    LOOP_INTERVAL_SECONDS: int = int(os.getenv("LOOP_INTERVAL_SECONDS", "60"))
    
    # Circuit Breaker & Safety Safeguards
    CIRCUIT_FAILURE_THRESHOLD: int = 3
    CIRCUIT_RECOVERY_TIME_SEC: int = 60
    MAX_ACCOUNT_DRAWDOWN_PCT: float = float(os.getenv("MAX_ACCOUNT_DRAWDOWN_PCT", "5.0"))


# =============================================================================
# 2. INFRASTRUCTURE & RESILIENCE UTILITIES (CARRIED OVER & ADAPTED)
# =============================================================================
class ForexCircuitBreaker:
    """Melindungi bot dari eksekusi berulang jika API OANDA error berturut-turut."""
    def __init__(self, failure_threshold: int = 3, recovery_time_sec: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_time_sec = recovery_time_sec
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.state = "CLOSED" 

    def record_success(self):
        self.failure_count = 0
        self.state = "CLOSED"

    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"

    def can_execute(self) -> bool:
        if self.state == "CLOSED":
            return True
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_time_sec:
                self.state = "HALF-OPEN"
                return True
            return False
        return True


class ForexProcessLocker:
    """Memastikan hanya 1 instance Forex Bot yang berjalan dalam 1 server."""
    def __init__(self, lock_file: str):
        self.lock_file = lock_file
        self.fp = None

    def acquire(self) -> None:
        try:
            self.fp = open(self.lock_file, 'w')
            if os.name == 'posix':
                import fcntl
                fcntl.flock(self.fp, fcntl.LOCK_EX | fcntl.LOCK_NB)
            elif os.name == 'nt':
                import msvcrt
                msvcrt.locking(self.fp.fileno(), msvcrt.LK_NBLCK, 1)
        except (IOError, OSError) as e:
            raise PermissionError(f"Proses terkunci! Bot Forex instance lain sedang berjalan ({self.lock_file}). Detail: {e}")

    def release(self) -> None:
        if self.fp:
            try:
                if os.name == 'posix':
                    import fcntl
                    fcntl.flock(self.fp, fcntl.LOCK_UN)
                elif os.name == 'nt':
                    import msvcrt
                    msvcrt.locking(self.fp.fileno(), msvcrt.LK_UNLCK, 1)
            except Exception:
                pass
            finally:
                self.fp.close()
                if os.path.exists(self.lock_file):
                    try:
                        os.remove(self.lock_file)
                    except Exception:
                        pass


class ForexCheckpointManager:
    """Manajemen simpan/muat titik pemulihan eksekusi pipeline."""
    def __init__(self, filepath: str):
        self.filepath = filepath

    def save_checkpoint(self, step_number: int, step_name: str, execution_id: str):
        data_payload = {
            "execution_id": execution_id,
            "completed_step": step_number,
            "step_name": step_name,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        try:
            with open(self.filepath, "w") as f:
                json.dump(data_payload, f, indent=2)
        except Exception:
            pass

    def clear(self):
        if os.path.exists(self.filepath):
            try:
                os.remove(self.filepath)
            except Exception:
                pass


class ForexStepContext:
    """Context Manager observabilitas eksekusi per-step."""
    def __init__(self, step_number: int, step_name: str, logger: logging.Logger, critical: bool = True):
        self.step_number = step_number
        self.step_name = step_name
        self.logger = logger
        self.critical = critical
        self.start_time = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        self.logger.info(f"▶ [STEP {self.step_number}] Mula: {self.step_name}")
        return self

    def __exit__(self, exc_type: Optional[Type[BaseException]], exc_val: Optional[BaseException], exc_tb: Optional[Any]) -> bool:
        elapsed = time.perf_counter() - self.start_time
        if exc_type is None:
            self.logger.info(f"✔ [STEP {self.step_number}] Selesai: {self.step_name} ({elapsed:.4f}s)")
            return False
            
        self.logger.error(f"✖ [STEP {self.step_number} GAGAL] {self.step_name} ({elapsed:.4f}s). Error: {exc_val}")
        if exc_type in (KeyboardInterrupt, SystemExit):
            return False 
            
        if exc_type is MemoryError:
            self.logger.critical("⚠ [MEMORI PENUH] Menjalankan Garbage Collection Sweep...")
            gc.collect()
            return False 
            
        if self.critical:
            self.logger.critical(f"🛑 [FATAL] Tahap kritis {self.step_number} gagal. Menghentikan siklus.")
            self.logger.debug(traceback.format_exc())
            return False 
        else:
            self.logger.warning(f"🔀 [DEGRADASI] Tahap non-kritis {self.step_number} dilewati. Memakai fallback.")
            return True 


# =============================================================================
# 3. FOREX AUTONOMOUS ORCHESTRATOR ENGINE
# =============================================================================
class ForexAutonomousOrchestrator:
    """
    Production Controller Utama untuk Forex Trading Autonomous Bot berbasis OANDA.
    Mengintegrasikan modul-modul independen (broker, data, risk, ml, execution).
    """

    def __init__(self):
        self.config = ForexOrchestratorConfig()
        self.logger = self._setup_logging()
        self.locker = ForexProcessLocker(self.config.LOCK_FILE)
        self.circuit_breaker = ForexCircuitBreaker(
            failure_threshold=self.config.CIRCUIT_FAILURE_THRESHOLD,
            recovery_time_sec=self.config.CIRCUIT_RECOVERY_TIME_SEC
        )
        self.checkpoint_mgr = ForexCheckpointManager(self.config.CHECKPOINT_FILE)

        self.state: Dict[str, Any] = {
            "execution_id": f"EXEC-FX-{self.config.STATE_SUFFIX.upper()}-{int(time.time())}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "broker_connection_status": "UNKNOWN",
            "execution_gate_open": False,
            "execution_gate_reasons": []
        }

    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger("ForexAutonomousOrchestrator")
        if logger.handlers:
            return logger
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s')
        
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
        fh = logging.FileHandler(self.config.LOG_FILE)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
        return logger

    def run(self) -> None:
        """Loop eksekusi utama (Continuous Loop)."""
        self.logger.info("==================================================================")
        self.logger.info(f"🚀 OANDA FOREX AUTONOMOUS BOT STARTED IN [{self.config.EXECUTION_MODE.upper()}] MODE")
        self.logger.info(f"📁 Lock File: {self.config.LOCK_FILE} | Config: {self.config.OANDA_CFG_PATH}")
        self.logger.info("==================================================================")

        try:
            self.locker.acquire()
            
            # Continuous Autopilot Loop
            while True:
                self._execute_single_pipeline_cycle()
                self.logger.info(f"💤 Menunggu {self.config.LOOP_INTERVAL_SECONDS} detik untuk siklus berikutnya...\n")
                time.sleep(self.config.LOOP_INTERVAL_SECONDS)

        except (KeyboardInterrupt, SystemExit):
            self.logger.warning("Sinyal penghentian diterima. Menutup bot Forex secara aman...")
        except Exception as e:
            self.logger.critical(f"FATAL PIPELINE CRASH: {e}")
            self.logger.debug(traceback.format_exc())
        finally:
            self.locker.release()
            self.checkpoint_mgr.clear()
            self.logger.info("👋 Lock file dilepaskan. Bot Forex berhenti.")

    def _execute_single_pipeline_cycle(self) -> None:
        """Menjalankan 1 siklus alur kerja otonom penuh dari Data hingga Execution."""
        self.state["execution_id"] = f"EXEC-FX-{self.config.STATE_SUFFIX.upper()}-{int(time.time())}"
        self.state["timestamp"] = datetime.now(timezone.utc).isoformat()

        # Step 1: Broker Connection & Health Check
        with ForexStepContext(1, "OANDA Broker Connection & Sync", self.logger, critical=True):
            # TODO (Step 3): Inisialisasi OandaClient dari broker/oanda_client.py
            self.state["broker_connection_status"] = "CONNECTED"
            self.checkpoint_mgr.save_checkpoint(1, "BrokerSync", self.state['execution_id'])

        # Step 2: Discover & Filter Tradeable Forex Instruments
        with ForexStepContext(2, "Market Data Scanning & Pair Filtering", self.logger, critical=True):
            # TODO (Step 4): Memanggil MarketDataFetcher untuk filter Spread & Liquidity
            self.checkpoint_mgr.save_checkpoint(2, "MarketDataScan", self.state['execution_id'])

        # Step 3: Feature Engineering Calculation (Indicators & Lags)
        with ForexStepContext(3, "Feature Engineering & Indicator Calculation", self.logger, critical=True):
            # TODO (Step 5): FeatureEngine.build_features()
            self.checkpoint_mgr.save_checkpoint(3, "FeatureEngineering", self.state['execution_id'])

        # Step 4: ML Regime Detection & Strategy Selection
        with ForexStepContext(4, "ML Regime Classification & Strategy Selection", self.logger, critical=True):
            # TODO (Step 8): MLStrategySelector evaluate regime & select best strategy
            self.checkpoint_mgr.save_checkpoint(4, "StrategySelection", self.state['execution_id'])

        # Step 5: Risk Validation & Volatility Position Sizing (Deterministic)
        with ForexStepContext(5, "Risk Engine Audit & Position Sizing", self.logger, critical=True):
            # TODO (Step 6 & 7): RiskEngine & PositionSizer (ATR SL/TP calculation)
            self.checkpoint_mgr.save_checkpoint(5, "RiskManagement", self.state['execution_id'])

        # Step 6: Order Execution & Order Manager
        with ForexStepContext(6, "Order Execution & Gate Check", self.logger, critical=True):
            if self.config.EXECUTION_MODE == "live" and not self.circuit_breaker.can_execute():
                self.logger.warning("🛑 Circuit Breaker OPEN! Eksekusi order live diblokir.")
            else:
                # TODO (Step 11): OrderManager.execute_order()
                pass
            self.checkpoint_mgr.save_checkpoint(6, "OrderExecution", self.state['execution_id'])

        # Step 7: Portfolio Balance Update & Persistence State
        with ForexStepContext(7, "State Persistence & Portfolio Sync", self.logger, critical=False):
            # TODO (Step 12 & 13): StateStore.save_state()
            gc.collect()


# =============================================================================
# 4. ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    orchestrator = ForexAutonomousOrchestrator()
    orchestrator.run()
