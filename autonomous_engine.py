"""
================================================================================
PROJECT     : FXP (Forex Autonomous Trading Bot)
REPOSITORY  : https://github.com/raidvoltus/FX
MODULE      : autonomous_engine.py
VERSION     : 2026.Q3.v4.0.0 (Execution Authority Edition)
PYTHON      : 3.11+
================================================================================
"""

import os
import sys
import time
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

import polars as pl

from config import Config
from broker.oanda_client import OandaClient
from data.market_data import MarketDataFetcher
from data.feature_engine import FeatureEngineFacade
from risk.risk_engine import RiskEngine
from risk.position_sizer import PositionSizer
from ml.strategy_selector import MLStrategySelector
from ml.MLClassificationLive import MLClassificationLive
from execution.order_manager import OrderManager
from portfolio.portfolio_manager import PortfolioManager
from persistence.state_store import StateStore
from monitoring.health_monitor import HealthMonitor
from signal_forex import UnifiedForexSignalEngine
from storage import UnifiedStorageEngine

logger = logging.getLogger("FXP.AutonomousEngine")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] [AutonomousEngine]: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class AutonomousEngine:

    def __init__(self, mode: Optional[str] = None):
        self.config = Config.load()
        self.mode = str(mode or os.getenv("EXECUTION_MODE", "dry-run")).lower().strip()
        self.is_live = (self.mode == "live")

        logger.info(f"Inisialisasi AutonomousEngine | Mode: {self.mode.upper()}")

        # Dynamic Configurations
        self.granularity = str(getattr(self.config, "DEFAULT_GRANULARITY", "M5"))
        self.candle_count = int(getattr(self.config, "HISTORICAL_CANDLES_COUNT", 500))

        # Infrastructure & Core Services
        self.broker = OandaClient(environment="practice" if not self.is_live else "live")
        self.data_fetcher = MarketDataFetcher(self.broker)
        self.feature_engine = FeatureEngineFacade()

        # Analytics, ML, & 9-Gateway Pre-Execution Filter
        self.strategy_selector = MLStrategySelector()
        self.ml_classifier = MLClassificationLive()
        self.signal_engine = UnifiedForexSignalEngine()

        # Risk & Sizing
        self.risk_engine = RiskEngine()
        self.position_sizer = PositionSizer()

        # Execution, State, & Storage
        self.order_manager = OrderManager(self.broker)
        self.portfolio_manager = PortfolioManager(self.broker, mode=self.mode)
        self.state_store = StateStore()
        self.storage_engine = UnifiedStorageEngine(mode=self.mode)
        self.health_monitor = HealthMonitor()

    def run_cycle(self) -> Dict[str, Any]:
        start_time = time.perf_counter()

        # 1. Health Diagnostics Ping
        health_report = self.health_monitor.ping()
        if health_report.get("system_status") == "UNHEALTHY":
            logger.critical("Status kesehatan sistem UNHEALTHY. Siklus dihentikan.")
            return {"status": "HALTED", "reason": "SYSTEM_UNHEALTHY"}

        # 2. Filter Tradeable Instruments
        try:
            available_instruments = self.broker.api.get_instruments()
            active_instruments = self.data_fetcher.filter_tradeable_instruments(available_instruments)
        except Exception as e:
            logger.error(f"Gagal memfilter instrumen OANDA: {e}")
            active_instruments = []

        if not active_instruments:
            return {"status": "SKIPPED", "reason": "NO_TRADEABLE_INSTRUMENTS"}

        # 3. Fetch Portfolio Summary
        portfolio_summary = self.portfolio_manager.get_summary()
        account_balance = float(portfolio_summary.get("balance", 0.0))

        executed_orders: List[Dict[str, Any]] = []

        # 4. Pair Evaluation Loop
        for instrument in active_instruments:
            df_candles = self.data_fetcher.get_candles(
                instrument, 
                granularity=self.granularity, 
                count=self.candle_count
            )
            if df_candles.height == 0:
                continue

            df_features = self.feature_engine.build_features(df_candles)

            selected_strategy, confidence = self.strategy_selector.evaluate(df_features)
            if selected_strategy == "hold":
                continue

            signal = self.ml_classifier.generate_signal(df_features)
            if signal.direction not in ["BUY", "SELL"]:
                continue

            entry_price = float(df_candles["close"][-1])
            sl_raw, tp_raw = self.risk_engine.calculate_stop_levels(df_features, signal)

            # PRE-EXECUTION GATEWAY: 9-Gateway Pipeline Filtering
            pred_payload = pl.DataFrame({
                "instrument": [instrument],
                "close": [entry_price],
                "calibrated_prob": [getattr(signal, "probability", 0.65)],
                "confidence_score": [getattr(signal, "confidence", 0.65)],
                "predicted_return": [getattr(signal, "expected_return", 0.002)],
                "stop_loss": [sl_raw],
                "take_profit": [tp_raw],
                "signal_direction": [signal.direction]
            })

            gateway_df = self.signal_engine.execute_pipeline(pred_payload)

            if gateway_df.is_empty() or not bool(gateway_df["is_valid_execution"][0]):
                reason = str(gateway_df["final_validator_reason"][0]) if not gateway_df.is_empty() else "GATEWAY_REJECTED"
                logger.warning(f"[{instrument}] Sinyal ditolak 9-Gateway Filter: {reason}")
                continue

            opt_tp = float(gateway_df["optimized_take_profit"][0])
            opt_sl = float(gateway_df["optimized_stop_loss"][0])
            final_direction = str(gateway_df["candidate_signal"][0])

            # Position Sizing & Risk Engine Check
            units = self.position_sizer.calculate_units(
                account_balance=account_balance,
                entry_price=entry_price,
                stop_loss_price=opt_sl,
                instrument=instrument
            )

            is_valid, reason = self.risk_engine.validate_trade(instrument, units, portfolio_summary)
            if not is_valid:
                logger.warning(f"[{instrument}] Order ditolak Risk Engine: {reason}")
                continue

            # Execution Gate
            if self.is_live:
                exec_res = self.order_manager.execute_order(
                    instrument=instrument,
                    units=units,
                    order_type=final_direction,
                    stop_loss=opt_sl,
                    take_profit=opt_tp
                )
                executed_orders.append(exec_res)
            else:
                sim_res = {
                    "status": "SIMULATED",
                    "instrument": instrument,
                    "direction": final_direction,
                    "units": units,
                    "entry_price": entry_price,
                    "stop_loss": opt_sl,
                    "take_profit": opt_tp,
                    "timestamp": time.time()
                }
                executed_orders.append(sim_res)

            # Persist Signal Log to Storage
            self.storage_engine.persist_signals(gateway_df)

        # 5. Save System State & Portfolio Persistence
        updated_summary = self.portfolio_manager.get_summary()
        self.portfolio_manager.save_state(updated_summary)
        self.state_store.save_state(updated_summary)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return {
            "status": "SUCCESS",
            "execution_mode": self.mode,
            "granularity": self.granularity,
            "candles_count": self.candle_count,
            "executed_orders_count": len(executed_orders),
            "executed_orders": executed_orders,
            "latency_ms": elapsed_ms,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
