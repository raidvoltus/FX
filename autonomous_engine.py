"""
================================================================================
PROJECT     : FXP (Forex Autonomous Trading Bot)
REPOSITORY  : https://github.com/raidvoltus/FX
MODULE      : autonomous_engine.py
DESCRIPTION : Core Autonomous Execution Engine & Pipeline Authority.
VERSION     : 2026.Q3.v3.2.0 (Pre-Execution 9-Gateway & Dynamic Config Edition)
PYTHON      : 3.11+ / 3.12+
ARCHITECTURE: Clean Architecture, Decoupled Event Loop, Pre-Execution Gatekeeper
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

# Configuration & Logging Setup
logger = logging.getLogger("FXP.AutonomousEngine")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [AutonomousEngine]: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class AutonomousEngine:
    """
    Engine Pengendali Utama Execution Authority.
    Mengoordinasikan alur:
    Market Ingestion -> Features -> ML Model -> 9-Gateway Filter -> Risk Engine -> Position Sizer -> Execution.
    """

    def __init__(self, mode: Optional[str] = None):
        self.config = Config.load()
        self.mode = str(mode or os.getenv("EXECUTION_MODE", "dry-run")).lower().strip()
        self.is_live = (self.mode == "live")

        logger.info(f"🚀 Inisialisasi AutonomousEngine dalam mode: {self.mode.upper()}")

        # 1. Dynamic Environment Configurations
        self.granularity = str(getattr(self.config, "DEFAULT_GRANULARITY", "M5"))
        self.candle_count = int(getattr(self.config, "HISTORICAL_CANDLES_COUNT", 500))

        # 2. Broker & Data Ingestion Layer
        self.broker = OandaClient(environment="practice" if not self.is_live else "live")
        self.data_fetcher = MarketDataFetcher(self.broker)
        self.feature_engine = FeatureEngineFacade()

        # 3. Analytics, ML & Pre-Execution Gatekeeping Layer
        self.strategy_selector = MLStrategySelector()
        self.ml_classifier = MLClassificationLive()
        self.signal_engine = UnifiedForexSignalEngine()  # 9-Gateway Pre-Execution Gatekeeper

        # 4. Risk & Position Sizing Layer
        self.risk_engine = RiskEngine()
        self.position_sizer = PositionSizer()

        # 5. Execution, Portfolio & State Layer
        self.order_manager = OrderManager(self.broker)
        self.portfolio_manager = PortfolioManager(self.broker, mode=self.mode)
        self.state_store = StateStore()
        self.health_monitor = HealthMonitor()

    def run_cycle(self) -> Dict[str, Any]:
        """
        Menjalankan 1 Siklus Autonom Lengkap dengan Pre-Execution Validation Gate.
        """
        start_time = time.perf_counter()
        logger.info("==================================================")
        logger.info(f"🔄 MEMULAI SIKLUS AUTONOM FOREX (Mode: {self.mode.upper()})")
        logger.info(f"⚙️ Config: Granularity={self.granularity} | Candles Count={self.candle_count}")
        logger.info("==================================================")

        # Step 1: Health Diagnostics Ping
        health_report = self.health_monitor.ping()
        if health_report.get("system_status") == "UNHEALTHY":
            logger.critical("🚨 Status kesehatan sistem UNHEALTHY. Siklus trading dihentikan sementara.")
            return {"status": "HALTED", "reason": "SYSTEM_UNHEALTHY"}

        # Step 2: Filter Tradeable Instruments (Spread & Volatility Filter)
        try:
            available_instruments = self.broker.api.get_instruments()
            active_instruments = self.data_fetcher.filter_tradeable_instruments(available_instruments)
        except Exception as e:
            logger.error(f"❌ Gagal mengambil/memfilter instrumen OANDA: {e}")
            active_instruments = []

        if not active_instruments:
            logger.warning("⚠️ Tidak ada pasangan Forex yang lolos filter spread saat ini.")
            return {"status": "SKIPPED", "reason": "NO_TRADEABLE_INSTRUMENTS"}

        # Step 3: Fetch Account Balance & Active Portfolio Summary
        portfolio_summary = self.portfolio_manager.get_summary()
        account_balance = float(portfolio_summary.get("balance", 0.0))

        executed_orders: List[Dict[str, Any]] = []
        cycle_telemetry: List[Dict[str, Any]] = []

        # Step 4: Iterasi Per Pasangan Mata Uang (Pair Evaluation Loop)
        for instrument in active_instruments:
            logger.info(f"\n--- [MEMPROSES INSTRUMEN: {instrument}] ---")

            # 4a. Fetch Candle Data (Dynamic Config)
            df_candles = self.data_fetcher.get_candles(
                instrument, 
                granularity=self.granularity, 
                count=self.candle_count
            )
            if df_candles.height == 0:
                logger.warning(f"⚠️ [{instrument}] Data candle kosong, melewati instrumen.")
                continue

            # 4b. Technical Indicators & Feature Extraction
            df_features = self.feature_engine.build_features(df_candles)

            # 4c. ML Strategy Selector & Market Regime Detection
            selected_strategy, confidence = self.strategy_selector.evaluate(df_features)
            if selected_strategy == "hold":
                logger.info(f"⏸ [{instrument}] Rejim pasar tidak terarah / Noise. Strategi memilih HOLD.")
                continue

            # 4d. ML Directional Signal Model Generation
            signal = self.ml_classifier.generate_signal(df_features)
            if signal.direction not in ["BUY", "SELL"]:
                logger.info(f"⏸ [{instrument}] Sinyal ML merekomendasikan HOLD.")
                continue

            # 4e. Initial Dynamic SL/TP Calculation
            entry_price = float(df_candles["close"][-1])
            sl_price_raw, tp_price_raw = self.risk_engine.calculate_stop_levels(df_features, signal)

            # 4f. PRE-EXECUTION GATEWAY: Pass Signal to 9-Gateway Pipeline
            pred_payload = pl.DataFrame({
                "instrument": [instrument],
                "close": [entry_price],
                "calibrated_prob": [getattr(signal, "probability", 0.65)],
                "confidence_score": [getattr(signal, "confidence", 0.65)],
                "predicted_return": [getattr(signal, "expected_return", 0.002)],
                "stop_loss": [sl_price_raw],
                "take_profit": [tp_price_raw],
                "signal_direction": [signal.direction]
            })

            gateway_df = self.signal_engine.execute_pipeline(pred_payload)

            if gateway_df.is_empty() or not bool(gateway_df["is_valid_execution"][0]):
                reason = str(gateway_df["final_validator_reason"][0]) if not gateway_df.is_empty() else "GATEWAY_REJECTED"
                logger.warning(f"🚫 [{instrument}] Sinyal ditolak oleh 9-Gateway Pipeline: {reason}")
                continue

            # Extract Optimized Geometry Parameters from 9-Gateway
            opt_tp = float(gateway_df["optimized_take_profit"][0])
            opt_sl = float(gateway_df["optimized_stop_loss"][0])
            final_direction = str(gateway_df["candidate_signal"][0])

            # 4g. Adaptive Position Sizing (Units Calculation)
            units = self.position_sizer.calculate_units(
                account_balance=account_balance,
                entry_price=entry_price,
                stop_loss_price=opt_sl,
                instrument=instrument
            )

            # 4h. Strict Risk Engine Validation Gate
            is_valid, reason = self.risk_engine.validate_trade(instrument, units, portfolio_summary)
            if not is_valid:
                logger.warning(f"🚫 [{instrument}] Order ditolak Risk Engine: {reason}")
                continue

            # 4i. Order Execution Gate
            if self.is_live:
                exec_res = self.order_manager.execute_order(
                    instrument=instrument,
                    units=units,
                    order_type=final_direction,
                    stop_loss=opt_sl,
                    take_profit=opt_tp
                )
                executed_orders.append(exec_res)
                logger.info(f"🎉 [{instrument}] Hasil Eksekusi Live: {exec_res.get('status')}")
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
                logger.info(
                    f"🧪 [{instrument}] [DRY-RUN] Order Disimulasikan: {final_direction} {units} Units | "
                    f"Entry: {entry_price:.5f} | SL: {opt_sl:.5f} | TP: {opt_tp:.5f}"
                )

            cycle_telemetry.append({
                "instrument": instrument,
                "direction": final_direction,
                "entry_price": entry_price,
                "take_profit": opt_tp,
                "stop_loss": opt_sl,
                "units": units
            })

        # Step 5: Save System State & Portfolio Persistence
        updated_summary = self.portfolio_manager.get_summary()
        self.portfolio_manager.save_state(updated_summary)
        self.state_store.save_state(updated_summary)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        logger.info(f"\n✅ SIKLUS AUTONOM SELESAI dalam {elapsed_ms:.2f} ms. Total Eksekusi Order: {len(executed_orders)}")

        return {
            "status": "SUCCESS",
            "execution_mode": self.mode,
            "granularity": self.granularity,
            "candles_count": self.candle_count,
            "executed_orders_count": len(executed_orders),
            "executed_orders": executed_orders,
            "telemetry": cycle_telemetry,
            "latency_ms": elapsed_ms,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


if __name__ == "__main__":
    engine = AutonomousEngine()
    engine.run_cycle()
