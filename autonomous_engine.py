"""
================================================================================
MODULE      : autonomous_engine.py
DESCRIPTION : Core Autonomous Execution Engine for OANDA Forex Trading Bot.
VERSION     : 2026.1.0 (End-to-End Forex Autonomous Loop Edition)
PYTHON      : 3.11+ / 3.12+
ARCHITECTURE: Clean Architecture, Decoupled Autonomous Event Loop
================================================================================
"""

import os
import sys
import time
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

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
from self_learning import UnifiedSelfLearningEngine

# Setup Logging
logger = logging.getLogger("Forex.AutonomousEngine")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [AutonomousEngine]: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class AutonomousEngine:
    """
    Engine Pengendali Utama yang Memutar Siklus Trading Autonom.
    Mengoordinasikan seluruh sub-sistem dari Data Ingestion hingga Execution & Self-Learning.
    """

    def __init__(self, mode: Optional[str] = None):
        self.config = Config.load()
        self.mode = str(mode or os.getenv("EXECUTION_MODE", "dry-run")).lower().strip()
        self.is_live = (self.mode == "live")

        logger.info(f"🚀 Inisialisasi AutonomousEngine dalam mode: {self.mode.upper()}")

        # 1. Broker & Ingestion Layer
        self.broker = OandaClient(environment="practice" if not self.is_live else "live")
        self.data_fetcher = MarketDataFetcher(self.broker)
        self.feature_engine = FeatureEngineFacade()

        # 2. Analytics & Strategy Layer
        self.strategy_selector = MLStrategySelector()
        self.ml_classifier = MLClassificationLive()

        # 3. Risk & Position Sizing Layer
        self.risk_engine = RiskEngine()
        self.position_sizer = PositionSizer()

        # 4. Execution & Portfolio Layer
        self.order_manager = OrderManager(self.broker)
        self.portfolio_manager = PortfolioManager(self.broker, mode=self.mode)
        self.state_store = StateStore()
        self.health_monitor = HealthMonitor()

        # 5. Self-Learning & Adaptive Risk Engine
        self.self_learning_engine = UnifiedSelfLearningEngine(
            cooldown_period_days=int(getattr(self.config, "RETRAIN_COOLDOWN_DAYS", 7)),
            min_samples_to_retrain=int(getattr(self.config, "MIN_RETRAIN_SAMPLES", 100))
        )

    def run_cycle(self) -> Dict[str, Any]:
        """
        Menjalankan 1 Siklus Autonom Lengkap.
        """
        start_time = time.perf_counter()
        logger.info("==================================================")
        logger.info(f"🔄 MEMULAI SIKLUS AUTONOM FOREX (Mode: {self.mode.upper()})")
        logger.info("==================================================")

        # Step 1: System Health Diagnostic Ping
        health_report = self.health_monitor.ping()
        if health_report.get("system_status") == "UNHEALTHY":
            logger.critical("🚨 Status kesehatan sistem UNHEALTHY. Siklus trading dihentikan sementara.")
            return {"status": "HALTED", "reason": "SYSTEM_UNHEALTHY"}

        # Step 2: Filter Tradeable Instruments (Spread & Liquidity Filter)
        try:
            available_instruments = self.broker.api.get_instruments()
            active_instruments = self.data_fetcher.filter_tradeable_instruments(available_instruments)
        except Exception as e:
            logger.error(f"❌ Gagal mengambil/memfilter instrumen dari OANDA: {e}")
            active_instruments = []

        if not active_instruments:
            logger.warning("⚠️ Tidak ada pasangan Forex yang lolos filter spread saat ini.")
            return {"status": "SKIPPED", "reason": "NO_TRADEABLE_INSTRUMENTS"}

        # Step 3: Fetch Account Balance & Active Positions Summary
        portfolio_summary = self.portfolio_manager.get_summary()
        account_balance = float(portfolio_summary.get("balance", 0.0))

        executed_orders: List[Dict[str, Any]] = []
        candles_cache: Dict[str, Any] = {}

        # Step 4: Iterasi Per Pasangan Mata Uang (Pair Evaluation)
        for instrument in active_instruments:
            logger.info(f"\n--- [MEMPROSES INSTRUMEN: {instrument}] ---")

            # 4a. Fetch OHLCV Candles Data
            df_candles = self.data_fetcher.get_candles(instrument, granularity="M5", count=200)
            if df_candles.height == 0:
                logger.warning(f"⚠️ [{instrument}] Data candle kosong, melewati instrumen.")
                continue

            candles_cache[instrument] = df_candles

            # 4b. Extract Technical Features & Indicators
            df_features = self.feature_engine.build_features(df_candles)

            # 4c. ML Strategy Selector & Market Regime Classifier
            selected_strategy, confidence = self.strategy_selector.evaluate(df_features)
            if selected_strategy == "hold":
                logger.info(f"⏸ [{instrument}] Rejim pasar tidak terarah / Noise. Strategi memilih HOLD.")
                continue

            # 4d. ML Directional Signal Model Generation
            signal = self.ml_classifier.generate_signal(df_features)
            if signal.direction not in ["BUY", "SELL"]:
                logger.info(f"⏸ [{instrument}] Sinyal ML merekomendasikan HOLD.")
                continue

            # 4e. Dynamic Risk Levels (Stop Loss & Take Profit)
            sl_price, tp_price = self.risk_engine.calculate_stop_levels(df_features, signal)

            # 4f. Adaptive Position Sizing (Units Calculation)
            entry_price = float(df_candles["close"][-1])
            units = self.position_sizer.calculate_units(
                account_balance=account_balance,
                entry_price=entry_price,
                stop_loss_price=sl_price,
                instrument=instrument
            )

            # 4g. Strict Risk Engine Validation Gate
            is_valid, reason = self.risk_engine.validate_trade(instrument, units, portfolio_summary)
            if not is_valid:
                logger.warning(f"🚫 [{instrument}] Order ditolak Risk Engine: {reason}")
                continue

            # 4h. Order Manager Execution
            if self.is_live:
                exec_res = self.order_manager.execute_order(
                    instrument=instrument,
                    units=units,
                    order_type=signal.direction,
                    stop_loss=sl_price,
                    take_profit=tp_price
                )
                executed_orders.append(exec_res)
                logger.info(f"🎉 [{instrument}] Hasil Eksekusi Live: {exec_res.get('status')}")
            else:
                sim_res = {
                    "status": "SIMULATED",
                    "instrument": instrument,
                    "direction": signal.direction,
                    "units": units,
                    "entry_price": entry_price,
                    "stop_loss": sl_price,
                    "take_profit": tp_price,
                    "timestamp": time.time()
                }
                executed_orders.append(sim_res)
                logger.info(
                    f"🧪 [{instrument}] [DRY-RUN] Order Disimulasikan: {signal.direction} {units} Units | "
                    f"Entry: {entry_price:.5f} | SL: {sl_price:.5f} | TP: {tp_price:.5f}"
                )

        # Step 5: Save State & Portfolio Tracking
        updated_summary = self.portfolio_manager.get_summary()
        self.portfolio_manager.save_state(updated_summary)
        self.state_store.save_state(updated_summary)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        logger.info(f"\n✅ SIKLUS AUTONOM SELESAI dalam {elapsed_ms:.2f} ms. Total Eksekusi Order: {len(executed_orders)}")

        return {
            "status": "SUCCESS",
            "execution_mode": self.mode,
            "executed_orders_count": len(executed_orders),
            "executed_orders": executed_orders,
            "latency_ms": elapsed_ms,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


if __name__ == "__main__":
    engine = AutonomousEngine()
    engine.run_cycle()
