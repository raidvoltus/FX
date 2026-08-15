"""
================================================================================
PROJECT     : FXP - Forex Autonomous Trading Bot
REPOSITORY  : https://github.com/raidvoltus/FX
MODULE      : autonomous_engine.py
VERSION     : 2026.Q3.v4.3.0 (Execution Authority & Pre-Execution 9-Gateway)
PYTHON      : 3.11+
================================================================================
ARCHITECTURE:
    Market Ingestion -> Feature Building -> ML Strategy/Signal -> 9-Gateway Filter ->
    Position Sizing -> Risk Engine Validation -> Order Execution -> Storage Persistence
================================================================================
"""

import sys
import time
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Tuple, Optional

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
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter('[%(asctime)s] [%(levelname)s] [AutonomousEngine] %(message)s'))
    logger.addHandler(ch)
    logger.setLevel(logging.INFO)


class AutonomousEngine:
    """
    Core Execution Authority untuk bot trading autonomis FXP.
    Memegang kontrol penuh atas pipa data trading dari pasar hingga eksekusi broker.
    """

    def __init__(self) -> None:
        self.config = Config.load()

        # Dynamic Configurations dari Config Engine
        self.granularity = str(getattr(self.config, "DEFAULT_GRANULARITY", "M5"))
        self.candle_count = int(getattr(self.config, "HISTORICAL_CANDLES_COUNT", 500))

        # Infrastructure & REST Broker API Context
        self.broker = OandaClient(
            api_url=self.config.OANDA_API_URL,
            api_token=self.config.OANDA_API_TOKEN,
            account_id=self.config.OANDA_ACCOUNT_ID
        )
        self.data_fetcher = MarketDataFetcher(self.broker)
        self.feature_engine = FeatureEngineFacade()

        # Analytics, ML, & Pre-Execution Gatekeeper
        self.strategy_selector = MLStrategySelector()
        self.ml_classifier = MLClassificationLive()
        self.signal_engine = UnifiedForexSignalEngine()

        # Risk Engine & Position Sizer
        self.risk_engine = RiskEngine()
        self.position_sizer = PositionSizer()

        # Execution, State, & Mode-Isolated Storage
        self.order_manager = OrderManager(self.broker)
        self.portfolio_manager = PortfolioManager(self.broker)
        self.state_store = StateStore()
        self.storage_engine = UnifiedStorageEngine()
        self.health_monitor = HealthMonitor()

        logger.info(
            "AutonomousEngine aktif | Endpoint: %s | Granularity: %s | Historical Count: %s",
            self.config.OANDA_API_URL,
            self.granularity,
            self.candle_count
        )

    # =========================================================================
    # STARTUP CONTRACTS (Called by main.py Orchestrator)
    # =========================================================================

    def validate_account_credentials(self) -> Tuple[bool, str]:
        """
        Delegasi validasi 3-tingkat otentikasi REST API dan lisensi Account ID ke OandaClient.
        """
        return self.broker.validate_account_credentials()

    def reconcile_open_positions(self) -> Tuple[bool, str]:
        """
        Startup Reconciliation: Sinkronisasi posisi terbuka remote dari OANDA Server
        ke state lokal (PortfolioManager & StateStore) setelah restart/power loss.
        """
        try:
            remote_positions = self.broker.fetch_open_positions()
            
            # Sinkronkan snapshot posisi remote ke state lokal
            if hasattr(self.portfolio_manager, "reconcile_remote_positions"):
                self.portfolio_manager.reconcile_remote_positions(remote_positions)
            
            if hasattr(self.state_store, "update_positions_snapshot"):
                self.state_store.update_positions_snapshot(remote_positions)

            msg = f"Reconciled {len(remote_positions)} remote position(s) from OANDA server."
            return True, msg
        except Exception as err:
            return False, f"Gagal merekonsiliasi posisi terbuka: {str(err)}"

    # =========================================================================
    # SINGLE CYCLE EXECUTION LOOP
    # =========================================================================

    def run_cycle(self) -> Dict[str, Any]:
        """
        Menjalankan 1 Siklus Perdagangan Autonom Lengkap.
        Alur: Ingestion -> ML Model -> 9-Gateway Filter -> Sizing -> Risk Check -> Execution -> Persist.
        """
        start_time = time.perf_counter()

        # 1. Diagnostic Health Check
        health_report = self.health_monitor.ping()
        if health_report.get("system_status") == "UNHEALTHY":
            logger.critical("Status kesehatan sistem UNHEALTHY. Siklus trading dihentikan.")
            return {"status": "HALTED", "reason": "SYSTEM_UNHEALTHY"}

        # 2. Filter Tradeable Instruments (Spread & Liquidity Filter)
        try:
            available_instruments = self.broker.get_instruments()
            active_instruments = self.data_fetcher.filter_tradeable_instruments(available_instruments)
        except Exception as err:
            logger.error("Gagal mengambil/memfilter instrumen dari broker: %s", err)
            active_instruments = []

        if not active_instruments:
            logger.warning("Tidak ada pasangan Forex yang memenuhi kriteria spread/likuiditas.")
            return {"status": "SKIPPED", "reason": "NO_TRADEABLE_INSTRUMENTS"}

        # 3. Snapshot Balance & Portfolio Summary
        portfolio_summary = self.portfolio_manager.get_summary()
        account_balance = float(portfolio_summary.get("balance", 0.0))

        executed_orders: List[Dict[str, Any]] = []

        # 4. Iterasi Evaluasi Per Pasangan Mata Uang
        for instrument in active_instruments:
            try:
                # 4a. Fetch Candle Data (Dynamic Config)
                df_candles = self.data_fetcher.get_candles(
                    instrument,
                    granularity=self.granularity,
                    count=self.candle_count
                )
                if df_candles.height == 0:
                    continue

                # 4b. Technical Feature Engineering
                df_features = self.feature_engine.build_features(df_candles)

                # 4c. ML Strategy Selection (Regime Detection)
                selected_strategy, confidence = self.strategy_selector.evaluate(df_features)
                if selected_strategy == "hold":
                    continue

                # 4d. ML Directional Signal Model Generation
                signal = self.ml_classifier.generate_signal(df_features)
                if signal.direction not in ["BUY", "SELL"]:
                    continue

                # 4e. Dynamic Stop Loss & Take Profit Price Targets
                entry_price = float(df_candles["close"][-1])
                sl_raw, tp_raw = self.risk_engine.calculate_stop_levels(df_features, signal)

                # 4f. PRE-EXECUTION GATEWAY: Pemrosesan melalui 9-Gateway Signal Engine
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
                    logger.warning("[%s] Sinyal ditolak oleh 9-Gateway Filter: %s", instrument, reason)
                    continue

                # Ekstraksi Parameter Geometri Hasil Optimasi 9-Gateway
                opt_tp = float(gateway_df["optimized_take_profit"][0])
                opt_sl = float(gateway_df["optimized_stop_loss"][0])
                final_direction = str(gateway_df["candidate_signal"][0])

                # 4g. Adaptive Position Sizing
                units = self.position_sizer.calculate_units(
                    account_balance=account_balance,
                    entry_price=entry_price,
                    stop_loss_price=opt_sl,
                    instrument=instrument
                )

                # 4h. Strict Risk Engine Validation Gate
                is_valid, risk_reason = self.risk_engine.validate_trade(instrument, units, portfolio_summary)
                if not is_valid:
                    logger.warning("[%s] Order ditolak Risk Engine: %s", instrument, risk_reason)
                    continue

                # 4i. Order Execution Gate
                exec_res = self.order_manager.execute_order(
                    instrument=instrument,
                    units=units,
                    order_type=final_direction,
                    stop_loss=opt_sl,
                    take_profit=opt_tp
                )
                executed_orders.append(exec_res)

                # 4j. Audit Persistence (Persist Validated Signal Log to Storage)
                self.storage_engine.persist_signals(gateway_df)

            except Exception as err:
                logger.error("Anomali pada evaluasi instrumen [%s]: %s", instrument, err, exc_info=True)
                continue

        # 5. Save System State & Local Snapshot Update
        updated_summary = self.portfolio_manager.get_summary()
        if hasattr(self.portfolio_manager, "save_state"):
            self.portfolio_manager.save_state(updated_summary)
        if hasattr(self.state_store, "save_state"):
            self.state_store.save_state(updated_summary)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return {
            "status": "SUCCESS",
            "granularity": self.granularity,
            "candles_count": self.candle_count,
            "executed_orders_count": len(executed_orders),
            "executed_orders": executed_orders,
            "latency_ms": elapsed_ms,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
