"""
================================================================================
PROJECT     : FXP - Forex Autonomous Trading Bot
REPOSITORY  : https://github.com/raidvoltus/FX
MODULE      : autonomous_engine.py
VERSION     : 2026.Q3.v4.6.0 (P0 UNKNOWN Order Resolution & Hard Gates)
PYTHON      : 3.11+
================================================================================
"""

import uuid
import time
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Tuple

import polars as pl

from config import Config
from broker.oanda_client import OandaClient, BrokerStateError, MarketDataUnavailableError
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


class AutonomousEngine:

    def __init__(self) -> None:
        self.config = Config.load()
        self.granularity = str(getattr(self.config, "DEFAULT_GRANULARITY", "M5"))
        self.candle_count = int(getattr(self.config, "HISTORICAL_CANDLES_COUNT", 500))

        self.broker = OandaClient(
            api_url=self.config.OANDA_API_URL,
            api_token=self.config.OANDA_API_TOKEN,
            account_id=self.config.OANDA_ACCOUNT_ID
        )
        self.data_fetcher = MarketDataFetcher(self.broker)
        self.feature_engine = FeatureEngineFacade()

        self.strategy_selector = MLStrategySelector()
        self.ml_classifier = MLClassificationLive()
        self.signal_engine = UnifiedForexSignalEngine()

        self.risk_engine = RiskEngine()
        self.position_sizer = PositionSizer()

        self.order_manager = OrderManager(self.broker)
        self.portfolio_manager = PortfolioManager(self.broker)
        self.state_store = StateStore()
        self.storage_engine = UnifiedStorageEngine()
        self.health_monitor = HealthMonitor()

    def validate_account_credentials(self) -> Tuple[bool, str]:
        return self.broker.validate_account_credentials()

    def reconcile_remote_state(self) -> Tuple[bool, str]:
        try:
            remote_state = self.broker.get_account_state()
            
            positions_data = remote_state.get("positions", [])
            trades_data = remote_state.get("trades", [])
            orders_data = remote_state.get("orders", [])

            if hasattr(self.portfolio_manager, "reconcile_full_state"):
                self.portfolio_manager.reconcile_full_state(remote_state)
            
            if hasattr(self.state_store, "update_snapshot"):
                self.state_store.update_snapshot(remote_state)

            msg = f"Full Reconciliation OK | Positions: {len(positions_data)} | Open Trades: {len(trades_data)} | Pending Orders: {len(orders_data)}"
            return True, msg
        except Exception as err:
            return False, f"Reconciliation Error: {err}"

    def run_cycle(self) -> Dict[str, Any]:
        start_time = time.perf_counter()

        health_report = self.health_monitor.ping()
        if health_report.get("system_status") == "UNHEALTHY":
            return {"status": "HALTED", "reason": "SYSTEM_UNHEALTHY"}

        try:
            available_instruments = self.broker.get_instruments()
            active_instruments = self.data_fetcher.filter_tradeable_instruments(available_instruments)
        except Exception as err:
            logger.error("Fail-Closed: Instrument fetch failed: %s", err)
            raise BrokerStateError(f"Market Data Ingestion Error: {err}") from err

        if not active_instruments:
            return {"status": "SKIPPED", "reason": "NO_TRADEABLE_INSTRUMENTS"}

        portfolio_summary = self.portfolio_manager.get_summary()
        account_balance = float(portfolio_summary.get("balance", 0.0))

        executed_orders: List[Dict[str, Any]] = []
        rejected_orders: List[Dict[str, Any]] = []

        for instrument in active_instruments:
            try:
                if hasattr(self.portfolio_manager, "has_active_position_or_order"):
                    if self.portfolio_manager.has_active_position_or_order(instrument):
                        logger.info("[%s] Diabaikan: Posisi/Pending order sudah aktif.", instrument)
                        continue

                df_candles = self.broker.get_candles(instrument, granularity=self.granularity, count=self.candle_count)

                df_features = self.feature_engine.build_features(df_candles)
                selected_strategy, _ = self.strategy_selector.evaluate(df_features)
                if selected_strategy == "hold":
                    continue

                signal = self.ml_classifier.generate_signal(df_features)
                if signal.direction not in ["BUY", "SELL"]:
                    continue

                sl_raw, tp_raw = self.risk_engine.calculate_stop_levels(df_features, signal)

                pred_payload = pl.DataFrame({
                    "instrument": [instrument],
                    "close": [float(df_candles["close"][-1])],
                    "calibrated_prob": [getattr(signal, "probability", 0.65)],
                    "confidence_score": [getattr(signal, "confidence", 0.65)],
                    "predicted_return": [getattr(signal, "expected_return", 0.002)],
                    "stop_loss": [sl_raw],
                    "take_profit": [tp_raw],
                    "signal_direction": [signal.direction]
                })

                gateway_df = self.signal_engine.execute_pipeline(pred_payload)

                if gateway_df.is_empty() or not bool(gateway_df["is_valid_execution"][0]):
                    logger.warning("[%s] Sinyal ditolak 9-Gateway Filter.", instrument)
                    continue

                opt_tp = float(gateway_df["optimized_take_profit"][0])
                opt_sl = float(gateway_df["optimized_stop_loss"][0])
                final_direction = str(gateway_df["candidate_signal"][0])

                if final_direction not in ["BUY", "SELL"]:
                    logger.warning("[%s] Direction akhir '%s' tidak valid.", instrument, final_direction)
                    continue

                executable_price = self.broker.get_executable_price(instrument, final_direction)

                # HARD GATE: Validasi Geometri Arah SL/TP dan Risk-Reward Ratio
                if final_direction == "BUY":
                    if not (opt_sl < executable_price < opt_tp):
                        logger.warning("[%s] SL/TP Geometri BUY invalid (SL: %.5f, Price: %.5f, TP: %.5f)", instrument, opt_sl, executable_price, opt_tp)
                        continue
                elif final_direction == "SELL":
                    if not (opt_tp < executable_price < opt_sl):
                        logger.warning("[%s] SL/TP Geometri SELL invalid (TP: %.5f, Price: %.5f, SL: %.5f)", instrument, opt_tp, executable_price, opt_sl)
                        continue

                risk_dist = abs(executable_price - opt_sl)
                reward_dist = abs(opt_tp - executable_price)
                if risk_dist <= 1e-8 or (reward_dist / risk_dist) < 1.50:
                    logger.warning("[%s] Risk-Reward ratio < 1.50 (RR: %.2f)", instrument, reward_dist / max(risk_dist, 1e-8))
                    continue

                units = self.position_sizer.calculate_units(
                    account_balance=account_balance,
                    entry_price=executable_price,
                    stop_loss_price=opt_sl,
                    instrument=instrument
                )

                risk_payload = {
                    "instrument": instrument,
                    "units": units,
                    "executable_price": executable_price,
                    "stop_loss": opt_sl,
                    "take_profit": opt_tp,
                    "account_equity": float(portfolio_summary.get("NAV", account_balance)),
                    "current_exposure_pct": float(portfolio_summary.get("exposure_pct", 0.0)),
                    "daily_pnl_pct": float(portfolio_summary.get("daily_pnl_pct", 0.0)),
                    "drawdown_pct": float(portfolio_summary.get("drawdown_pct", 0.0)),
                    "open_positions_count": int(portfolio_summary.get("open_positions_count", 0)),
                    "probability": float(gateway_df["calibrated_prob"][0]),
                    "confidence": float(gateway_df["confidence_score"][0])
                }

                is_valid, risk_reason = self.risk_engine.validate_trade(risk_payload)
                if not is_valid:
                    logger.warning("[%s] Order ditolak Risk Engine Firewall: %s", instrument, risk_reason)
                    continue

                client_order_id = f"FXP-{instrument.replace('_', '')}-{int(time.time())}-{uuid.uuid4().hex[:6]}"
                
                exec_res = self.broker.execute_idempotent_order(
                    instrument=instrument,
                    units=units,
                    order_type=final_direction,
                    stop_loss=opt_sl,
                    take_profit=opt_tp,
                    client_order_id=client_order_id
                )

                status = str(exec_res.get("status", "UNKNOWN")).upper()
                if status == "EXECUTED":
                    executed_orders.append(exec_res)
                    self.storage_engine.persist_signals(gateway_df)
                elif status == "REJECTED":
                    rejected_orders.append(exec_res)
                elif status == "UNKNOWN":
                    # P0 PROTECTION: Hentikan trading dan picu rekonsiliasi darurat
                    logger.critical("🚨 Order UNKNOWN terdeteksi [%s] (ID: %s). MEMICU REKONSILIASI DARURAT...", instrument, client_order_id)
                    rec_ok, rec_msg = self.reconcile_remote_state()
                    if not rec_ok:
                        raise BrokerStateError(f"CRITICAL: Order UNKNOWN ({client_order_id}) dan rekonsiliasi gagal: {rec_msg}")
                    
                    return {
                        "status": "HALTED",
                        "reason": f"UNKNOWN_ORDER_REQUIRES_INSPECTION [{client_order_id}]",
                        "executed_orders_count": len(executed_orders),
                        "latency_ms": (time.perf_counter() - start_time) * 1000.0,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }

            except (BrokerStateError, MarketDataUnavailableError) as critical_err:
                logger.error("💥 Terjadi pengecualian kritis broker pada [%s]: %s", instrument, critical_err)
                raise critical_err
            except Exception as err:
                logger.error("Anomali pada evaluasi [%s]: %s", instrument, err, exc_info=True)
                continue

        updated_summary = self.portfolio_manager.get_summary()
        if hasattr(self.portfolio_manager, "save_state"):
            self.portfolio_manager.save_state(updated_summary)

        status_flag = "SUCCESS" if executed_orders or not rejected_orders else "PARTIAL"

        return {
            "status": status_flag,
            "executed_orders_count": len(executed_orders),
            "rejected_orders_count": len(rejected_orders),
            "executed_orders": executed_orders,
            "latency_ms": (time.perf_counter() - start_time) * 1000.0,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
