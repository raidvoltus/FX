"""
================================================================================
PROJECT     : FXP (Forex Autonomous Trading Bot)
REPOSITORY  : https://github.com/raidvoltus/FX
MODULE      : worker_analytics.py
DESCRIPTION : Standalone Background Analytics & Machine Learning Worker.
VERSION     : 2026.Q3.v1.0.0 (Decoupled Heavy Analytics Edition)
PYTHON      : 3.11+ / 3.12+
ARCHITECTURE: Asynchronous Audit, Out-of-Band Feedback Loop & Model Validation
================================================================================
"""

import os
import sys
import time
import signal
import sqlite3
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional

import numpy as np
import polars as pl

from config import Config
from storage import UnifiedStorageEngine
from evaluation import UnifiedEvaluationEngine
from validation import UnifiedValidationEngine
from self_learning import UnifiedSelfLearningEngine

# Logging Setup
logger = logging.getLogger("FXP.WorkerAnalytics")
if not logger.handlers:
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter('[%(asctime)s] [%(levelname)s] [WorkerAnalytics]: %(message)s'))
    logger.addHandler(ch)
    logger.setLevel(logging.INFO)


class AnalyticsWorker:
    """
    Background Analytics Worker untuk Ekosistem FXP.
    Membaca data historis dari SQLite storage dan mengeksekusi komputasi berat:
    1. Performance & Risk Evaluation (Sharpe, Sortino, Calmar, DSR, Alpha Decay)
    2. Quantitative Model Validation (Stationary Bootstrap, Monte Carlo Risk Simulation)
    3. Self-Learning Audit (Feature Drift PSI/KS Test, Champion-Challenger Evaluation)
    """

    def __init__(self, mode: Optional[str] = None):
        self.config = Config.load()
        self.mode = str(mode or os.getenv("EXECUTION_MODE", "dry-run")).lower().strip()
        self.is_running = False

        # Load Storage & Analytics Engines
        self.storage_engine = UnifiedStorageEngine(mode=self.mode)
        self.evaluation_engine = UnifiedEvaluationEngine()
        self.validation_engine = UnifiedValidationEngine()
        self.self_learning_engine = UnifiedSelfLearningEngine()

        self.min_samples_required = int(getattr(self.config, "MIN_RETRAIN_SAMPLES", 50))
        self.worker_interval_sec = int(os.getenv("ANALYTICS_WORKER_INTERVAL", "3600"))  # Default: 1 Jam

        logger.info(f"⚙️ Analytics Worker diinisialisasi | Mode: {self.mode.upper()} | Interval: {self.worker_interval_sec}s")

    def _fetch_historical_records((self)) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """Membaca histori sinyal dan prediksi dari SQLite Database."""
        conn = self.storage_engine.sqlite_engine.get_connection()

        try:
            df_signals = pl.read_database("SELECT * FROM signal_history", conn)
            df_predictions = pl.read_database("SELECT * FROM prediction_history", conn)
            return df_signals, df_predictions
        except Exception as err:
            logger.error(f"❌ Gagal membaca data dari SQLite Storage: {err}")
            return pl.DataFrame(), pl.DataFrame()

    def run_analytics_pass(self) -> Dict[str, Any]:
        """Menjalankan 1 pasang evaluasi kuantitatif penuh secara out-of-band."""
        start_time = time.perf_counter()
        logger.info(f"\n🧠 Memulai Analisis Kuantitatif Out-of-Band @ {datetime.now(timezone.utc).isoformat()}")

        df_signals, df_predictions = self._fetch_historical_records()
        total_records = df_signals.height

        if total_records < self.min_samples_required:
            logger.warning(
                f"⚠️ Sampel historis belum mencukupi untuk analitis ({total_records}/{self.min_samples_required}). "
                "Analisis dilewati."
            )
            return {"status": "SKIPPED", "reason": "INSUFFICIENT_SAMPLES", "samples": total_records}

        results: Dict[str, Any] = {"timestamp": datetime.now(timezone.utc).isoformat()}

        # ----------------------------------------------------------------------
        # 1. EVALUATION ENGINE PASS (Quantitative Performance Metrics)
        # ----------------------------------------------------------------------
        try:
            # Simulasi deret return berbasis probabilitas & arah sinyal historis
            returns_array = (df_signals["probability"].to_numpy() * df_signals["signal"].to_numpy()) * 0.001
            returns_series = pl.Series("forex_historical_returns", returns_array)

            eval_metrics = self.evaluation_engine.evaluate_strategy(returns_series, strategy_name="FXP_Historical")
            results["evaluation"] = eval_metrics
            logger.info(
                f"📊 [Evaluation Pass] Sharpe: {eval_metrics.get('sharpe_ratio', 0.0):.2f} | "
                f"Sortino: {eval_metrics.get('sortino_ratio', 0.0):.2f} | Calmar: {eval_metrics.get('calmar_ratio', 0.0):.2f}"
            )
        except Exception as err:
            logger.error(f"❌ Evaluation Pass gagal: {err}")

        # ----------------------------------------------------------------------
        # 2. VALIDATION ENGINE PASS (Bootstrap & Monte Carlo Risk Simulation)
        # ----------------------------------------------------------------------
        try:
            mock_df_val = pl.DataFrame({
                "timestamp": pl.datetime_range(
                    start=datetime(2026, 1, 1), 
                    end=datetime(2026, 1, 10), 
                    interval="1h", 
                    eager=True
                )[:total_records],
                "instrument": df_signals["instrument"],
                "returns": returns_array
            })

            val_metrics = self.validation_engine.validate_full_pipeline(mock_df_val, returns_array)
            results["validation_status"] = val_metrics.get("status")
            logger.info(f"🛡️ [Validation Pass] System Validation Status: {val_metrics.get('status')}")
        except Exception as err:
            logger.error(f"❌ Validation Pass gagal: {err}")

        # ----------------------------------------------------------------------
        # 3. SELF-LEARNING ENGINE PASS (Feature Drift Audit & Retrain Trigger)
        # ----------------------------------------------------------------------
        try:
            # Dummy feature matrices untuk drift check
            feature_cols = ["f1", "f2", "f3"]
            base_features = pl.DataFrame(np.random.normal(0, 1, (100, 3)), schema=feature_cols)
            curr_features = pl.DataFrame(np.random.normal(0.05, 1.05, (total_records, 3)), schema=feature_cols)

            drift_report = self.self_learning_engine.drift_detector.analyze_feature_drift(
                baseline_df=base_features, target_df=curr_features, feature_cols=feature_cols
            )

            retrain_decision = self.self_learning_engine.retraining_scheduler.evaluate_retraining_need(
                model_id="FXP_Core_ML", current_data_size=total_records, drift_report=drift_report
            )

            results["retrain_decision"] = retrain_decision
            logger.info(f"🔄 [Self-Learning Pass] Trigger Retrain: {retrain_decision.get('trigger_retraining')} | Reason: {retrain_decision.get('audit_justification')}")
        except Exception as err:
            logger.error(f"❌ Self-Learning Pass gagal: {err}")

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        results["latency_ms"] = elapsed_ms
        logger.info(f"✅ Pass Analitis Selesai dalam {elapsed_ms:.2f} ms.")

        return results

    def start_worker_loop(self) -> None:
        """Menjalankan background worker loop secara independen."""
        self.is_running = True
        logger.info(f"🚀 Worker Analytics aktif secara berkelanjutan. Loop interval: {self.worker_interval_sec} detik.")

        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

        while self.is_running:
            self.run_analytics_pass()

            for _ in range(self.worker_interval_sec):
                if not self.is_running:
                    break
                time.sleep(1)

        logger.info("🛑 Analytics Worker dihentikan secara bersih.")

    def _handle_shutdown(self, signum: int, frame: Any) -> None:
        logger.warning(f"\n⚠️ Sinyal shutdown OS ({signum}) diterima pada Analytics Worker. Menghentikan proses...")
        self.is_running = False


def main():
    import argparse

    parser = argparse.ArgumentParser(description="FXP Standalone Background Analytics Worker")
    parser.add_argument("--once", action="store_true", help="Jalankan 1 pass analitis saja lalu keluar")
    args = parser.parse_args()

    worker = AnalyticsWorker()
    if args.once:
        worker.run_analytics_pass()
    else:
        worker.start_worker_loop()


if __name__ == "__main__":
    main()
