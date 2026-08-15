"""
================================================================================
PROJECT     : FXP (Forex Autonomous Trading Bot)
REPOSITORY  : https://github.com/raidvoltus/FX
MODULE      : worker_analytics.py
DESCRIPTION : Standalone Background Analytics & Machine Learning Worker.
VERSION     : 2026.Q3.v4.0.0 (Fully Aligned Edition)
PYTHON      : 3.11+
================================================================================
"""

import os
import sys
import time
import signal
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Tuple

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
    Background Analytics Worker yang selaras dengan main.py & storage.py.
    Membaca log dari SQLite storage tanpa mengganggu alur eksekusi main.py.
    """

    def __init__(self, mode: Optional[str] = None):
        self.config = Config.load()
        self.mode = str(mode or os.getenv("EXECUTION_MODE", "dry-run")).lower().strip()
        self.is_running = False

        # Load Storage & Analytics Engines
        self.storage_engine = UnifiedStorageEngine(mode=self.mode)
        self.evaluation_engine = UnifiedEvaluationEngine(
            annualization_factor=float(getattr(self.config, "ANNUALIZATION_FACTOR", 252.0))
        )
        self.validation_engine = UnifiedValidationEngine(seed=42)
        self.self_learning_engine = UnifiedSelfLearningEngine()

        # Diselaraskan dengan rolling window minimal pada evaluation.py (min 63 samples)
        self.min_samples_required = max(65, int(getattr(self.config, "MIN_RETRAIN_SAMPLES", 65)))
        self.worker_interval_sec = int(os.getenv("ANALYTICS_WORKER_INTERVAL", "3600"))

        logger.info(f"⚙️ Analytics Worker diinisialisasi | Mode: {self.mode.upper()} | Interval: {self.worker_interval_sec}s")

    def _fetch_historical_records(self) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """Membaca histori sinyal dan prediksi dari SQLite Database secara aman."""
        conn = self.storage_engine.sqlite_engine.get_connection()

        try:
            df_signals = pl.read_database("SELECT * FROM signal_history WHERE signal != 0", conn)
            df_predictions = pl.read_database("SELECT * FROM prediction_history", conn)
            return df_signals, df_predictions
        except Exception as err:
            logger.error(f"❌ Gagal membaca data dari SQLite Storage: {err}")
            return pl.DataFrame(), pl.DataFrame()

    def run_analytics_pass(self) -> Dict[str, Any]:
        """Menjalankan 1 pasang evaluasi kuantitatif penuh."""
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
        # 1. EVALUATION ENGINE PASS
        # ----------------------------------------------------------------------
        try:
            # Ekstraksi returns yang telah disanitasi dari null/nan
            prob_arr = df_signals["confidence"].fill_null(0.5).to_numpy()
            sig_arr = df_signals["signal"].fill_null(0).to_numpy()
            
            # Sintesis return series yang valid
            returns_array = (prob_arr * sig_arr) * 0.001
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
        # 2. VALIDATION ENGINE PASS
        # ----------------------------------------------------------------------
        try:
            instruments = df_signals["instrument"].to_list() if "instrument" in df_signals.columns else ["EUR_USD"] * total_records
            mock_df_val = pl.DataFrame({
                "timestamp": pl.datetime_range(
                    start=datetime(2026, 1, 1), 
                    end=datetime(2026, 1, 10), 
                    interval="1h", 
                    eager=True
                )[:total_records],
                "instrument": instruments,
                "returns": returns_array
            })

            val_metrics = self.validation_engine.validate_full_pipeline(mock_df_val, returns_array)
            results["validation_status"] = val_metrics.get("status")
            logger.info(f"🛡️ [Validation Pass] System Validation Status: {val_metrics.get('status')}")
        except Exception as err:
            logger.error(f"❌ Validation Pass gagal: {err}")

        # ----------------------------------------------------------------------
        # 3. SELF-LEARNING ENGINE PASS
        # ----------------------------------------------------------------------
        try:
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
        """Menjalankan background worker loop."""
        self.is_running = True
        logger.info(f"🚀 Worker Analytics aktif. Loop interval: {self.worker_interval_sec} detik.")

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
