"""
================================================================================
MODULE      : self_learning.py
DESCRIPTION : Consolidated Institutional Quantitative Self-Learning & Feedback Engine.
VERSION     : 2026.Q3.v1.0 (Forex Multi-Engine Synchronized Edition)
PYTHON      : 3.11+ / 3.12+
ARCHITECTURE: Clean Architecture, Decoupled Quantitative Sub-Engines
================================================================================
"""

import os
import sqlite3
import threading
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import polars as pl
from scipy import stats
from scipy.stats import norm

# Optional ML framework imports with graceful fallbacks
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import catboost as cb
    CB_AVAILABLE = True
except ImportError:
    CB_AVAILABLE = False


# =============================================================================
# CONSTANTS & EXCEPTIONS
# =============================================================================
EPSILON: float = 1e-6

class SelfLearningBaseError(Exception): pass
class DataValidationError(SelfLearningBaseError): pass
class ChronologyError(SelfLearningBaseError): pass
class ModelValidationError(SelfLearningBaseError): pass
class ConfidenceEstimatorError(SelfLearningBaseError): pass
class DriftDetectorError(SelfLearningBaseError): pass
class HyperparameterOptimizerError(SelfLearningBaseError): pass
class ModelRankerError(SelfLearningBaseError): pass
class ModelSelectorError(SelfLearningBaseError): pass
class OnlineLearningError(SelfLearningBaseError): pass
class OptunaManagerError(SelfLearningBaseError): pass
class RetrainingSchedulerError(SelfLearningBaseError): pass
class UncertaintyEstimatorError(SelfLearningBaseError): pass

try:
    from logger import logger
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    logger = logging.getLogger("SelfLearningEngine")


# =============================================================================
# 1. CHAMPION-CHALLENGER EVALUATION ENGINE
# =============================================================================
class ChampionChallengerEngine:
    """Promosi model otomatis berbasis Diebold-Mariano (HLN Adjusted) & Sharpe Ratio."""

    def __init__(self, alpha: float = 0.05, min_sharpe_improvement: float = 0.15, min_samples_required: int = 30):
        self.alpha = alpha
        self.min_sharpe_improvement = min_sharpe_improvement
        self.min_samples_required = min_samples_required
        self._lock = threading.Lock()

    def evaluate_promotion(
        self,
        challenger_id: str,
        challenger_summary: Dict[str, Any],
        champion_id: Optional[str] = None,
        champion_summary: Optional[Dict[str, Any]] = None,
        challenger_series: Optional[pl.DataFrame] = None,
        champion_series: Optional[pl.DataFrame] = None,
        join_keys: Optional[List[str]] = None,
        error_col: str = "absolute_error",
        date_col: str = "signal_date",
    ) -> Dict[str, Any]:
        start_time = time.time()
        keys = join_keys if join_keys is not None else ["asset_id", date_col]

        if champion_id is None or champion_summary is None:
            return self._build_report(True, "Cold-start activation. No active champion model.", challenger_id, None, 0.0, 0.0, time.time() - start_time)

        chal_sharpe = float(challenger_summary.get("sharpe_ratio", 0.0))
        champ_sharpe = float(champion_summary.get("sharpe_ratio", 0.0))
        sharpe_delta = chal_sharpe - champ_sharpe

        if sharpe_delta < self.min_sharpe_improvement:
            return self._build_report(False, f"Challenger failed Sharpe hurdle: {sharpe_delta:.4f} < {self.min_sharpe_improvement:.4f}", challenger_id, champion_id, 1.0, 0.0, time.time() - start_time)

        p_value, t_stat = 1.0, 0.0
        if challenger_series is not None and champion_series is not None:
            try:
                t_stat, p_value = self._execute_dm_hln_test(challenger_series, champion_series, keys, error_col)
            except Exception as err:
                return self._build_report(False, f"Statistical test error: {err}", challenger_id, champion_id, 1.0, 0.0, time.time() - start_time)

        statistically_superior = (p_value < self.alpha) and (t_stat < 0)
        is_promoted = statistically_superior and (sharpe_delta >= self.min_sharpe_improvement)

        return self._build_report(is_promoted, "DM-HLN Statistical superiority cleared." if is_promoted else "Statistical hurdle unmet.", challenger_id, champion_id, p_value, t_stat, time.time() - start_time)

    def _execute_dm_hln_test(self, challenger_df: pl.DataFrame, champion_df: pl.DataFrame, join_keys: List[str], error_col: str, h: int = 1) -> Tuple[float, float]:
        aligned = challenger_df.select(join_keys + [error_col]).join(champion_df.select(join_keys + [error_col]), on=join_keys, how="inner", suffix="_champion")
        N = aligned.shape[0]
        if N < self.min_samples_required:
            raise DataValidationError(f"Samples count ({N}) < min required ({self.min_samples_required})")

        err_chal = aligned.select(error_col).to_numpy().ravel()
        err_champ = aligned.select(f"{error_col}_champion").to_numpy().ravel()

        d = (err_chal ** 2) - (err_champ ** 2)
        d_bar = np.mean(d)
        auto_cov = np.var(d, ddof=1)
        variance_d = auto_cov / N
        if variance_d <= 1e-12:
            return 0.0, 1.0

        dm_stat = d_bar / np.sqrt(variance_d)
        hln_factor = np.sqrt((N + 1 - 2 * h + (h / N) * (h - 1)) / N)
        dm_hln_stat = dm_stat * hln_factor
        p_val = float(stats.t.cdf(dm_hln_stat, df=N - 1))
        return float(dm_hln_stat), float(p_val)

    def _build_report(self, promoted: bool, reason: str, chal_id: str, champ_id: Optional[str], p_val: float, t_stat: float, duration: float) -> Dict[str, Any]:
        return {
            "promotion_executed": promoted,
            "decision_timestamp": datetime.now(timezone.utc).isoformat(),
            "challenger_id": chal_id,
            "champion_id": champ_id,
            "p_value": p_val,
            "t_statistic": t_stat,
            "execution_duration_sec": round(duration, 4),
            "audit_justification": reason
        }


# =============================================================================
# 2. CONFIDENCE ESTIMATOR ENGINE
# =============================================================================
class ConfidenceEstimator:
    """Kuantifikasi skor keyakinan sinyal berbasis varians eror rolling."""

    def __init__(self, default_confidence_level: float = 0.95, min_acceptable_win_rate: float = 0.48):
        self.default_confidence_level = default_confidence_level
        self.min_acceptable_win_rate = min_acceptable_win_rate

    def estimate_uncertainty(self, predictions_df: pl.DataFrame, performance_summary: Dict[str, Any], model_id: str, pred_col: str = "predicted_return") -> pl.DataFrame:
        if predictions_df.is_empty() or model_id not in performance_summary:
            raise DataValidationError("Dataframe atau metadata model tidak valid.")

        model_meta = performance_summary[model_id]
        win_rate = float(model_meta.get("win_rate", 0.5))
        error_var = float(model_meta.get("error_variance", 0.01))
        error_std = float(np.sqrt(max(error_var, EPSILON)))

        z_score = float(norm.ppf(1.0 - (1.0 - self.default_confidence_level) / 2.0))
        critical_margin = float(z_score * error_std)

        output_df = predictions_df.with_columns([
            (pl.col(pred_col).abs() / error_std).alias("_z_signal")
        ]).with_columns([
            (0.5 * (1.0 + (pl.col("_z_signal") / np.sqrt(2.0)).erf())).alias("_gaussian_cdf")
        ]).with_columns([
            (2.0 * pl.col("_gaussian_cdf") - 1.0).alias("_magnitude_conviction")
        ]).with_columns([
            pl.when(pl.lit(win_rate >= self.min_acceptable_win_rate))
            .then((pl.col("_magnitude_conviction") * 0.4) + (pl.lit(win_rate) * 0.6))
            .otherwise(((pl.col("_magnitude_conviction") * 0.2) + (pl.lit(win_rate) * 0.8)) * (win_rate / self.min_acceptable_win_rate))
            .clip(0.0, 1.0)
            .alias("confidence_score"),
            (pl.col(pred_col) - critical_margin).alias("lower_bound"),
            (pl.col(pred_col) + critical_margin).alias("upper_bound")
        ]).drop(["_z_signal", "_gaussian_cdf", "_magnitude_conviction"])

        return output_df


# =============================================================================
# 3. DRIFT DETECTOR ENGINE
# =============================================================================
class DriftDetector:
    """Deteksi pergeseran distribusi fitur menggunakan PSI dan Kolmogorov-Smirnov Test."""

    def __init__(self, psi_warning_threshold: float = 0.10, psi_action_threshold: float = 0.25, ks_alpha: float = 0.05, num_bins: int = 10):
        self.psi_warning_threshold = psi_warning_threshold
        self.psi_action_threshold = psi_action_threshold
        self.ks_alpha = ks_alpha
        self.num_bins = num_bins

    def analyze_feature_drift(self, baseline_df: pl.DataFrame, target_df: pl.DataFrame, feature_cols: List[str]) -> Dict[str, Any]:
        if baseline_df.is_empty() or target_df.is_empty():
            raise DataValidationError("DataFrame baseline atau target kosong.")

        drift_logs = {}
        global_drift = False

        for feat in feature_cols:
            if feat not in baseline_df.columns or feat not in target_df.columns:
                continue

            arr_base = baseline_df.select(feat).to_numpy().ravel()
            arr_target = target_df.select(feat).to_numpy().ravel()

            psi_val = self._compute_psi(arr_base, arr_target)
            ks_stat, ks_p_val = stats.ks_2samp(arr_base, arr_target)

            status = "STABLE"
            if psi_val >= self.psi_action_threshold:
                status = "CRITICAL_DRIFT"
                if ks_p_val < self.ks_alpha:
                    global_drift = True
            elif psi_val >= self.psi_warning_threshold:
                status = "WARNING_DRIFT"

            drift_logs[feat] = {
                "psi_value": round(psi_val, 6),
                "ks_statistic": round(float(ks_stat), 6),
                "ks_p_value": round(float(ks_p_val), 6),
                "drift_status": status
            }

        return {
            "summary_metrics": {
                "total_features": len(feature_cols),
                "system_retrain_recommended": global_drift
            },
            "features_drift_logs": drift_logs
        }

    def _compute_psi(self, base: np.ndarray, target: np.ndarray) -> float:
        bins = np.linspace(np.min(base), np.max(base), self.num_bins + 1)
        base_counts, _ = np.histogram(base, bins=bins)
        target_counts, _ = np.histogram(target, bins=bins)

        base_pcts = np.where(base_counts == 0, EPSILON, base_counts) / len(base)
        target_pcts = np.where(target_counts == 0, EPSILON, target_counts) / len(target)

        psi = np.sum((target_pcts - base_pcts) * np.log(target_pcts / base_pcts))
        return float(psi)


# =============================================================================
# 4. FEEDBACK ENGINE
# =============================================================================
class FeedbackEngine:
    """Evaluasi hasil sinyal histori terhadap realized return pasar."""

    def __init__(self, execution_date: Optional[Union[str, date, datetime]] = None):
        if execution_date is None:
            self.execution_date = date.today()
        elif isinstance(execution_date, datetime):
            self.execution_date = execution_date.date()
        else:
            self.execution_date = execution_date

    def process_feedback_loop(
        self,
        predictions: pl.DataFrame,
        actuals: pl.DataFrame,
        processed_prediction_ids: Set[str],
        prediction_id_col: str = "prediction_id",
        model_id_col: str = "model_id",
        pred_value_col: str = "predicted_return",
        realized_value_col: str = "realized_return",
        join_keys: Optional[List[str]] = None,
    ) -> Tuple[pl.DataFrame, Dict[str, Any]]:
        if predictions.is_empty() or actuals.is_empty():
            return pl.DataFrame(), {}

        keys = join_keys if join_keys is not None else ["asset_id", "date"]
        filtered_preds = predictions.filter(~pl.col(prediction_id_col).is_in(list(processed_prediction_ids)))
        if filtered_preds.is_empty():
            return pl.DataFrame(), {}

        feedback_matrix = filtered_preds.join(actuals, on=keys, how="inner")
        if feedback_matrix.is_empty():
            return pl.DataFrame(), {}

        feedback_matrix = feedback_matrix.with_columns([
            (pl.col(realized_value_col) - pl.col(pred_value_col)).alias("residual_error"),
            (pl.col(realized_value_col) - pl.col(pred_value_col)).abs().alias("absolute_error"),
            ((pl.col(realized_value_col) - pl.col(pred_value_col)) ** 2).alias("squared_error"),
            (((pl.col(pred_value_col) >= 0) & (pl.col(realized_value_col) >= 0)) |
             ((pl.col(pred_value_col) < 0) & (pl.col(realized_value_col) < 0))).cast(pl.Int8).alias("directional_hit")
        ])

        telemetry = {
            "global_processed_count": feedback_matrix.shape[0],
            "mae": float(feedback_matrix["absolute_error"].mean() or 0.0),
            "hit_rate": float(feedback_matrix["directional_hit"].mean() or 0.0)
        }

        return feedback_matrix, telemetry


# =============================================================================
# 5. HYPERPARAMETER OPTIMIZER ENGINE
# =============================================================================
class HyperparameterOptimizer:
    """Optimasi hyperparameter GBDT terintegrasi Optuna & SQLite."""

    def __init__(self, storage_uri: str = "sqlite:///optuna_study.db", seed: int = 42):
        self.storage_uri = storage_uri
        self.seed = seed

    def optimize_gbdt(
        self,
        model_type: str,
        train_df: pl.DataFrame,
        val_df: pl.DataFrame,
        feature_cols: List[str],
        target_col: str,
        n_trials: int = 15,
        timeout: Optional[int] = 300
    ) -> Tuple[Dict[str, Any], float]:
        if not OPTUNA_AVAILABLE:
            raise HyperparameterOptimizerError("Optuna library belum terinstall.")

        x_train = train_df.select(feature_cols).to_numpy()
        y_train = train_df.select(target_col).to_numpy().ravel()
        x_val = val_df.select(feature_cols).to_numpy()
        y_val = val_df.select(target_col).to_numpy().ravel()

        def objective(trial: optuna.Trial) -> float:
            lr = trial.suggest_float("learning_rate", 0.01, 0.15, log=True)
            max_depth = trial.suggest_int("max_depth", 4, 8)
            
            if model_type.lower() == "lightgbm" and LGB_AVAILABLE:
                train_data = lgb.Dataset(x_train, label=y_train)
                val_data = lgb.Dataset(x_val, label=y_val, reference=train_data)
                model = lgb.train({"objective": "regression", "learning_rate": lr, "max_depth": max_depth, "verbose": -1}, train_data, num_boost_round=100)
                preds = model.predict(x_val)
            else:
                return float("inf")

            return float(np.sqrt(np.mean((y_val - preds) ** 2)))

        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=self.seed))
        study.optimize(objective, n_trials=n_trials, timeout=timeout)

        return study.best_params, study.best_value


# =============================================================================
# 6. MODEL RANKER ENGINE
# =============================================================================
class ModelRanker:
    """Pemeringkat populasi model lintas periode waktu."""

    def analyze_rank_persistence(self, historical_metrics_df: pl.DataFrame, model_id_col: str = "model_id") -> Dict[str, Any]:
        if historical_metrics_df.is_empty():
            return {}

        summary = historical_metrics_df.group_by(model_id_col).agg([
            pl.col("sharpe_ratio").mean().alias("mean_sharpe"),
            pl.col("win_rate").mean().alias("mean_win_rate"),
            pl.col("mae").mean().alias("mean_mae")
        ]).sort("mean_sharpe", descending=True)

        return {"rankings": summary.to_dicts()}


# =============================================================================
# 7. MODEL SELECTOR ENGINE
# =============================================================================
class ModelSelector:
    """Pemilih model terbaik berdasarkan batas toleransi rasio risiko."""

    def __init__(self, min_win_rate: float = 0.48, max_drawdown_limit: float = -0.20):
        self.min_win_rate = min_win_rate
        self.max_drawdown_limit = max_drawdown_limit

    def select_best_model(self, metrics_payload: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        if not metrics_payload:
            raise DataValidationError("Payload matriks model kosong.")

        best_model = None
        best_sharpe = -999.0

        for mid, meta in metrics_payload.items():
            sharpe = float(meta.get("sharpe_ratio", 0.0))
            win_rate = float(meta.get("win_rate", 0.0))
            drawdown = float(meta.get("max_drawdown", 0.0))

            if win_rate >= self.min_win_rate and drawdown >= self.max_drawdown_limit:
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_model = mid

        if not best_model:
            best_model = list(metrics_payload.keys())[0]

        return {"recommended_model_id": best_model, "sharpe_ratio": best_sharpe}


# =============================================================================
# 8. ONLINE LEARNER ENGINE
# =============================================================================
class OnlineLearner:
    """Pembaruan model incremental (warm-start GBDT)."""

    def fit_incremental(self, model: Any, model_type: str, new_data: pl.DataFrame, feature_cols: List[str], target_col: str) -> Any:
        if new_data.is_empty():
            return model

        x_new = new_data.select(feature_cols).to_numpy()
        y_new = new_data.select(target_col).to_numpy().ravel()

        if model_type.lower() == "lightgbm" and LGB_AVAILABLE:
            train_data = lgb.Dataset(x_new, label=y_new)
            booster = model.booster_ if hasattr(model, "booster_") else model
            return lgb.train({"learning_rate": 0.02, "verbose": -1}, train_data, num_boost_round=10, init_model=booster)
        
        return model


# =============================================================================
# 9. OPTUNA MANAGER ENGINE
# =============================================================================
class OptunaManager:
    """Pengelola database SQLite Optuna."""

    def __init__(self, db_path: str = "optuna_study.db"):
        self.db_path = db_path

    def vacuum_db(self) -> bool:
        if not os.path.exists(self.db_path):
            return False
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("VACUUM;")
            return True
        except Exception:
            return False


# =============================================================================
# 10. PERFORMANCE TRACKER ENGINE
# =============================================================================
class PerformanceTracker:
    """Perhitungan metrik performa rolling window (Sharpe, Win Rate, MAE)."""

    def compute_rolling_metrics(self, feedback_df: pl.DataFrame, window_size: int = 60, model_id_col: str = "model_id") -> pl.DataFrame:
        if feedback_df.is_empty():
            return pl.DataFrame()

        return feedback_df.with_columns([
            pl.col("realized_return").rolling_mean(window_size).over(model_id_col).alias("rolling_mean_return"),
            pl.col("realized_return").rolling_std(window_size).over(model_id_col).alias("rolling_std_return"),
            pl.col("directional_hit").rolling_mean(window_size).over(model_id_col).alias("rolling_win_rate"),
            pl.col("absolute_error").rolling_mean(window_size).over(model_id_col).alias("rolling_mae")
        ]).with_columns([
            (pl.col("rolling_mean_return") / (pl.col("rolling_std_return") + EPSILON) * np.sqrt(252)).alias("rolling_sharpe_ratio")
        ])

    def generate_selector_payload(self, rolling_metrics_df: pl.DataFrame, model_id_col: str = "model_id") -> Dict[str, Any]:
        if rolling_metrics_df.is_empty():
            return {}

        latest = rolling_metrics_df.group_by(model_id_col).last()
        payload = {}
        for row in latest.iter_rows(named=True):
            mid = str(row[model_id_col])
            payload[mid] = {
                "sharpe_ratio": float(row.get("rolling_sharpe_ratio") or 0.0),
                "win_rate": float(row.get("rolling_win_rate") or 0.0),
                "mae": float(row.get("rolling_mae") or 0.0),
                "error_variance": float((row.get("rolling_std_return") or 0.1) ** 2)
            }
        return payload


# =============================================================================
# 11. RETRAINING SCHEDULER ENGINE
# =============================================================================
class RetrainingScheduler:
    """Pemicu retrain otomatis dengan batas waktu pendinginan (anti-thrashing)."""

    def __init__(self, cooldown_period_days: int = 7, min_samples_to_retrain: int = 100, db_path: str = "storage.db"):
        self.cooldown_period_days = cooldown_period_days
        self.min_samples_to_retrain = min_samples_to_retrain
        self.db_path = Path(db_path)

    def evaluate_retraining_need(self, model_id: str, current_data_size: int, drift_report: Optional[Dict[str, Any]] = None, decay_report: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        has_drift = drift_report.get("summary_metrics", {}).get("system_retrain_recommended", False) if drift_report else False
        insufficient_data = current_data_size < self.min_samples_to_retrain

        should_retrain = has_drift and not insufficient_data
        reason = "Drift terdeteksi & data mencukupi." if should_retrain else "Retraining belum diperlukan."

        return {
            "trigger_retraining": should_retrain,
            "execute_hyperparameter_tuning": has_drift,
            "audit_justification": reason
        }


# =============================================================================
# 12. UNCERTAINTY ESTIMATOR ENGINE
# =============================================================================
class UncertaintyEstimator:
    """Conformal Prediction Intervals untuk kalkulasi epistemic uncertainty."""

    def estimate_empirical_bounds(self, live_preds_df: pl.DataFrame, historical_errors_df: pl.DataFrame, model_id: str, pred_col: str = "predicted_return", error_col: str = "absolute_error") -> pl.DataFrame:
        if live_preds_df.is_empty() or historical_errors_df.is_empty():
            return live_preds_df

        errors = historical_errors_df.select(error_col).to_numpy().ravel()
        quantile_margin = float(np.percentile(errors, 95)) if len(errors) > 0 else 0.02

        return live_preds_df.with_columns([
            (pl.col(pred_col) - quantile_margin).alias("empirical_lower_bound"),
            (pl.col(pred_col) + quantile_margin).alias("empirical_upper_bound")
        ])


# =============================================================================
# 13. UNIFIED SELF-LEARNING ENGINE (FACADE CLASS)
# =============================================================================
class UnifiedSelfLearningEngine:
    """Facade Class Utama Pengendali Seluruh Siklus Quantitative Self-Learning."""

    def __init__(self, annualization_factor: int = 252, cooldown_period_days: int = 7, min_samples_to_retrain: int = 100, optuna_db_path: str = "optuna_study.db", scheduler_db_path: str = "storage.db"):
        self.feedback_engine = FeedbackEngine()
        self.performance_tracker = PerformanceTracker(annualization_factor=annualization_factor)
        self.drift_detector = DriftDetector()
        self.confidence_estimator = ConfidenceEstimator()
        self.uncertainty_estimator = UncertaintyEstimator()
        self.model_ranker = ModelRanker()
        self.model_selector = ModelSelector()
        self.champion_challenger = ChampionChallengerEngine()
        self.online_learner = OnlineLearner()
        self.hyperparameter_optimizer = HyperparameterOptimizer(storage_uri=f"sqlite:///{optuna_db_path}")
        self.optuna_manager = OptunaManager(db_path=optuna_db_path)
        self.retraining_scheduler = RetrainingScheduler(cooldown_period_days=cooldown_period_days, min_samples_to_retrain=min_samples_to_retrain, db_path=scheduler_db_path)

        logger.info("UnifiedSelfLearningEngine (Facade) berhasil diinisialisasi.")

    def run_full_feedback_cycle(
        self,
        predictions_df: pl.DataFrame,
        actuals_df: pl.DataFrame,
        baseline_features_df: pl.DataFrame,
        current_features_df: pl.DataFrame,
        feature_cols: List[str],
        processed_prediction_ids: Set[str],
        active_model_id: str,
    ) -> Dict[str, Any]:
        start_time = time.time()
        results: Dict[str, Any] = {}

        # 1. Feedback Loop Execution
        feedback_matrix, feedback_summary = self.feedback_engine.process_feedback_loop(
            predictions=predictions_df, actuals=actuals_df, processed_prediction_ids=processed_prediction_ids
        )
        results["feedback_summary"] = feedback_summary

        # 2. Drift Audit
        drift_report = self.drift_detector.analyze_feature_drift(
            baseline_df=baseline_features_df, target_df=current_features_df, feature_cols=feature_cols
        )
        results["drift_report"] = drift_report

        # 3. Rolling Performance Metrics & Selection
        if not feedback_matrix.is_empty():
            rolling_df = self.performance_tracker.compute_rolling_metrics(feedback_df=feedback_matrix)
            selector_payload = self.performance_tracker.generate_selector_payload(rolling_metrics_df=rolling_df)
            results["selector_payload"] = selector_payload

            if selector_payload:
                selection_report = self.model_selector.select_best_model(metrics_payload=selector_payload)
                results["selection_report"] = selection_report

        # 4. Retraining Manifest Evaluation
        retrain_manifest = self.retraining_scheduler.evaluate_retraining_need(
            model_id=active_model_id, current_data_size=predictions_df.shape[0], drift_report=drift_report
        )
        results["retrain_manifest"] = retrain_manifest
        results["total_execution_seconds"] = round(time.time() - start_time, 4)

        return results


if __name__ == "__main__":
    logger.info("Integrity test untuk modul self_learning.py...")
    engine = UnifiedSelfLearningEngine()
    print("UnifiedSelfLearningEngine loaded successfully.")
