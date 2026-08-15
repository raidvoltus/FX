"""
================================================================================
MODULE      : evaluation.py
DESCRIPTION : Consolidated Quantitative Performance & Risk Evaluation Engine.
VERSION     : 2026.Q3.v14.0.0 (Forex & OANDA Production Edition)
PYTHON      : 3.11+ / 3.12+
ARCHITECTURE: Clean Architecture, Decoupled Quantitative Metrics & Audit Reports
================================================================================
"""

import hashlib
import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import polars as pl
from scipy.stats import kendalltau, norm
from scipy.stats import kurtosis as scipy_kurtosis
from scipy.stats import skew as scipy_skew
from scipy.stats import t as student_t

# =============================================================================
# CONSTANTS & EXCEPTIONS HANDLER
# =============================================================================

EPSILON: float = 1e-8
DEFAULT_ANNUALIZATION_FACTOR: float = 252.0  # 252 Hari Perdagangan Forex

try:
    from logger import get_logger
    logger = get_logger(__name__)
except ImportError:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    logger = logging.getLogger("evaluation")

try:
    from exceptions import DataValidationError, EvaluationError
except ImportError:
    class DataValidationError(Exception):
        """Dilemparkan ketika masukan data tidak memenuhi kualifikasi evaluasi."""
        pass

    class EvaluationError(Exception):
        """Dilemparkan ketika terjadi kegagalan komputasi matematika evaluasi."""
        pass


# =============================================================================
# ENUMS & DATACLASSES (IMMUTABLE AUDIT DATA STRUCTURES)
# =============================================================================

class DownsideDeviationMode(str, Enum):
    """Modus kalkulasi denominator untuk Downside Deviation."""
    POPULATION = "population"
    SAMPLE = "sample"
    DOWNSIDE_ONLY = "downside_only"


class WeightConstraintMode(str, Enum):
    """Modus validasi rentang batas bobot portofolio."""
    LONG_ONLY = "long_only"           # [0.0, 1.0]
    MARKET_NEUTRAL = "market_neutral" # [-1.0, 1.0]
    UNCONSTRAINED = "unconstrained"


# --- Audit Metadata Structures ---

@dataclass(frozen=True)
class AlphaDecayAuditMetadata:
    timestamp: str
    engine_version: str = "7.0.0"
    evaluator_version: str = "v13.7.0"
    combined_audit_hash: str = ""
    rolling_window: int = 63
    ddof: int = 1
    metric_mode: str = "information_ratio"


@dataclass(frozen=True)
class AlphaDecayDetailedReport:
    metric_name: str
    decay_slope: float
    t_statistic: float
    p_value: float
    r_squared: float
    kendall_tau: float
    initial_metric_avg: float
    terminal_metric_avg: float
    total_drop_pct: float
    observations_count: int
    rolling_windows_count: int
    audit: AlphaDecayAuditMetadata


@dataclass(frozen=True)
class CalmarAuditMetadata:
    timestamp: str
    engine_version: str = "5.0.0"
    evaluator_version: str = "v13.4.0"
    strategy_hash: str = ""
    return_type: str = "compounded_geometrics"


@dataclass(frozen=True)
class CalmarDetailedReport:
    metric_name: str
    calmar_ratio: float
    is_infinite_calmar: bool
    annualized_return_cagr: float
    max_drawdown: float
    peak_value: float
    trough_value: float
    peak_index: int
    trough_index: int
    annualization_factor: float
    observations_count: int
    audit: CalmarAuditMetadata


@dataclass(frozen=True)
class DSRAuditMetadata:
    timestamp: str
    engine_version: str = "3.0.0"
    evaluator_version: str = "v13.2.0"
    strategy_hash: str = ""
    num_trials: int = 1


@dataclass(frozen=True)
class DSRDetailedReport:
    metric_name: str
    dsr_probability: float
    original_raw_sharpe: float
    original_annualized_sharpe: float
    expected_max_raw_sharpe: float
    asymptotic_variance_raw_sr: float
    z_score: float
    sample_skewness: float
    sample_kurtosis: float
    variance_of_trials_raw: float
    observations_count: int
    audit: DSRAuditMetadata


@dataclass(frozen=True)
class ExpectancyAuditMetadata:
    timestamp: str
    engine_version: str = "6.0.0"
    evaluator_version: str = "v13.6.0"
    combined_audit_hash: str = ""
    execution_status: str = "success"


@dataclass(frozen=True)
class ExpectancyDetailedReport:
    metric_name: str
    expectancy_value: float
    profit_factor: float
    hit_ratio: float
    avg_win: float
    avg_loss: float
    win_loss_ratio: float
    total_trades_count: int
    winning_trades_count: int
    losing_trades_count: int
    audit: ExpectancyAuditMetadata


@dataclass(frozen=True)
class HitRatioAuditMetadata:
    timestamp: str
    engine_version: str = "6.0.0"
    evaluator_version: str = "v13.5.0"
    combined_audit_hash: str = ""
    threshold_type: str = "static"
    execution_status: str = "success"


@dataclass(frozen=True)
class HitRatioDetailedReport:
    metric_name: str
    total_hit_ratio: float
    long_hit_ratio: float
    short_hit_ratio: float
    total_trades_count: int
    winning_trades_count: int
    losing_trades_count: int
    long_trades_count: int
    long_wins_count: int
    short_trades_count: int
    short_wins_count: int
    audit: HitRatioAuditMetadata


@dataclass(frozen=True)
class ISAuditMetadata:
    timestamp: str
    engine_version: str = "8.0.0"
    evaluator_version: str = "v13.9.0"
    combined_audit_hash: str = ""
    bps_multiplier: float = 10000.0


@dataclass(frozen=True)
class ISDetailedReport:
    metric_name: str
    total_implementation_shortfall_bps: float
    explicit_costs_bps: float
    execution_slippage_bps: float
    opportunity_costs_bps: float
    paper_return_nominal: float
    actual_return_nominal: float
    total_shares_ordered: float
    total_shares_executed: float
    execution_rate_pct: float
    observations_count: int
    audit: ISAuditMetadata


@dataclass(frozen=True)
class SharpeAuditMetadata:
    timestamp: str
    engine_version: str = "2.0.0"
    evaluator_version: str = "v13.1.0"
    observation_hash: str = ""
    risk_free_rate_type: str = "static"


@dataclass(frozen=True)
class SharpeDetailedReport:
    metric_name: str
    sharpe_ratio: float
    mean_excess_return_per_period: float
    volatility_per_period: float
    annualized_excess_return: float
    annualized_volatility: float
    annualization_factor: float
    observations_count: int
    audit: SharpeAuditMetadata


@dataclass(frozen=True)
class SortinoAuditMetadata:
    timestamp: str
    engine_version: str = "4.0.0"
    evaluator_version: str = "v13.3.0"
    strategy_hash: str = ""
    target_return_type: str = "static"
    return_type: str = "arithmetic"


@dataclass(frozen=True)
class SortinoDetailedReport:
    metric_name: str
    sortino_ratio: float
    is_infinite_sortino: bool
    mean_excess_return_per_period: float
    downside_deviation_per_period: float
    annualized_excess_return: float
    annualized_downside_volatility: float
    annualization_factor: float
    downside_mode: str
    observations_count: int
    downside_observations_count: int
    audit: SortinoAuditMetadata


@dataclass(frozen=True)
class TurnoverAuditMetadata:
    timestamp: str
    engine_version: str = "8.0.0"
    evaluator_version: str = "v13.8.0"
    combined_audit_hash: str = ""
    periods_per_year: float = 252.0
    ddof: int = 1
    weight_constraint: str = "long_only"


@dataclass(frozen=True)
class TurnoverDetailedReport:
    metric_name: str
    mean_period_turnover: float
    annualized_turnover: float
    mean_long_increase: float
    mean_long_decrease: float
    mean_short_increase: float
    mean_short_decrease: float
    max_period_turnover: float
    median_period_turnover: float
    percentile_95_turnover: float
    turnover_volatility: float
    observations_count: int
    assets_count: int
    audit: TurnoverAuditMetadata


# =============================================================================
# HELPER UTILITIES (_EvaluationUtils)
# =============================================================================

class _EvaluationUtils:
    """Helper internal untuk validasi data, mitigasi float, dan hashing audit."""

    @staticmethod
    def generate_sha256_hash(data: Union[pl.Series, pl.DataFrame, np.ndarray, str]) -> str:
        try:
            hasher = hashlib.sha256()
            if isinstance(data, (pl.Series, pl.DataFrame)):
                for col in (data.columns if isinstance(data, pl.DataFrame) else [data.name]):
                    s = data if isinstance(data, pl.Series) else data.get_column(col)
                    arr = s.to_numpy()
                    hasher.update(arr.tobytes())
                    hasher.update(str(arr.dtype).encode("utf-8"))
            elif isinstance(data, np.ndarray):
                hasher.update(data.tobytes())
            else:
                hasher.update(str(data).encode("utf-8"))
            hasher.update(b"v2026.Q3.ForexEngine")
            return hasher.hexdigest()
        except Exception:
            return "hash_generation_fallback"

    @staticmethod
    def validate_series(name: str, series: pl.Series, min_len: int = 1, check_binary: bool = False) -> pl.Series:
        if series.is_empty() or len(series) < min_len:
            raise DataValidationError(f"Input [{name}] memerlukan minimal {min_len} observasi.")

        if series.null_count() > 0 or series.is_nan().sum() > 0:
            raise DataValidationError(f"Ditemukan kontaminasi nilai Null atau NaN pada [{name}].")

        if series.is_infinite().sum() > 0:
            raise DataValidationError(f"Ditemukan nilai tak terhingga (Inf/-Inf) pada [{name}].")

        if check_binary:
            if not series.is_in([-1, 0, 1]).all():
                raise DataValidationError(f"Vektor arah posisi [{name}] harus eksklusif -1, 0, atau 1.")
        else:
            if series.dtype not in (pl.Float32, pl.Float64):
                series = series.cast(pl.Float64)

        return series


# =============================================================================
# INDIVIDUAL EVALUATOR ENGINES
# =============================================================================

class AlphaDecayEvaluator:
    """Evaluator Laju Peluruhan Performa Strategi (Alpha Decay)."""

    def __init__(self, default_rolling_window: int = 63) -> None:
        if default_rolling_window < 10:
            raise DataValidationError("Default rolling_window harus minimal 10 observasi.")
        self._default_rolling_window = default_rolling_window
        self._latest_reports: Dict[str, AlphaDecayDetailedReport] = {}

    def _calculate_decay_statistics(self, metrics_series: np.ndarray) -> Tuple[float, float, float, float, float]:
        m = len(metrics_series)
        x = np.arange(m, dtype=np.float64)
        x_mean, y_mean = np.mean(x), np.mean(metrics_series)
        x_diff, y_diff = x - x_mean, metrics_series - y_mean
        
        num, den = np.sum(x_diff * y_diff), np.sum(x_diff ** 2)
        if den < EPSILON:
            return 0.0, 0.0, 1.0, 0.0, 0.0

        slope = float(num / den)
        y_pred = slope * x + (y_mean - slope * x_mean)
        residuals = metrics_series - y_pred
        rss, tss = np.sum(residuals ** 2), np.sum(y_diff ** 2)
        r_squared = float(1.0 - (rss / tss)) if tss > EPSILON else 0.0

        tau_stat, _ = kendalltau(x, metrics_series)
        tau_stat = float(tau_stat) if math.isfinite(tau_stat) else 0.0

        if m <= 2:
            return slope, 0.0, 1.0, r_squared, tau_stat

        s_squared = rss / (m - 2)
        se_slope = math.sqrt(s_squared / den) if s_squared > 0 else 0.0
        if se_slope < EPSILON:
            return slope, 0.0, 1.0, r_squared, tau_stat

        t_stat = slope / se_slope
        p_val = float(2.0 * student_t.sf(abs(t_stat), df=m - 2))
        return slope, t_stat, p_val, r_squared, tau_stat

    def compute(self, returns: pl.Series, benchmark: Optional[pl.Series] = None, rolling_window: Optional[int] = None) -> float:
        window = rolling_window or self._default_rolling_window
        min_required = window * 3
        v_returns = _EvaluationUtils.validate_series("returns", returns, min_required)
        strategy_name = v_returns.name or "default_strategy"

        if benchmark is not None:
            if len(benchmark) != len(returns):
                raise DataValidationError("Dimensi returns dan benchmark tidak sepadan.")
            v_bench = _EvaluationUtils.validate_series("benchmark", benchmark, min_required)
            active_series = v_returns - v_bench
            metric_mode_str = "information_ratio"
        else:
            active_series = v_returns
            metric_mode_str = "sharpe_like"

        try:
            df_local = pl.DataFrame([active_series.alias("target")])
            rolling_stats = df_local.select([
                pl.col("target").rolling_mean(window_size=window).alias("r_mean"),
                pl.col("target").rolling_std(window_size=window, ddof=1).alias("r_std")
            ]).drop_nulls()

            r_mean = rolling_stats.get_column("r_mean").to_numpy()
            r_std = rolling_stats.get_column("r_std").to_numpy()
            valid_mask = r_std > EPSILON

            if not np.any(valid_mask):
                raise EvaluationError(f"Volatilitas lokal 0 pada strategi [{strategy_name}].")

            metrics_series = np.where(valid_mask, r_mean / r_std, 0.0)
            slope, t_stat, p_val, r_squared, kendall_tau = self._calculate_decay_statistics(metrics_series)

            segment_len = max(len(metrics_series) // 3, 1)
            initial_avg = float(np.mean(metrics_series[:segment_len]))
            terminal_avg = float(np.mean(metrics_series[-segment_len:]))
            total_drop = float("nan") if abs(initial_avg) < EPSILON else float(((terminal_avg - initial_avg) / abs(initial_avg)) * 100.0)

            audit_hash = _EvaluationUtils.generate_sha256_hash(v_returns)
            audit_meta = AlphaDecayAuditMetadata(
                timestamp=datetime.now(timezone.utc).isoformat(),
                combined_audit_hash=audit_hash,
                rolling_window=window,
                metric_mode=metric_mode_str
            )

            self._latest_reports[strategy_name] = AlphaDecayDetailedReport(
                metric_name="Alpha Decay Slope",
                decay_slope=slope, t_statistic=t_stat, p_value=p_val, r_squared=r_squared,
                kendall_tau=kendall_tau, initial_metric_avg=initial_avg, terminal_metric_avg=terminal_avg,
                total_drop_pct=total_drop, observations_count=len(v_returns),
                rolling_windows_count=len(metrics_series), audit=audit_meta
            )
            return slope
        except Exception as exc:
            raise EvaluationError(f"Kegagalan Alpha Decay: {str(exc)}") from exc

    def get_detailed_report(self, strategy_name: str = "default_strategy") -> AlphaDecayDetailedReport:
        if strategy_name not in self._latest_reports:
            raise EvaluationError(f"Belum ada data evaluasi Alpha Decay untuk '{strategy_name}'.")
        return self._latest_reports[strategy_name]


class CalmarRatioEvaluator:
    """Evaluator Rasio Efisiensi CAGR terhadap Maximum Drawdown."""

    def __init__(self, default_annualization_factor: float = DEFAULT_ANNUALIZATION_FACTOR) -> None:
        if default_annualization_factor <= EPSILON:
            raise DataValidationError("Annualization factor harus > 0.")
        self._annualization_factor = default_annualization_factor
        self._latest_reports: Dict[str, CalmarDetailedReport] = {}

    def compute(self, returns: pl.Series, annualization_factor: Optional[float] = None) -> float:
        ann_factor = annualization_factor or self._annualization_factor
        v_returns = _EvaluationUtils.validate_series("returns", returns, min_len=2)
        strategy_name = v_returns.name or "default_strategy"

        if (v_returns <= -1.0).sum() > 0:
            raise DataValidationError("Ditemukan return <= -100% (Kehancuran Modal Total).")

        try:
            log_returns = np.log1p(v_returns.to_numpy())
            equity_curve = np.exp(np.cumsum(log_returns))
            n_obs = len(equity_curve)

            max_dd, peak = 0.0, 1.0
            peak_idx, trough_idx = -1, -1
            current_peak_idx = -1
            saved_peak, saved_trough = 1.0, 1.0

            for i in range(n_obs):
                val = equity_curve[i]
                if val > peak:
                    peak = val
                    current_peak_idx = i
                if peak > EPSILON:
                    dd = (peak - val) / peak
                    if dd > max_dd:
                        max_dd = dd
                        peak_idx = current_peak_idx
                        trough_idx = i
                        saved_peak, saved_trough = peak, val

            total_multiplier = float(equity_curve[-1])
            ann_return_cagr = -1.0 if total_multiplier <= EPSILON else math.pow(total_multiplier, float(ann_factor / n_obs)) - 1.0

            is_inf = False
            if max_dd < EPSILON:
                calmar_ratio = float("inf") if ann_return_cagr > 0 else 0.0
                is_inf = ann_return_cagr > 0
            else:
                calmar_ratio = ann_return_cagr / max_dd

            audit_meta = CalmarAuditMetadata(
                timestamp=datetime.now(timezone.utc).isoformat(),
                strategy_hash=_EvaluationUtils.generate_sha256_hash(v_returns)
            )

            self._latest_reports[strategy_name] = CalmarDetailedReport(
                metric_name="Calmar Ratio", calmar_ratio=calmar_ratio, is_infinite_calmar=is_inf,
                annualized_return_cagr=ann_return_cagr, max_drawdown=max_dd,
                peak_value=float(saved_peak), trough_value=float(saved_trough),
                peak_index=int(peak_idx), trough_index=int(trough_idx),
                annualization_factor=ann_factor, observations_count=n_obs, audit=audit_meta
            )
            return calmar_ratio
        except Exception as exc:
            raise EvaluationError(f"Kegagalan Calmar Ratio: {str(exc)}") from exc

    def get_detailed_report(self, strategy_name: str = "default_strategy") -> CalmarDetailedReport:
        if strategy_name not in self._latest_reports:
            raise EvaluationError(f"Belum ada data Calmar untuk '{strategy_name}'.")
        return self._latest_reports[strategy_name]


class DeflatedSharpeEvaluator:
    """Evaluator Deflated Sharpe Ratio (DSR) berbasis Bailey-López de Prado."""

    _EULER_MASCHERONI: float = 0.5772156649015328

    def __init__(self, default_annualization_factor: float = DEFAULT_ANNUALIZATION_FACTOR) -> None:
        self._annualization_factor = default_annualization_factor
        self._latest_reports: Dict[str, DSRDetailedReport] = {}

    def compute(
        self,
        returns: pl.Series,
        num_trials: int = 1,
        variance_of_trials_raw: float = 0.0,
        risk_free_rate_period: float = 0.0,
        annualization_factor: Optional[float] = None
    ) -> float:
        ann_factor = annualization_factor or self._annualization_factor
        v_returns = _EvaluationUtils.validate_series("returns", returns, min_len=4)
        strategy_name = v_returns.name or "strategy_dsr"

        try:
            excess = (v_returns.cast(pl.Float64) - risk_free_rate_period).to_numpy()
            n_obs = len(excess)
            mean_ex, std_ex = float(np.mean(excess)), float(np.std(excess, ddof=1))

            if std_ex < EPSILON:
                return 0.5

            raw_sr = mean_ex / std_ex
            skew = float(scipy_skew(excess, bias=False))
            kurt = float(scipy_kurtosis(excess, fisher=False, bias=False))

            expected_max_raw_sr = 0.0
            if num_trials > 1 and variance_of_trials_raw > EPSILON:
                std_trials = math.sqrt(variance_of_trials_raw)
                term1 = (1.0 - self._EULER_MASCHERONI) * norm.ppf(1.0 - 1.0 / num_trials)
                term2 = self._EULER_MASCHERONI * norm.ppf(1.0 - 1.0 / (num_trials * math.e))
                expected_max_raw_sr = float((term1 + term2) * std_trials)

            var_sr_raw = (1.0 - skew * raw_sr + ((kurt - 1.0) / 4.0) * (raw_sr ** 2)) / (n_obs - 1)
            var_sr_raw = max(var_sr_raw, EPSILON)
            z_score = (raw_sr - expected_max_raw_sr) / math.sqrt(var_sr_raw)
            dsr_prob = float(norm.cdf(z_score))

            audit_meta = DSRAuditMetadata(
                timestamp=datetime.now(timezone.utc).isoformat(),
                strategy_hash=_EvaluationUtils.generate_sha256_hash(v_returns),
                num_trials=num_trials
            )

            self._latest_reports[strategy_name] = DSRDetailedReport(
                metric_name="Deflated Sharpe Ratio Probability", dsr_probability=dsr_prob,
                original_raw_sharpe=raw_sr, original_annualized_sharpe=raw_sr * math.sqrt(ann_factor),
                expected_max_raw_sharpe=expected_max_raw_sr, asymptotic_variance_raw_sr=var_sr_raw,
                z_score=z_score, sample_skewness=skew, sample_kurtosis=kurt,
                variance_of_trials_raw=variance_of_trials_raw, observations_count=n_obs, audit=audit_meta
            )
            return dsr_prob
        except Exception as exc:
            raise EvaluationError(f"Kegagalan DSR: {str(exc)}") from exc

    def get_detailed_report(self, strategy_name: str = "strategy_dsr") -> DSRDetailedReport:
        if strategy_name not in self._latest_reports:
            raise EvaluationError(f"Belum ada data DSR untuk '{strategy_name}'.")
        return self._latest_reports[strategy_name]


class ExpectancyEvaluator:
    """Evaluator Ekspektasi Matematik Profit/Loss per Transaksi (Model Tharp)."""

    def __init__(self) -> None:
        self._latest_reports: Dict[str, ExpectancyDetailedReport] = {}

    def compute(self, returns: pl.Series, directions: Optional[pl.Series] = None) -> float:
        v_returns = _EvaluationUtils.validate_series("returns", returns, min_len=1)
        n_obs = len(v_returns)

        if directions is not None:
            if len(directions) != n_obs:
                raise DataValidationError("Dimensi returns dan directions tidak sepadan.")
            v_dirs = _EvaluationUtils.validate_series("directions", directions, check_binary=True).cast(pl.Int8)
        else:
            v_dirs = pl.Series("directions", np.ones(n_obs, dtype=np.int8))

        strategy_name = v_returns.name or "default_strategy"

        try:
            ret_arr, dir_arr = v_returns.to_numpy(), v_dirs.to_numpy()
            active_mask = dir_arr != 0
            total_trades = int(np.sum(active_mask))

            audit_hash = _EvaluationUtils.generate_sha256_hash(v_returns)

            if total_trades == 0:
                audit_meta = ExpectancyAuditMetadata(
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    combined_audit_hash=audit_hash, execution_status="no_active_trades"
                )
                self._latest_reports[strategy_name] = ExpectancyDetailedReport(
                    metric_name="Expectancy", expectancy_value=0.0, profit_factor=0.0, hit_ratio=0.0,
                    avg_win=0.0, avg_loss=0.0, win_loss_ratio=0.0, total_trades_count=0,
                    winning_trades_count=0, losing_trades_count=0, audit=audit_meta
                )
                return 0.0

            trade_profits = ret_arr * dir_arr
            win_mask = (trade_profits > 0.0) & active_mask
            loss_mask = (trade_profits <= 0.0) & active_mask

            total_wins, total_losses = int(np.sum(win_mask)), int(np.sum(loss_mask))
            hit_ratio = total_wins / total_trades

            sum_win, sum_loss = float(np.sum(trade_profits[win_mask])), float(np.sum(trade_profits[loss_mask]))
            avg_win = sum_win / total_wins if total_wins > 0 else 0.0
            avg_loss = sum_loss / total_losses if total_losses > 0 else 0.0

            expectancy_value = (hit_ratio * avg_win) + ((1.0 - hit_ratio) * avg_loss)

            win_loss_ratio = (avg_win / abs(avg_loss)) if abs(avg_loss) > EPSILON else (float("inf") if avg_win > 0 else 0.0)
            profit_factor = (sum_win / abs(sum_loss)) if abs(sum_loss) > EPSILON else (float("inf") if sum_win > 0 else 1.0)

            audit_meta = ExpectancyAuditMetadata(
                timestamp=datetime.now(timezone.utc).isoformat(),
                combined_audit_hash=audit_hash
            )

            self._latest_reports[strategy_name] = ExpectancyDetailedReport(
                metric_name="Expectancy", expectancy_value=expectancy_value, profit_factor=profit_factor,
                hit_ratio=hit_ratio, avg_win=avg_win, avg_loss=avg_loss, win_loss_ratio=win_loss_ratio,
                total_trades_count=total_trades, winning_trades_count=total_wins, losing_trades_count=total_losses,
                audit=audit_meta
            )
            return expectancy_value
        except Exception as exc:
            raise EvaluationError(f"Kegagalan Expectancy: {str(exc)}") from exc

    def get_detailed_report(self, strategy_name: str = "default_strategy") -> ExpectancyDetailedReport:
        if strategy_name not in self._latest_reports:
            raise EvaluationError(f"Belum ada data Expectancy untuk '{strategy_name}'.")
        return self._latest_reports[strategy_name]


class HitRatioEvaluator:
    """Evaluator Frekuensi Kemenangan (Win Rate / Hit Ratio) Pasar Forex."""

    def __init__(self, default_threshold: float = 0.0) -> None:
        self._default_threshold = default_threshold
        self._latest_reports: Dict[str, HitRatioDetailedReport] = {}

    def compute(self, returns: pl.Series, directions: Optional[pl.Series] = None, threshold: Optional[float] = None) -> float:
        v_returns = _EvaluationUtils.validate_series("returns", returns, min_len=1)
        n_obs = len(v_returns)

        if directions is not None:
            v_dirs = _EvaluationUtils.validate_series("directions", directions, check_binary=True).cast(pl.Int8)
        else:
            v_dirs = pl.Series("directions", np.ones(n_obs, dtype=np.int8))

        thresh_val = threshold if threshold is not None else self._default_threshold
        strategy_name = v_returns.name or "default_strategy"

        try:
            ret_arr, dir_arr = v_returns.to_numpy(), v_dirs.to_numpy()
            active_mask = dir_arr != 0
            total_trades = int(np.sum(active_mask))
            audit_hash = _EvaluationUtils.generate_sha256_hash(v_returns)

            if total_trades == 0:
                audit_meta = HitRatioAuditMetadata(
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    combined_audit_hash=audit_hash, execution_status="no_active_trades"
                )
                self._latest_reports[strategy_name] = HitRatioDetailedReport(
                    metric_name="Hit Ratio", total_hit_ratio=0.0, long_hit_ratio=0.0, short_hit_ratio=0.0,
                    total_trades_count=0, winning_trades_count=0, losing_trades_count=0,
                    long_trades_count=0, long_wins_count=0, short_trades_count=0, short_wins_count=0,
                    audit=audit_meta
                )
                return 0.0

            trade_profits = ret_arr * dir_arr
            win_mask = (trade_profits > thresh_val) & active_mask
            total_wins = int(np.sum(win_mask))
            total_hit_ratio = total_wins / total_trades

            long_mask = dir_arr == 1
            long_trades = int(np.sum(long_mask))
            long_wins = int(np.sum(win_mask & long_mask))
            long_hit_ratio = long_wins / long_trades if long_trades > 0 else 0.0

            short_mask = dir_arr == -1
            short_trades = int(np.sum(short_mask))
            short_wins = int(np.sum(win_mask & short_mask))
            short_hit_ratio = short_wins / short_trades if short_trades > 0 else 0.0

            audit_meta = HitRatioAuditMetadata(
                timestamp=datetime.now(timezone.utc).isoformat(),
                combined_audit_hash=audit_hash
            )

            self._latest_reports[strategy_name] = HitRatioDetailedReport(
                metric_name="Hit Ratio", total_hit_ratio=total_hit_ratio, long_hit_ratio=long_hit_ratio,
                short_hit_ratio=short_hit_ratio, total_trades_count=total_trades, winning_trades_count=total_wins,
                losing_trades_count=total_trades - total_wins, long_trades_count=long_trades, long_wins_count=long_wins,
                short_trades_count=short_trades, short_wins_count=short_wins, audit=audit_meta
            )
            return total_hit_ratio
        except Exception as exc:
            raise EvaluationError(f"Kegagalan Hit Ratio: {str(exc)}") from exc

    def get_detailed_report(self, strategy_name: str = "default_strategy") -> HitRatioDetailedReport:
        if strategy_name not in self._latest_reports:
            raise EvaluationError(f"Belum ada data Hit Ratio untuk '{strategy_name}'.")
        return self._latest_reports[strategy_name]


class ImplementationShortfallEvaluator:
    """Evaluator Perold Implementation Shortfall (Slippage, Fee, & Opportunity Cost)."""

    _BPS_MULTIPLIER: float = 10000.0

    def __init__(self) -> None:
        self._latest_reports: Dict[str, ISDetailedReport] = {}

    def compute(self, order_data: pl.DataFrame, portfolio_id: str = "default_portfolio") -> float:
        req_cols = ["side", "shares_ordered", "shares_executed", "arrival_price", "execution_price", "explicit_costs", "terminal_price"]
        for c in req_cols:
            if c not in order_data.columns:
                raise DataValidationError(f"IS membutuhkan kolom: {c}")

        v_df = order_data.select([
            pl.col("side").cast(pl.Int8),
            *[pl.col(c).cast(pl.Float64) for c in req_cols if c != "side"]
        ])

        try:
            paper_ret = (v_df["side"] * v_df["shares_ordered"] * (v_df["terminal_price"] - v_df["arrival_price"])).sum()
            actual_ret = ((v_df["side"] * v_df["shares_executed"] * (v_df["terminal_price"] - v_df["execution_price"])) - v_df["explicit_costs"]).sum()
            slippage_nom = (v_df["side"] * v_df["shares_executed"] * (v_df["execution_price"] - v_df["arrival_price"])).sum()
            opp_cost_nom = (v_df["side"] * (v_df["shares_ordered"] - v_df["shares_executed"]) * (v_df["terminal_price"] - v_df["arrival_price"])).sum()

            notional_val = float((v_df["shares_ordered"] * v_df["arrival_price"]).sum())
            if notional_val < EPSILON:
                return 0.0

            total_is_nom = float(paper_ret - actual_ret)
            explicit_nom = float(v_df["explicit_costs"].sum())

            total_is_bps = (total_is_nom / notional_val) * self._BPS_MULTIPLIER
            explicit_bps = (explicit_nom / notional_val) * self._BPS_MULTIPLIER
            slippage_bps = (float(slippage_nom) / notional_val) * self._BPS_MULTIPLIER
            opp_bps = (float(opp_cost_nom) / notional_val) * self._BPS_MULTIPLIER

            total_ordered = float(v_df["shares_ordered"].sum())
            total_executed = float(v_df["shares_executed"].sum())
            exec_rate = (total_executed / total_ordered * 100.0) if total_ordered > 0 else 0.0

            audit_meta = ISAuditMetadata(
                timestamp=datetime.now(timezone.utc).isoformat(),
                combined_audit_hash=_EvaluationUtils.generate_sha256_hash(v_df)
            )

            self._latest_reports[portfolio_id] = ISDetailedReport(
                metric_name="Implementation Shortfall", total_implementation_shortfall_bps=total_is_bps,
                explicit_costs_bps=explicit_bps, execution_slippage_bps=slippage_bps,
                opportunity_costs_bps=opp_bps, paper_return_nominal=float(paper_ret),
                actual_return_nominal=float(actual_ret), total_shares_ordered=total_ordered,
                total_shares_executed=total_executed, execution_rate_pct=exec_rate,
                observations_count=v_df.height, audit=audit_meta
            )
            return total_is_bps
        except Exception as exc:
            raise EvaluationError(f"Kegagalan Implementation Shortfall: {str(exc)}") from exc

    def get_detailed_report(self, portfolio_id: str = "default_portfolio") -> ISDetailedReport:
        if portfolio_id not in self._latest_reports:
            raise EvaluationError(f"Belum ada data IS untuk '{portfolio_id}'.")
        return self._latest_reports[portfolio_id]


class SharpeRatioEvaluator:
    """Evaluator Annualized Sharpe Ratio."""

    def __init__(self, default_annualization_factor: float = DEFAULT_ANNUALIZATION_FACTOR, default_risk_free_rate: float = 0.0) -> None:
        self._annualization_factor = default_annualization_factor
        self._default_rf = default_risk_free_rate
        self._latest_reports: Dict[str, SharpeDetailedReport] = {}

    def compute(self, returns: pl.Series, risk_free_rate: Optional[float] = None, annualization_factor: Optional[float] = None) -> float:
        rf = risk_free_rate if risk_free_rate is not None else self._default_rf
        ann_factor = annualization_factor or self._annualization_factor
        v_returns = _EvaluationUtils.validate_series("returns", returns, min_len=2)
        strategy_name = v_returns.name or "default_strategy"

        excess = v_returns - rf
        try:
            mean_ex = float(excess.mean())
            std_ex = float(excess.std(ddof=1))

            if std_ex < EPSILON:
                sharpe_ratio = 0.0
            else:
                sharpe_ratio = (mean_ex / std_ex) * math.sqrt(ann_factor)

            audit_meta = SharpeAuditMetadata(
                timestamp=datetime.now(timezone.utc).isoformat(),
                observation_hash=_EvaluationUtils.generate_sha256_hash(v_returns)
            )

            self._latest_reports[strategy_name] = SharpeDetailedReport(
                metric_name="Annualized Sharpe Ratio", sharpe_ratio=sharpe_ratio,
                mean_excess_return_per_period=mean_ex, volatility_per_period=std_ex,
                annualized_excess_return=mean_ex * ann_factor,
                annualized_volatility=std_ex * math.sqrt(ann_factor),
                annualization_factor=ann_factor, observations_count=len(v_returns),
                audit=audit_meta
            )
            return sharpe_ratio
        except Exception as exc:
            raise EvaluationError(f"Kegagalan Sharpe Ratio: {str(exc)}") from exc

    def get_detailed_report(self, strategy_name: str = "default_strategy") -> SharpeDetailedReport:
        if strategy_name not in self._latest_reports:
            raise EvaluationError(f"Belum ada data Sharpe untuk '{strategy_name}'.")
        return self._latest_reports[strategy_name]


class SortinoRatioEvaluator:
    """Evaluator Annualized Sortino Ratio (Downside Risk Adjusted)."""

    def __init__(self, default_annualization_factor: float = DEFAULT_ANNUALIZATION_FACTOR, default_target_return: float = 0.0) -> None:
        self._annualization_factor = default_annualization_factor
        self._default_target_return = default_target_return
        self._latest_reports: Dict[str, SortinoDetailedReport] = {}

    def compute(self, returns: pl.Series, target_return: Optional[float] = None, annualization_factor: Optional[float] = None) -> float:
        mar = target_return if target_return is not None else self._default_target_return
        ann_factor = annualization_factor or self._annualization_factor
        v_returns = _EvaluationUtils.validate_series("returns", returns, min_len=2)
        strategy_name = v_returns.name or "default_strategy"

        excess = v_returns - mar
        try:
            excess_arr = excess.to_numpy()
            mean_ex = float(np.mean(excess_arr))

            downside_diff = np.minimum(excess_arr, 0.0)
            downside_obs = int(np.sum(excess_arr < 0.0))
            sum_sq_down = float(np.sum(downside_diff ** 2))

            downside_dev = math.sqrt(sum_sq_down / len(excess_arr))

            is_inf = False
            if downside_dev < EPSILON:
                sortino_ratio = float("inf") if mean_ex > 0 else 0.0
                is_inf = mean_ex > 0
            else:
                sortino_ratio = (mean_ex / downside_dev) * math.sqrt(ann_factor)

            audit_meta = SortinoAuditMetadata(
                timestamp=datetime.now(timezone.utc).isoformat(),
                strategy_hash=_EvaluationUtils.generate_sha256_hash(v_returns)
            )

            self._latest_reports[strategy_name] = SortinoDetailedReport(
                metric_name="Annualized Sortino Ratio", sortino_ratio=sortino_ratio,
                is_infinite_sortino=is_inf, mean_excess_return_per_period=mean_ex,
                downside_deviation_per_period=downside_dev, annualized_excess_return=mean_ex * ann_factor,
                annualized_downside_volatility=downside_dev * math.sqrt(ann_factor),
                annualization_factor=ann_factor, downside_mode="population",
                observations_count=len(v_returns), downside_observations_count=downside_obs,
                audit=audit_meta
            )
            return sortino_ratio
        except Exception as exc:
            raise EvaluationError(f"Kegagalan Sortino Ratio: {str(exc)}") from exc

    def get_detailed_report(self, strategy_name: str = "default_strategy") -> SortinoDetailedReport:
        if strategy_name not in self._latest_reports:
            raise EvaluationError(f"Belum ada data Sortino untuk '{strategy_name}'.")
        return self._latest_reports[strategy_name]


class PortfolioTurnoverEvaluator:
    """Evaluator Portfolio Turnover Rate Multi-Aset Forex."""

    def __init__(self, periods_per_year: float = DEFAULT_ANNUALIZATION_FACTOR) -> None:
        self._periods_per_year = periods_per_year
        self._latest_reports: Dict[str, TurnoverDetailedReport] = {}

    def compute(self, portfolio_weights: pl.DataFrame, portfolio_name: str = "default_portfolio") -> float:
        if portfolio_weights.width == 0 or portfolio_weights.height < 2:
            raise DataValidationError("Turnover membutuhkan minimal 1 kolom aset dan 2 observasi waktu.")

        v_df = portfolio_weights.select([pl.col(c).cast(pl.Float64) for c in portfolio_weights.columns])

        try:
            diff_exprs = [pl.col(c).diff().alias(f"{c}_diff") for c in v_df.columns]
            transform_df = v_df.select(diff_exprs).drop_nulls()

            abs_diffs = [pl.col(f"{c}_diff").abs() for c in v_df.columns]
            period_turnover = (0.5 * pl.sum_horizontal(abs_diffs)).alias("pt")

            turnover_series = transform_df.select([period_turnover]).get_column("pt").to_numpy()
            mean_turnover = float(np.mean(turnover_series))
            ann_turnover = mean_turnover * self._periods_per_year
            max_turnover = float(np.max(turnover_series))
            vol_turnover = float(np.std(turnover_series, ddof=1)) if len(turnover_series) > 1 else 0.0

            audit_meta = TurnoverAuditMetadata(
                timestamp=datetime.now(timezone.utc).isoformat(),
                combined_audit_hash=_EvaluationUtils.generate_sha256_hash(v_df),
                periods_per_year=self._periods_per_year
            )

            self._latest_reports[portfolio_name] = TurnoverDetailedReport(
                metric_name="Portfolio Turnover Rate", mean_period_turnover=mean_turnover,
                annualized_turnover=ann_turnover, mean_long_increase=mean_turnover * 0.5,
                mean_long_decrease=mean_turnover * 0.5, mean_short_increase=0.0, mean_short_decrease=0.0,
                max_period_turnover=max_turnover, median_period_turnover=float(np.median(turnover_series)),
                percentile_95_turnover=float(np.percentile(turnover_series, 95)),
                turnover_volatility=vol_turnover, observations_count=v_df.height,
                assets_count=v_df.width, audit=audit_meta
            )
            return mean_turnover
        except Exception as exc:
            raise EvaluationError(f"Kegagalan Turnover: {str(exc)}") from exc

    def get_detailed_report(self, portfolio_name: str = "default_portfolio") -> TurnoverDetailedReport:
        if portfolio_name not in self._latest_reports:
            raise EvaluationError(f"Belum ada data Turnover untuk '{portfolio_name}'.")
        return self._latest_reports[portfolio_name]


# =============================================================================
# FACADE CLASS: UnifiedEvaluationEngine
# =============================================================================

class UnifiedEvaluationEngine:
    """
    Kelas Facade terpusat untuk mengeksekusi seluruh pipa evaluasi kuantitatif
    (Sharpe, Sortino, Calmar, Expectancy, Win Rate, DSR, Alpha Decay, Turnover)
    hanya dengan satu pemanggilan method.
    """

    def __init__(
        self,
        annualization_factor: float = DEFAULT_ANNUALIZATION_FACTOR,
        risk_free_rate: float = 0.0,
        rolling_window_alpha: int = 63
    ) -> None:
        self.sharpe_evaluator = SharpeRatioEvaluator(default_annualization_factor=annualization_factor, default_risk_free_rate=risk_free_rate)
        self.sortino_evaluator = SortinoRatioEvaluator(default_annualization_factor=annualization_factor)
        self.calmar_evaluator = CalmarRatioEvaluator(default_annualization_factor=annualization_factor)
        self.expectancy_evaluator = ExpectancyEvaluator()
        self.hit_ratio_evaluator = HitRatioEvaluator()
        self.dsr_evaluator = DeflatedSharpeEvaluator(default_annualization_factor=annualization_factor)
        self.alpha_decay_evaluator = AlphaDecayEvaluator(default_rolling_window=rolling_window_alpha)
        self.is_evaluator = ImplementationShortfallEvaluator()
        self.turnover_evaluator = PortfolioTurnoverEvaluator(periods_per_year=annualization_factor)

        logger.info("UnifiedEvaluationEngine (Forex Edition) berhasil diinisialisasi.")

    def evaluate_strategy(
        self,
        returns: pl.Series,
        benchmark: Optional[pl.Series] = None,
        directions: Optional[pl.Series] = None,
        strategy_name: str = "forex_strategy"
    ) -> Dict[str, Any]:
        """
        Menjalankan evaluasi kuantitatif lengkap untuk satu deret waktu return strategi.
        """
        strategy_returns = returns.alias(strategy_name)

        sharpe = self.sharpe_evaluator.compute(strategy_returns)
        sortino = self.sortino_evaluator.compute(strategy_returns)
        calmar = self.calmar_evaluator.compute(strategy_returns)
        expectancy = self.expectancy_evaluator.compute(strategy_returns, directions=directions)
        hit_ratio = self.hit_ratio_evaluator.compute(strategy_returns, directions=directions)
        dsr_prob = self.dsr_evaluator.compute(strategy_returns)

        alpha_decay_slope = 0.0
        try:
            alpha_decay_slope = self.alpha_decay_evaluator.compute(strategy_returns, benchmark=benchmark)
        except DataValidationError:
            logger.warning(f"Jumlah sampel ({len(returns)}) tidak memenuhi syarat minimal Alpha Decay.")

        return {
            "strategy_name": strategy_name,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "calmar_ratio": calmar,
            "expectancy_value": expectancy,
            "hit_ratio_win_rate": hit_ratio,
            "dsr_probability": dsr_prob,
            "alpha_decay_slope": alpha_decay_slope,
            "evaluation_timestamp": datetime.now(timezone.utc).isoformat()
        }

    def get_full_reports(self, strategy_name: str = "forex_strategy") -> Dict[str, Any]:
        """Mengambil seluruh objek laporan audit imutabel detail per metrik."""
        reports = {}
        for name, eval_obj in [
            ("sharpe", self.sharpe_evaluator),
            ("sortino", self.sortino_evaluator),
            ("calmar", self.calmar_evaluator),
            ("expectancy", self.expectancy_evaluator),
            ("hit_ratio", self.hit_ratio_evaluator),
            ("dsr", self.dsr_evaluator),
            ("alpha_decay", self.alpha_decay_evaluator)
        ]:
            try:
                reports[name] = eval_obj.get_detailed_report(strategy_name)
            except EvaluationError:
                reports[name] = None
        return reports


if __name__ == "__main__":
    logger.info("Pengujian integritas modul evaluation.py...")
    engine = UnifiedEvaluationEngine()
    mock_returns = pl.Series("eur_usd_strategy", np.random.normal(0.0005, 0.01, 100))
    metrics = engine.evaluate_strategy(mock_returns)
    print("Hasil Evaluasi Kuantitatif Strategy:", metrics)

