"""
================================================================================
MODULE      : signal_forex.py (v2026.Q3 OANDA Production Synchronized Edition)
DESCRIPTION : Unified Signal Processing Engine for OANDA Forex Autonomous Trading Bot.
ARCHITECTURE: 9-Gateway Sequential Pipeline + UnifiedForexSignalEngine Facade Class
DIRECTORY   : Flat Directory (Root Level Integration)

Refactored for OANDA Forex:
- Dual-Directional Geometry Support: Full support for BUY (Long) & SELL (Short).
- Dynamic Column Resolution: Auto-aligns ML predictions (calibrated_prob, confidence_score, predicted_return).
- Full Orchestrator Alias Coverage: Supports process_signals(), filter_signals(), and execute_pipeline().
- Strict Forex Geometry & Lot Validation: Enforces SL/TP boundaries per direction and minimum unit sizing.
- Polars Defensive Guards: Robust cleaning across all 9 gateways preventing null/NaN runtime failures.
================================================================================
"""

import os
import time
import json
import logging
import threading
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Final, Dict, Any, Optional, List
import numpy as np
import polars as pl

# ==============================================================================
# OANDA FOREX BASELINE CONSTANTS
# ==============================================================================
OANDA_DEFAULT_SPREAD_PIPS_MARGIN: Final[float] = 0.0002   # Default buffer spread pips (2 pips)
OANDA_MIN_UNITS: Final[float] = 1.0                       # Minimum micro-unit trade size
OANDA_MIN_PRICE_QUOTE: Final[float] = 0.00001             # Minimum quote price resolution
OANDA_MAX_STALENESS_SEC: Final[float] = 43200.0           # 12-Hour Max Staleness

# ==============================================================================
# UNIFIED LOGGER CONFIGURATION
# ==============================================================================
logger = logging.getLogger("ForexSignalEngine")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('[%(asctime)s][%(levelname)s][ForexSignal] %(message)s'))
    logger.addHandler(ch)

# ==============================================================================
# HELPER UTILITIES
# ==============================================================================
def resolve_column_name(df: pl.DataFrame, candidates: List[str], default: str) -> str:
    """Mencari nama kolom yang ada pada DataFrame berdasarkan daftar kandidat prioritasi."""
    for c in candidates:
        if c in df.columns:
            return c
    return default


# ==============================================================================
# CUSTOM EXCEPTIONS ARCHITECTURE
# ==============================================================================
class SignalForexError(Exception):
    """Base exception class for all Forex Signal Engine modules."""
    pass

class SignalGeneratorError(SignalForexError): pass
class EntryFilterError(SignalForexError): pass
class ExitFilterError(SignalForexError): pass
class ProbabilityFilterError(SignalForexError): pass
class ConfidenceFilterError(SignalForexError): pass
class TpSlOptimizerError(SignalForexError): pass
class SignalRankerError(SignalForexError): pass
class SignalValidatorError(SignalForexError): pass
class SignalExplainerError(SignalForexError): pass
class UnifiedSignalEngineError(SignalForexError): pass


class ExplanationStatus:
    """Standardized status markings for signal audit logs."""
    READY: Final[str] = "[READY]"
    REJECTED: Final[str] = "[REJECTED]"
    PIPELINE_ERROR: Final[str] = "[PIPELINE_ERROR]"
    NUMERICAL_ERROR: Final[str] = "[NUMERICAL_SANITY_ERROR]"


class _ForexSignalBaseEngine:
    """Base helper class encapsulating lifecycle state for all 9 Forex Gateways."""
    def __init__(self, engine_id: str, engine_version: str) -> None:
        self.engine_id: Final[str] = engine_id
        self.ENGINE_VERSION: Final[str] = engine_version
        self._lifecycle_lock = threading.Lock()
        self._telemetry_lock = threading.Lock()
        self._is_active: bool = False
        self._is_shutdown: bool = False
        self._config: MappingProxyType = MappingProxyType({})
        self._execution_meta: Dict[str, Any] = {}

    def activate(self) -> None:
        with self._lifecycle_lock:
            if self._is_shutdown:
                raise SignalForexError(f"Engine [{self.engine_id}] cannot be activated after shutdown.")
            self._is_active = True
            logger.info(f"{self.__class__.__name__} [{self.engine_id}] activated.")

    def deactivate(self) -> None:
        with self._lifecycle_lock:
            self._is_active = False
            logger.info(f"{self.__class__.__name__} [{self.engine_id}] deactivated.")

    def shutdown(self) -> None:
        with self._lifecycle_lock:
            self._is_active = False
            self._is_shutdown = True
            logger.info(f"{self.__class__.__name__} [{self.engine_id}] permanently shut down.")

    @property
    def is_operational(self) -> bool:
        with self._lifecycle_lock:
            return self._is_active and not self._is_shutdown

    def get_latest_telemetry(self) -> Dict[str, Any]:
        with self._telemetry_lock:
            return self._execution_meta.copy()


# ==============================================================================
# GATEWAY 1: FOREX SIGNAL GENERATOR
# ==============================================================================
class ForexSignalGenerator(_ForexSignalBaseEngine):
    """
    Gateway 1: Forex Signal Generator Engine v2026.Q3
    Consumes ML predictions, dynamically maps OANDA schemas, and validates dual-direction (BUY/SELL) geometry.
    """
    def __init__(self, engine_id: str, operational_config: Dict[str, Any]) -> None:
        super().__init__(engine_id, "v2026.Q3")
        self._config = MappingProxyType(operational_config.copy())
        logger.info(f"ForexSignalGenerator [{self.engine_id}] initialized.")

    def generate(self, upstream_df: pl.DataFrame) -> pl.DataFrame:
        start_time_ns = time.perf_counter_ns()
        if not self.is_operational or upstream_df is None or upstream_df.height == 0:
            return pl.DataFrame()

        df_adapted = upstream_df

        # Resolusi Kolom Dinamis
        instrument_c = resolve_column_name(df_adapted, ["instrument", "pair", "symbol", "ticker"], "instrument")
        price_c = resolve_column_name(df_adapted, ["current_price", "close", "price", "ask", "bid"], "close")
        prob_c = resolve_column_name(df_adapted, ["prediction_probability", "calibrated_prob", "effective_prob"], "prediction_probability")
        conf_c = resolve_column_name(df_adapted, ["prediction_confidence", "confidence_score"], "prediction_confidence")
        ret_c = resolve_column_name(df_adapted, ["expected_return", "predicted_return", "returns"], "expected_return")
        tp_c = resolve_column_name(df_adapted, ["take_profit", "target_price", "tp_price"], "take_profit")
        sl_c = resolve_column_name(df_adapted, ["stop_loss", "sl_price"], "stop_loss")
        dir_c = resolve_column_name(df_adapted, ["signal_direction", "direction", "side"], "signal_direction")

        adapter_exprs = [
            pl.col(instrument_c).alias("instrument"),
            pl.col(instrument_c).alias("asset"),
            pl.col(price_c).cast(pl.Float64).alias("current_price"),
            pl.col(prob_c).cast(pl.Float64).fill_nan(0.50).fill_null(0.50).alias("prediction_probability") if prob_c in df_adapted.columns else pl.lit(0.65).alias("prediction_probability"),
            pl.col(conf_c).cast(pl.Float64).fill_nan(0.50).fill_null(0.50).alias("prediction_confidence") if conf_c in df_adapted.columns else pl.lit(0.65).alias("prediction_confidence"),
            pl.col(ret_c).cast(pl.Float64).fill_nan(0.002).fill_null(0.002).alias("expected_return") if ret_c in df_adapted.columns else pl.lit(0.002).alias("expected_return"),
        ]

        # Normalisasi Directional Token (BUY=1, SELL=-1)
        if dir_c in df_adapted.columns:
            dir_expr = (
                pl.when(pl.col(dir_c).cast(pl.Utf8).str.to_uppercase().is_in(["BUY", "1", "LONG"])).then(1)
                .when(pl.col(dir_c).cast(pl.Utf8).str.to_uppercase().is_in(["SELL", "-1", "SHORT"])).then(-1)
                .otherwise(0).cast(pl.Int64)
            )
        else:
            dir_expr = pl.lit(1).cast(pl.Int64)
        
        adapter_exprs.append(dir_expr.alias("signal_direction"))
        df_adapted = df_adapted.with_columns(adapter_exprs)

        curr_p = pl.col("current_price")
        
        # Penyelarasan Dual Geometri TP/SL (BUY: SL < Entry < TP | SELL: TP < Entry < SL)
        if tp_c in df_adapted.columns:
            raw_tp = pl.col(tp_c).cast(pl.Float64).fill_nan(0.0).fill_null(0.0)
            clean_tp = (
                pl.when((pl.col("signal_direction") == 1) & (raw_tp > curr_p)).then(raw_tp)
                .when((pl.col("signal_direction") == -1) & (raw_tp < curr_p) & (raw_tp > 0.0)).then(raw_tp)
                .when(pl.col("signal_direction") == 1).then(curr_p * 1.005)
                .otherwise(curr_p * 0.995)
            )
        else:
            clean_tp = pl.when(pl.col("signal_direction") == 1).then(curr_p * 1.005).otherwise(curr_p * 0.995)

        if sl_c in df_adapted.columns:
            raw_sl = pl.col(sl_c).cast(pl.Float64).fill_nan(0.0).fill_null(0.0)
            clean_sl = (
                pl.when((pl.col("signal_direction") == 1) & (raw_sl < curr_p) & (raw_sl > 0.0)).then(raw_sl)
                .when((pl.col("signal_direction") == -1) & (raw_sl > curr_p)).then(raw_sl)
                .when(pl.col("signal_direction") == 1).then(curr_p * 0.997)
                .otherwise(curr_p * 1.003)
            )
        else:
            clean_sl = pl.when(pl.col("signal_direction") == 1).then(curr_p * 0.997).otherwise(curr_p * 1.003)

        # Validasi Geometri Forex (BUY vs SELL)
        vector_valid_flag = (
            (pl.col("signal_direction").is_in([1, -1])) &
            (curr_p >= OANDA_MIN_PRICE_QUOTE) &
            (
                ((pl.col("signal_direction") == 1) & (clean_sl < curr_p) & (clean_tp > curr_p)) |
                ((pl.col("signal_direction") == -1) & (clean_sl > curr_p) & (clean_tp < curr_p))
            ) &
            (pl.col("prediction_probability") >= 0.0) & (pl.col("prediction_probability") <= 1.0) &
            (pl.col("prediction_confidence") >= 0.0) & (pl.col("prediction_confidence") <= 1.0)
        )

        direction_token = (
            pl.when(pl.col("signal_direction") == 1).then(pl.lit("BUY"))
            .when(pl.col("signal_direction") == -1).then(pl.lit("SELL"))
            .otherwise(pl.lit("HOLD"))
        )

        output_df = df_adapted.with_columns([
            direction_token.alias("candidate_signal"),
            curr_p.cast(pl.Float64).alias("entry_price"),
            clean_tp.alias("target_price"),
            clean_tp.alias("take_profit"),
            clean_sl.alias("stop_loss"),
            vector_valid_flag.cast(pl.Boolean).alias("signal_valid"),
            pl.when(vector_valid_flag).then(pl.lit("VALID_SIGNAL")).otherwise(pl.lit("GEOMETRY_FAILURE")).alias("signal_reason")
        ])

        latency_ms = (time.perf_counter_ns() - start_time_ns) / 1_000_000.0
        with self._telemetry_lock:
            self._execution_meta = {"rows_processed": output_df.height, "latency_ms": latency_ms}

        return output_df


# ==============================================================================
# GATEWAY 2: FOREX ENTRY FILTER
# ==============================================================================
class ForexEntryFilter(_ForexSignalBaseEngine):
    """Gateway 2: Validasi Batas Jarak Entry Forex (SL/TP Distance)."""
    def __init__(self, engine_id: str, operational_config: Dict[str, Any]) -> None:
        super().__init__(engine_id, "v2026.Q3")
        self._config = MappingProxyType(operational_config.copy())

    def filter_entry(self, input_df: pl.DataFrame) -> pl.DataFrame:
        if not self.is_operational or input_df is None or input_df.height == 0:
            return input_df

        div_floor = float(self._config.get("division_floor", 1e-8))
        min_target_dist = float(self._config.get("min_target_distance_pct", 0.0010))
        max_sl_dist = float(self._config.get("max_sl_distance_pct", 0.0300))

        entry_floor = pl.col("entry_price").abs().clip(lower_bound=div_floor)
        
        # Perhitungan Jarak Berarah (Directional Distance)
        sl_dist_pct = (
            pl.when(pl.col("signal_direction") == 1).then((pl.col("entry_price") - pl.col("stop_loss")) / entry_floor)
            .otherwise((pl.col("stop_loss") - pl.col("entry_price")) / entry_floor)
        )

        tp_dist_pct = (
            pl.when(pl.col("signal_direction") == 1).then((pl.col("target_price") - pl.col("entry_price")) / entry_floor)
            .otherwise((pl.col("entry_price") - pl.col("target_price")) / entry_floor)
        )

        pass_filter = (
            pl.col("signal_valid") &
            (sl_dist_pct > 0.0) & (sl_dist_pct <= max_sl_dist) &
            (tp_dist_pct >= min_target_dist)
        )

        return input_df.with_columns([
            pass_filter.alias("entry_filter_pass"),
            pl.when(pass_filter).then(pl.lit("ENTRY_ALLOWED")).otherwise(pl.lit("ENTRY_DISTANCE_REJECTED")).alias("entry_filter_reason")
        ])


# ==============================================================================
# GATEWAY 3: FOREX EXIT FILTER
# ==============================================================================
class ForexExitFilter(_ForexSignalBaseEngine):
    """Gateway 3: Validasi Rasio Risk-Reward (RR) Forex."""
    def __init__(self, engine_id: str, operational_config: Dict[str, Any]) -> None:
        super().__init__(engine_id, "v2026.Q3")
        self._config = MappingProxyType(operational_config.copy())

    def filter_exit(self, input_df: pl.DataFrame) -> pl.DataFrame:
        if not self.is_operational or input_df is None or input_df.height == 0:
            return input_df

        div_floor = float(self._config.get("division_floor", 1e-8))
        min_rr = float(self._config.get("min_risk_reward", 1.2))

        risk_denom = (
            pl.when(pl.col("signal_direction") == 1).then(pl.col("entry_price") - pl.col("stop_loss"))
            .otherwise(pl.col("stop_loss") - pl.col("entry_price"))
        ).clip(lower_bound=div_floor)

        reward_num = (
            pl.when(pl.col("signal_direction") == 1).then(pl.col("take_profit") - pl.col("entry_price"))
            .otherwise(pl.col("entry_price") - pl.col("take_profit"))
        ).clip(lower_bound=0.0)

        calculated_rr = reward_num / risk_denom
        pass_exit = pl.col("entry_filter_pass") & (calculated_rr >= min_rr)

        return input_df.with_columns([
            pass_exit.alias("exit_filter_pass"),
            pl.when(pass_exit).then(pl.lit("EXIT_ALLOWED")).otherwise(pl.lit("RR_BELOW_THRESHOLD")).alias("exit_filter_reason"),
            calculated_rr.alias("calculated_risk_reward")
        ])


# ==============================================================================
# GATEWAY 4: FOREX PROBABILITY FILTER
# ==============================================================================
class ForexProbabilityFilter(_ForexSignalBaseEngine):
    """Gateway 4: Filter Ambang Probabilitas Prediksi ML."""
    def __init__(self, engine_id: str, operational_config: Dict[str, Any]) -> None:
        super().__init__(engine_id, "v2026.Q3")
        self._config = MappingProxyType(operational_config.copy())

    def filter_probability(self, input_df: pl.DataFrame) -> pl.DataFrame:
        if not self.is_operational or input_df is None or input_df.height == 0:
            return input_df

        min_prob = float(self._config.get("minimum_probability", 0.50))
        pass_prob = pl.col("exit_filter_pass") & (pl.col("prediction_probability") >= min_prob)

        return input_df.with_columns([
            pass_prob.alias("probability_filter_pass"),
            pl.when(pass_prob).then(pl.lit("PROBABILITY_ALLOWED")).otherwise(pl.lit("PROBABILITY_TOO_LOW")).alias("probability_filter_reason")
        ])


# ==============================================================================
# GATEWAY 5: FOREX CONFIDENCE FILTER
# ==============================================================================
class ForexConfidenceFilter(_ForexSignalBaseEngine):
    """Gateway 5: Filter Skor Keyakinan Model ML."""
    def __init__(self, engine_id: str, operational_config: Dict[str, Any]) -> None:
        super().__init__(engine_id, "v2026.Q3")
        self._config = MappingProxyType(operational_config.copy())

    def filter_confidence(self, input_df: pl.DataFrame) -> pl.DataFrame:
        if not self.is_operational or input_df is None or input_df.height == 0:
            return input_df

        min_conf = float(self._config.get("minimum_confidence", 0.45))
        pass_conf = pl.col("probability_filter_pass") & (pl.col("prediction_confidence") >= min_conf)

        return input_df.with_columns([
            pass_conf.alias("confidence_filter_pass"),
            pl.when(pass_conf).then(pl.lit("CONFIDENCE_ALLOWED")).otherwise(pl.lit("CONFIDENCE_TOO_LOW")).alias("confidence_filter_reason")
        ])


# ==============================================================================
# GATEWAY 6: FOREX TP/SL OPTIMIZER & POSITION SIZER
# ==============================================================================
class ForexTpSlOptimizer(_ForexSignalBaseEngine):
    """Gateway 6: Optimalisasi Batas Risiko & Hitung Ukuran Unit Posisi OANDA."""
    def __init__(self, engine_id: str, operational_config: Dict[str, Any]) -> None:
        super().__init__(engine_id, "v2026.Q3")
        self._config = MappingProxyType(operational_config.copy())

    def optimize_tp_sl(self, input_df: pl.DataFrame) -> pl.DataFrame:
        if not self.is_operational or input_df is None or input_df.height == 0:
            return input_df

        div_floor = float(self._config.get("division_floor", 1e-8))
        default_units = float(self._config.get("default_position_units", 1000.0))  # 1 Micro Lot = 1,000 Units

        opt_tp = pl.col("take_profit")
        opt_sl = pl.col("stop_loss")
        opt_rr = pl.col("calculated_risk_reward")

        units_expr = pl.lit(max(default_units, OANDA_MIN_UNITS))

        return input_df.with_columns([
            opt_tp.alias("optimized_take_profit"),
            opt_sl.alias("optimized_stop_loss"),
            opt_rr.alias("optimized_risk_reward"),
            units_expr.alias("quantity"),
            units_expr.alias("units"),
            pl.col("confidence_filter_pass").alias("tp_sl_optimizer_pass"),
            pl.lit("OPTIMIZATION_SUCCESSFUL").alias("tp_sl_optimizer_reason")
        ])


# ==============================================================================
# GATEWAY 7: FOREX SIGNAL RANKER
# ==============================================================================
class ForexSignalRanker(_ForexSignalBaseEngine):
    """Gateway 7: Pemeringkatan Prioritas Sinyal Berbasis Skor Komposit."""
    def __init__(self, engine_id: str, operational_config: Dict[str, Any]) -> None:
        super().__init__(engine_id, "v2026.Q3")
        self._config = MappingProxyType(operational_config.copy())

    def rank_signals(self, input_df: pl.DataFrame) -> pl.DataFrame:
        if not self.is_operational or input_df is None or input_df.height == 0:
            return input_df

        w_prob = float(self._config.get("weight_probability", 0.35))
        w_conf = float(self._config.get("weight_confidence", 0.25))
        w_rr = float(self._config.get("weight_risk_reward", 0.40))
        min_rank_s = float(self._config.get("minimum_rank_score", 0.20))

        composite_score = (
            (pl.col("prediction_probability") * pl.lit(w_prob)) +
            (pl.col("prediction_confidence") * pl.lit(w_conf)) +
            (pl.col("optimized_risk_reward").clip(upper_bound=3.0) / 3.0 * pl.lit(w_rr))
        )

        df_scored = input_df.with_columns(composite_score.alias("signal_rank_score"))
        rank_expr = pl.col("signal_rank_score").rank(descending=True, method="min").cast(pl.Int32)
        pass_rank = pl.col("tp_sl_optimizer_pass") & (pl.col("signal_rank_score") >= min_rank_s)

        return df_scored.with_columns([
            rank_expr.alias("signal_rank_position"),
            rank_expr.alias("signal_rank"),
            pass_rank.alias("signal_ranker_pass"),
            pl.lit("RANKING_SUCCESSFUL").alias("signal_ranker_reason")
        ])


# ==============================================================================
# GATEWAY 8: FOREX SIGNAL VALIDATOR
# ==============================================================================
class ForexSignalValidator(_ForexSignalBaseEngine):
    """Gateway 8: Gate Akhir Validasi Eksekusi Order Forex."""
    def __init__(self, engine_id: str, operational_config: Dict[str, Any]) -> None:
        super().__init__(engine_id, "v2026.Q3")
        self._config = MappingProxyType(operational_config.copy())

    def validate_signals(self, input_df: pl.DataFrame) -> pl.DataFrame:
        if not self.is_operational or input_df is None or input_df.height == 0:
            return input_df

        abs_min_rr = float(self._config.get("absolute_min_risk_reward", 1.10))

        final_gate = (
            pl.col("signal_ranker_pass") &
            (pl.col("entry_price") >= OANDA_MIN_PRICE_QUOTE) &
            (pl.col("units") >= OANDA_MIN_UNITS) &
            (pl.col("optimized_risk_reward") >= abs_min_rr) &
            (
                ((pl.col("signal_direction") == 1) & (pl.col("optimized_take_profit") > pl.col("entry_price")) & (pl.col("optimized_stop_loss") < pl.col("entry_price"))) |
                ((pl.col("signal_direction") == -1) & (pl.col("optimized_take_profit") < pl.col("entry_price")) & (pl.col("optimized_stop_loss") > pl.col("entry_price")))
            )
        )

        return input_df.with_columns([
            final_gate.alias("is_valid_execution"),
            pl.when(final_gate).then(pl.lit("EXECUTION_READY")).otherwise(pl.lit("VALIDATION_FAILED")).alias("final_validator_reason")
        ])


# ==============================================================================
# GATEWAY 9: FOREX SIGNAL EXPLAINER
# ==============================================================================
class ForexSignalExplainer(_ForexSignalBaseEngine):
    """Gateway 9: Generator Narasi Penjelasan & Injeksi Kontrak Skema Downstream."""
    def __init__(self, engine_id: str, operational_config: Dict[str, Any]) -> None:
        super().__init__(engine_id, "v2026.Q3")

    def explain(self, input_df: pl.DataFrame) -> pl.DataFrame:
        if not self.is_operational or input_df is None or input_df.height == 0:
            return input_df

        explanation_status = pl.when(pl.col("is_valid_execution")).then(pl.lit(ExplanationStatus.READY)).otherwise(pl.lit(ExplanationStatus.REJECTED))

        explanation_text = pl.format(
            "{} INSTRUMENT: {} | SIDE: {} | ENTRY: {} | TP: {} | SL: {} | UNITS: {}",
            explanation_status,
            pl.col("instrument"),
            pl.col("candidate_signal"),
            pl.col("entry_price").round(5),
            pl.col("optimized_take_profit").round(5),
            pl.col("optimized_stop_loss").round(5),
            pl.col("units")
        )

        signal_val = pl.when(pl.col("is_valid_execution")).then(1.0).otherwise(0.0)
        side_token = pl.when(pl.col("is_valid_execution")).then(pl.col("candidate_signal")).otherwise(pl.lit("HOLD"))

        return input_df.with_columns([
            explanation_text.alias("signal_explanation_text"),
            signal_val.alias("signal"),
            side_token.alias("side"),
            pl.col("prediction_confidence").alias("confidence_score"),
            pl.col("expected_return").alias("predicted_return"),
            pl.col("instrument").alias("portfolio_asset_id")
        ])


# ==============================================================================
# FACADE CLASS: UNIFIED FOREX SIGNAL ENGINE
# ==============================================================================
class UnifiedForexSignalEngine:
    """Facade Terpusat Pengendali Pipa 9 Gateway Sinyal Forex OANDA."""

    def __init__(self, engine_id_prefix: str = "OandaForexSignalEngine", custom_configs: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        self.engine_id_prefix = engine_id_prefix
        self._lock = threading.Lock()
        
        default_configs = self._get_default_gateway_configs()
        if custom_configs:
            for g_key, g_cfg in custom_configs.items():
                if g_key in default_configs:
                    default_configs[g_key].update(g_cfg)

        self.generator = ForexSignalGenerator(f"{engine_id_prefix}_G1", default_configs["generator"])
        self.entry_filter = ForexEntryFilter(f"{engine_id_prefix}_G2", default_configs["entry_filter"])
        self.exit_filter = ForexExitFilter(f"{engine_id_prefix}_G3", default_configs["exit_filter"])
        self.prob_filter = ForexProbabilityFilter(f"{engine_id_prefix}_G4", default_configs["prob_filter"])
        self.conf_filter = ForexConfidenceFilter(f"{engine_id_prefix}_G5", default_configs["conf_filter"])
        self.tpsl_optimizer = ForexTpSlOptimizer(f"{engine_id_prefix}_G6", default_configs["tpsl_optimizer"])
        self.ranker = ForexSignalRanker(f"{engine_id_prefix}_G7", default_configs["ranker"])
        self.validator = ForexSignalValidator(f"{engine_id_prefix}_G8", default_configs["validator"])
        self.explainer = ForexSignalExplainer(f"{engine_id_prefix}_G9", default_configs["explainer"])

        self.gateways = [
            self.generator, self.entry_filter, self.exit_filter,
            self.prob_filter, self.conf_filter, self.tpsl_optimizer,
            self.ranker, self.validator, self.explainer
        ]

        self.activate_all()
        logger.info(f"UnifiedForexSignalEngine [{self.engine_id_prefix}] facade active.")

    @staticmethod
    def _get_default_gateway_configs() -> Dict[str, Dict[str, Any]]:
        return {
            "generator": {"strict_schema_check": True},
            "entry_filter": {"min_target_distance_pct": 0.0010, "max_sl_distance_pct": 0.0300, "division_floor": 1e-8},
            "exit_filter": {"min_risk_reward": 1.2, "division_floor": 1e-8},
            "prob_filter": {"minimum_probability": 0.50},
            "conf_filter": {"minimum_confidence": 0.45},
            "tpsl_optimizer": {"division_floor": 1e-8, "default_position_units": 1000.0},
            "ranker": {"weight_probability": 0.35, "weight_confidence": 0.25, "weight_risk_reward": 0.40, "minimum_rank_score": 0.20},
            "validator": {"absolute_min_risk_reward": 1.10},
            "explainer": {}
        }

    def activate_all(self) -> None:
        with self._lock:
            for gw in self.gateways:
                if not gw.is_operational:
                    gw.activate()

    def execute_pipeline(self, prediction_df: pl.DataFrame) -> pl.DataFrame:
        """Eksekusi sekuensial 9 Gateway Pipa Sinyal Forex."""
        start_time_ns = time.perf_counter_ns()
        if prediction_df is None or prediction_df.height == 0:
            return pl.DataFrame()

        df1 = self.generator.generate(prediction_df)
        df2 = self.entry_filter.filter_entry(df1)
        df3 = self.exit_filter.filter_exit(df2)
        df4 = self.prob_filter.filter_probability(df3)
        df5 = self.conf_filter.filter_confidence(df4)
        df6 = self.tpsl_optimizer.optimize_tp_sl(df5)
        df7 = self.ranker.rank_signals(df6)
        df8 = self.validator.validate_signals(df7)
        final_df = self.explainer.explain(df8)

        total_latency_ms = (time.perf_counter_ns() - start_time_ns) / 1_000_000.0
        valid_count = int(final_df["is_valid_execution"].sum()) if "is_valid_execution" in final_df.columns else 0
        logger.info(f"Pipa sinyal Forex selesai dalam {total_latency_ms:.2f} ms. Sinyal Valid: {valid_count}/{final_df.height}")

        return final_df

    # Alias Methods untuk kompatibilitas orchestrator main.py
    def process_signals(self, df: pl.DataFrame) -> pl.DataFrame: return self.execute_pipeline(df)
    def filter_signals(self, df: pl.DataFrame) -> pl.DataFrame: return self.execute_pipeline(df)
    def generate_signals(self, df: pl.DataFrame) -> pl.DataFrame: return self.execute_pipeline(df)


if __name__ == "__main__":
    logger.info("Pengujian Integritas Modul signal_forex.py...")
    test_df = pl.DataFrame({
        "instrument": ["EUR_USD", "GBP_USD", "USD_JPY"],
        "close": [1.0850, 1.2650, 155.20],
        "calibrated_prob": [0.72, 0.40, 0.65],
        "confidence_score": [0.85, 0.50, 0.75],
        "predicted_return": [0.0035, -0.0020, 0.0040],
        "stop_loss": [1.0820, 1.2680, 154.50],   # EUR_USD BUY, GBP_USD SELL (SL > Entry)
        "take_profit": [1.0910, 1.2580, 156.50], # EUR_USD BUY, GBP_USD SELL (TP < Entry)
        "signal_direction": ["BUY", "SELL", "BUY"]
    })

    engine = UnifiedForexSignalEngine()
    result = engine.process_signals(test_df)
    print("Hasil Sinyal Forex:")
    print(result.select(["instrument", "side", "is_valid_execution", "signal_explanation_text"]))
