"""
================================================================================
MODULE: risk/risk_engine.py
DESCRIPTION: Strict Deterministic Quantitative Risk Engine for OANDA Forex.
VERSION: 2026.1.0 (OANDA Forex Risk Safeguard & Circuit Breaker Edition)
PYTHON VERSION: 3.11+ / 3.12+
ARCHITECTURE: Clean Architecture, Deterministic Gateways & Stratified Aggregator
================================================================================
"""

import time
import math
import logging
import hashlib
import json
import datetime
import numpy as np
import polars as pl
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

from config import Config

logger = logging.getLogger("Forex.RiskEngine")


# =============================================================================
# DATA STRUCTURES
# =============================================================================
@dataclass(frozen=True)
class ForexRiskOutput:
    composite_risk_score: float
    final_halt: bool
    final_warning: bool
    highest_severity: str
    hard_halt_sources: List[str]
    soft_halt_sources: List[str]
    telemetry: Dict[str, Any]


# =============================================================================
# DUAL-MODE DRAWDOWN GUARD
# =============================================================================
class ForexDrawdownGuard:
    """Mengukur penurunan modal (Drawdown) dari Floating Equity dan Realized PnL."""
    
    def __init__(self, max_unrealized_dd: float = 0.05, max_realized_dd: float = 0.03):
        self.max_unrealized_dd = max_unrealized_dd
        self.max_realized_dd = max_realized_dd
        self.peak_equity = -np.inf

    def evaluate(self, account_nav: float, balance: float) -> Tuple[float, bool, bool]:
        if account_nav > self.peak_equity:
            self.peak_equity = account_nav
            
        if self.peak_equity <= 0:
            return 0.0, False, False

        unrealized_dd = (self.peak_equity - account_nav) / self.peak_equity
        halt = unrealized_dd >= self.max_unrealized_dd
        warning = (unrealized_dd >= self.max_unrealized_dd * 0.6) and not halt
        
        return float(unrealized_dd), halt, warning


# =============================================================================
# SPREAD & LIQUIDITY GUARD
# =============================================================================
class ForexLiquidityGuard:
    """Memeriksa kelayakan spread dan likuiditas sebelum order dieksekusi."""

    def __init__(self, max_spread_pips: float = 3.0):
        self.max_spread_pips = max_spread_pips

    def evaluate(self, spread_pips: float) -> Tuple[float, bool]:
        halt = spread_pips > self.max_spread_pips
        score = min(spread_pips / self.max_spread_pips, 1.0)
        return float(score), halt


# =============================================================================
# VOLATILITY GUARD (ATR & PARKINSON ESTIMATOR)
# =============================================================================
class ForexVolatilityGuard:
    """Mendeteksi kondisi volatilitas ekstrem pada pasar Forex."""

    def __init__(self, window_size: int = 20, max_volatility_threshold: float = 0.03):
        self.window_size = window_size
        self.max_volatility_threshold = max_volatility_threshold

    def evaluate(self, df_klines: pl.DataFrame) -> Tuple[float, bool]:
        if df_klines.height < self.window_size or "close" not in df_klines.columns:
            return 0.0, False

        # Parkinson Volatility Calculation
        if "high" in df_klines.columns and "low" in df_klines.columns:
            pv = ((df_klines["high"] / df_klines["low"]).log() ** 2) / (4.0 * math.log(2.0))
            vol_raw = float(pv.tail(self.window_size).mean() or 0.0)
            vol_ann = math.sqrt(max(vol_raw, 1e-8)) * math.sqrt(252)
        else:
            returns = df_klines["close"].log().diff().drop_nulls()
            vol_ann = float(returns.tail(self.window_size).std(ddof=1) or 0.0) * math.sqrt(252)

        vol_score = min(vol_ann / self.max_volatility_threshold, 1.0)
        halt = vol_score >= 1.0
        return float(vol_score), halt


# =============================================================================
# CORE RISK ENGINE FACADE
# =============================================================================
class RiskEngine:
    """
    Penentu keputusan akhir yang Deterministic.
    ML tidak diperbolehkan menembus batasan dari RiskEngine ini.
    """

    def __init__(self, max_risk_per_trade: float = 0.01):
        self.config = Config.load()
        self.drawdown_guard = ForexDrawdownGuard(
            max_unrealized_dd=self.config.MAX_ACCOUNT_DRAWDOWN_PCT if hasattr(self.config, 'MAX_ACCOUNT_DRAWDOWN_PCT') else 0.05
        )
        self.liquidity_guard = ForexLiquidityGuard(max_spread_pips=self.config.MAX_SPREAD_PIPS)
        self.volatility_guard = ForexVolatilityGuard(window_size=self.config.ATR_PERIOD)

    def calculate_stop_levels(self, df_features: pl.DataFrame, raw_signal: Any) -> Tuple[float, float]:
        """
        Menghitung Stop Loss (SL) dan Take Profit (TP) adaptif berbasis volatilitas ATR.
        Formula SL = Entry ± (ATR * ATR_MULTIPLIER)
        Formula TP = Entry ± (SL_Distance * MIN_RISK_REWARD_RATIO)
        """
        last_close = float(df_features["close"].iloc[-1] if hasattr(df_features, 'iloc') else df_features["close"][-1])
        atr_val = float(df_features["atr_14"].iloc[-1] if "atr_14" in df_features.columns and hasattr(df_features, 'iloc') else df_features.get("atr_14", [last_close * 0.001])[-1])

        if atr_val <= 0 or math.isnan(atr_val):
            atr_val = last_close * 0.0015  # Fallback default 15 pips

        sl_distance = atr_val * self.config.ATR_STOP_LOSS_MULTIPLIER
        tp_distance = sl_distance * self.config.MIN_RISK_REWARD_RATIO

        direction = str(getattr(raw_signal, "direction", "BUY")).upper()

        if direction in ["BUY", "LONG", "1"]:
            sl_price = last_close - sl_distance
            tp_price = last_close + tp_distance
        else:
            sl_price = last_close + sl_distance
            tp_price = last_close - tp_distance

        return round(sl_price, 5), round(tp_price, 5)

    def validate_trade(self, instrument: str, units: int, current_portfolio: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Pemeriksaan deterministik akhir sebelum order dikirim ke OANDA API.
        """
        if units == 0:
            return False, "INVALID_UNITS_ZERO"

        # 1. Posisi Maksimum Terbuka
        open_pos_count = len(current_portfolio.get("open_positions", [])) if isinstance(current_portfolio, dict) else 0
        if open_pos_count >= self.config.MAX_OPEN_POSITIONS:
            return False, f"MAX_OPEN_POSITIONS_REACHED ({open_pos_count}/{self.config.MAX_OPEN_POSITIONS})"

        return True, "PASSED"
