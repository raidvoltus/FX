"""
================================================================================
MODULE: ml/strategy_selector.py
DESCRIPTION: ML Market Regime Detection & Automated Strategy Selection Engine.
VERSION: 2026.1.0 (Forex Regime Classification Edition)
PYTHON VERSION: 3.11+ / 3.12+
ARCHITECTURE: Clean Architecture, Machine Learning Regime Selection
================================================================================
"""

import logging
import numpy as np
import polars as pl
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional

from config import Config

logger = logging.getLogger("Forex.StrategySelector")


@dataclass(frozen=True)
class StrategySelectionResult:
    selected_strategy_name: str
    confidence: float
    regime_type: str
    metrics: Dict[str, float]


class MLStrategySelector:
    """
    ML Selector yang menganalisis rezim pasar (Market Regime) secara otomatis:
    1. TRENDING HIGH VOLATILITY  -> Momentum / ML Classification
    2. TRENDING LOW VOLATILITY   -> SMA Crossover
    3. RANGING / MEAN REVERTING  -> Bollinger Bands / Contrarian
    4. NOISE / UNCERTAIN         -> HOLD (Confidence < Threshold)
    """

    def __init__(self, model_path: Optional[str] = None):
        self.config = Config.load()
        self.min_confidence = self.config.MIN_ML_CONFIDENCE

    def evaluate(self, df_features: pl.DataFrame) -> Tuple[str, float]:
        """
        Mengevaluasi DataFrame fitur teknis terkini dan mengembalikan:
        (selected_strategy_name, confidence_score)
        """
        if df_features.height == 0:
            return "hold", 0.0

        res = self.evaluate_regime(df_features)
        return res.selected_strategy_name, res.confidence

    def evaluate_regime(self, df_features: pl.DataFrame) -> StrategySelectionResult:
        """
        Menghitung indikator rezim pasar berbasis Volatilitas & Trend Strength.
        """
        # Parameter Terakhir (Bar Terkini)
        tail = df_features.tail(1)
        
        volatility = float(tail.select(pl.col("volatility_14m")).item() if "volatility_14m" in tail.columns else 0.001)
        bb_z_score = float(tail.select(pl.col("bb_z_score")).item() if "bb_z_score" in tail.columns else 0.5)
        momentum_5m = float(tail.select(pl.col("momentum_5m")).item() if "momentum_5m" in tail.columns else 0.0)

        # 1. Klasifikasi Volatilitas & Tren
        is_high_volatility = volatility > 0.0020
        is_strong_trend = abs(momentum_5m) > 0.0015
        is_mean_reverting = bb_z_score > 0.85 or bb_z_score < 0.15

        # 2. Logika Penentuan Strategi & Confidence
        if is_strong_trend and is_high_volatility:
            regime = "TRENDING_HIGH_VOL"
            strategy = "ml_classification"
            confidence = min(0.65 + abs(momentum_5m) * 100, 0.95)

        elif is_strong_trend and not is_high_volatility:
            regime = "TRENDING_LOW_VOL"
            strategy = "sma"
            confidence = 0.75

        elif is_mean_reverting:
            regime = "RANGING_MEAN_REVERSION"
            strategy = "bollinger_bands" if is_high_volatility else "contrarian"
            confidence = 0.70

        elif abs(momentum_5m) > 0.0005:
            regime = "MOMENTUM_BUILDUP"
            strategy = "momentum"
            confidence = 0.68

        else:
            regime = "SIDEWAYS_NOISE"
            strategy = "hold"
            confidence = 0.40

        logger.info(
            f"🧠 [ML REGIME DETECTED] Regime: {regime} -> Selected Strategy: '{strategy}' "
            f"(Confidence: {confidence:.2f})"
        )

        return StrategySelectionResult(
            selected_strategy_name=strategy,
            confidence=float(confidence),
            regime_type=regime,
            metrics={
                "volatility": volatility,
                "bb_z_score": bb_z_score,
                "momentum_5m": momentum_5m
            }
        )
