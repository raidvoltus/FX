"""
================================================================================
MODULE: data/feature_engine.py
DESCRIPTION: Institutional Quantitative Feature Engineering Engine for OANDA Forex.
VERSION: 2026.1.0 (Forex Multi-Factor & Schema-Guarded Selector Edition)
PYTHON VERSION: 3.11+ / 3.12+
ARCHITECTURE: Clean Architecture, Decoupled Factor Ingestion & Feature Selection
================================================================================
"""

import math
import logging
import polars as pl
from datetime import datetime
from typing import Set, Any, Optional, List, Final

from config import Config

logger = logging.getLogger("Forex.Features")

# Kolom Metadata & Base OHLCV yang dilindungi agar tidak terhapus saat filtering variansi
EXCLUDED_METADATA_COLS: Final[Set[str]] = {
    "ticker", "symbol", "asset", "pair", "date", "datetime", "timestamp", 
    "interval", "data_source", "open", "high", "low", "close", "volume", 
    "returns", "returns_clean"
}


# =============================================================================
# 1. FACTOR EXTRACTORS
# =============================================================================
class TechnicalTrendFactors:
    """Mengekstrak indikator tren moving average, momentum, dan Bollinger Bands."""
    
    def build_all_features(self, df: pl.DataFrame) -> pl.DataFrame:
        if df.height == 0:
            return df
        
        group_col = "ticker" if "ticker" in df.columns else None
        
        exprs = [
            # Moving Averages (SMA 9, 20, 50, 200)
            pl.col("close").rolling_mean(window_size=9).over(group_col).fill_null(strategy="backward").alias("sma_9") if group_col else pl.col("close").rolling_mean(window_size=9).fill_null(strategy="backward").alias("sma_9"),
            pl.col("close").rolling_mean(window_size=20).over(group_col).fill_null(strategy="backward").alias("sma_20") if group_col else pl.col("close").rolling_mean(window_size=20).fill_null(strategy="backward").alias("sma_20"),
            pl.col("close").rolling_mean(window_size=50).over(group_col).fill_null(strategy="backward").alias("sma_50") if group_col else pl.col("close").rolling_mean(window_size=50).fill_null(strategy="backward").alias("sma_50"),
            
            # Bollinger Bands (Period 20, Dev 2)
            pl.col("close").rolling_std(window_size=20).over(group_col).fill_null(0.0).alias("bb_std_20") if group_col else pl.col("close").rolling_std(window_size=20).fill_null(0.0).alias("bb_std_20"),
            
            # Momentum Bar Returns
            (((pl.col("close") - pl.col("close").shift(1).over(group_col)) / pl.col("close").shift(1).over(group_col)) if group_col else ((pl.col("close") - pl.col("close").shift(1)) / pl.col("close").shift(1))).fill_null(0.0).alias("momentum_1m"),
            (((pl.col("close") - pl.col("close").shift(3).over(group_col)) / pl.col("close").shift(3).over(group_col)) if group_col else ((pl.col("close") - pl.col("close").shift(3)) / pl.col("close").shift(3))).fill_null(0.0).alias("momentum_3m"),
            (((pl.col("close") - pl.col("close").shift(5).over(group_col)) / pl.col("close").shift(5).over(group_col)) if group_col else ((pl.col("close") - pl.col("close").shift(5)) / pl.col("close").shift(5))).fill_null(0.0).alias("momentum_5m"),
        ]
        
        res = df.with_columns(exprs)

        # Turunan Bollinger Bands Upper & Lower
        res = res.with_columns([
            (pl.col("sma_20") + (2.0 * pl.col("bb_std_20"))).alias("bb_upper"),
            (pl.col("sma_20") - (2.0 * pl.col("bb_std_20"))).alias("bb_lower"),
        ])
        
        # Distance dari BB Upper & Lower (Normalized)
        res = res.with_columns([
            ((pl.col("close") - pl.col("bb_lower")) / (pl.col("bb_upper") - pl.col("bb_lower") + 1e-9)).alias("bb_z_score")
        ])

        return res


class MicrostructureVolatilityFactors:
    """Mengekstrak fitur volatilitas (ATR & Parkinson Volatility) untuk Risk Engine."""
    
    def extract_all(self, df: pl.DataFrame, atr_period: int = 14) -> pl.DataFrame:
        if df.height == 0:
            return df

        group_col = "ticker" if "ticker" in df.columns else None
        
        # Memastikan returns_clean tersedia
        working_df = df
        if "returns_clean" not in working_df.columns:
            if group_col:
                ret_expr = (pl.col("close") / pl.col("close").shift(1).over(group_col)).log().fill_null(0.0)
            else:
                ret_expr = (pl.col("close") / pl.col("close").shift(1)).log().fill_null(0.0)
            working_df = working_df.with_columns(ret_expr.alias("returns_clean"))

        # 1. True Range Calculation
        tr1 = pl.col("high") - pl.col("low")
        tr2 = (pl.col("high") - pl.col("close").shift(1).over(group_col)).abs() if group_col else (pl.col("high") - pl.col("close").shift(1)).abs()
        tr3 = (pl.col("low") - pl.col("close").shift(1).over(group_col)).abs() if group_col else (pl.col("low") - pl.col("close").shift(1)).abs()
        
        working_df = working_df.with_columns(
            pl.max_horizontal(tr1, tr2, tr3).fill_null(0.0).alias("true_range")
        )

        exprs = [
            # ATR (Average True Range)
            (pl.col("true_range").rolling_mean(window_size=atr_period).over(group_col) if group_col else pl.col("true_range").rolling_mean(window_size=atr_period)).fill_null(0.0).alias("atr_14"),
            
            # Volatilitas standar return 14 bar
            (pl.col("returns_clean").rolling_std(window_size=14).over(group_col) if group_col else pl.col("returns_clean").rolling_std(window_size=14)).fill_null(0.0).alias("volatility_14m"),
            
            # Parkinson High-Low Volatility
            ((pl.col("high") / pl.col("low")).log().pow(2) / (4.0 * math.log(2.0))).fill_null(0.0).alias("parkinson_vol_raw")
        ]

        return working_df.with_columns(exprs)


class LagFeatureExtractor:
    """Mengekstrak fitur Lag Return untuk ML Classification Model."""
    
    def extract_lags(self, df: pl.DataFrame, max_lags: int = 6) -> pl.DataFrame:
        if df.height == 0 or "returns_clean" not in df.columns:
            return df

        group_col = "ticker" if "ticker" in df.columns else None
        lag_exprs = []

        for i in range(1, max_lags + 1):
            if group_col:
                expr = pl.col("returns_clean").shift(i).over(group_col).fill_null(0.0).alias(f"lag_{i}")
            else:
                expr = pl.col("returns_clean").shift(i).fill_null(0.0).alias(f"lag_{i}")
            lag_exprs.append(expr)

        return df.with_columns(lag_exprs)


# =============================================================================
# 2. FEATURE SELECTOR WITH VARIANCE GUARD
# =============================================================================
class FeatureSelector:
    """
    Melindungi kolom metadata/teks dan menyaring fitur dengan variansi mendekati nol (< 1e-9).
    """
    def __init__(self, target_variance_threshold: float = 1e-9):
        self.target_variance_threshold = target_variance_threshold

    def fit_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        if df.height == 0:
            return df

        metadata_cols = [c for c in df.columns if c in EXCLUDED_METADATA_COLS]
        candidate_cols = [c for c in df.columns if c not in EXCLUDED_METADATA_COLS]

        valid_numeric_features = []
        
        for col in candidate_cols:
            s = df[col]

            # 1. Skip non-numeric series
            if not s.dtype.is_numeric():
                continue

            # 2. Variance evaluation
            s_clean = s.drop_nulls()
            if len(s_clean) > 1:
                try:
                    col_var = float(s_clean.var())
                    if math.isnan(col_var) or col_var < self.target_variance_threshold:
                        logger.info(f"🗑️ Filtering out low-variance feature: {col} (var={col_var:.10f})")
                        continue
                    valid_numeric_features.append(col)
                except Exception as e:
                    logger.warning(f"⚠️ Variance check bypassed for {col}: {e}")
                    valid_numeric_features.append(col)
            else:
                valid_numeric_features.append(col)

        logger.info(
            f"✔ [FEATURE_SELECTION_SUCCESS] Mempertahankan {len(metadata_cols)} metadata cols "
            f"dan {len(valid_numeric_features)} numeric features dari {len(candidate_cols)} kandidat."
        )

        return df.select(metadata_cols + valid_numeric_features)


# =============================================================================
# 3. CONSOLIDATED FACADE & INTERFACE ALIAS BINDINGS
# =============================================================================
class FeatureEngineFacade:
    """
    Facade tunggal untuk ekstraksi fitur Tren, Volatilitas, Lags, dan Selection.
    """
    def __init__(self, **kwargs: Any):
        self.technical_trend = TechnicalTrendFactors()
        self.microstructure = MicrostructureVolatilityFactors()
        self.lag_extractor = LagFeatureExtractor()
        self.selector = FeatureSelector()
        self.config = Config.load()

    def build_features(self, df: pl.DataFrame, run_selection: bool = True) -> pl.DataFrame:
        if df.height == 0:
            logger.warning("⚠️ DataFrame kosong dikirim ke FeatureEngineFacade.")
            return df

        eager_df = df.collect() if isinstance(df, pl.LazyFrame) else df

        # Dual-Schema Alignment
        if "ticker" in eager_df.columns and "asset" not in eager_df.columns:
            eager_df = eager_df.with_columns(pl.col("ticker").alias("asset"))
        elif "asset" in eager_df.columns and "ticker" not in eager_df.columns:
            eager_df = eager_df.with_columns(pl.col("asset").alias("ticker"))

        # 1. Technical & Trend Factors
        df_tech = self.technical_trend.build_all_features(eager_df)

        # 2. Volatility Factors (ATR 14)
        df_micro = self.microstructure.extract_all(df_tech, atr_period=self.config.ATR_PERIOD)

        # 3. Lag Features (6 Lags)
        df_lags = self.lag_extractor.extract_lags(df_micro, max_lags=self.config.REQUIRED_LAG_FEATURE_COUNT)

        # 4. Feature Selection
        if run_selection:
            df_final = self.selector.fit_transform(df_lags)
        else:
            df_final = df_lags

        return df_final

    def compute_full_feature_grid(self, df: pl.DataFrame, run_selection: bool = True) -> pl.DataFrame:
        """Alias method untuk kompatibilitas penuh dengan Orchestrator."""
        return self.build_features(df, run_selection=run_selection)

    def extract_features(self, df: pl.DataFrame, run_selection: bool = True) -> pl.DataFrame:
        """Alias method kedua."""
        return self.build_features(df, run_selection=run_selection)


# Explicit Interface Aliases
UnifiedFeatureEngine = FeatureEngineFacade
FeatureEngine = FeatureEngineFacade
