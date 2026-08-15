"""
=============================================================================
Module      : storage.py (v2026.Q3 OANDA Forex Synchronized Edition)
Description : Institutional Unified Storage Engine for OANDA Forex Bot.
Directory   : Flat Directory (Root Level with main.py)
Compliance  : OANDA Forex Rules (Micro/Standard Units, Isolated Mode Storage)
=============================================================================
"""

import os
import json
import sqlite3
import time
import hashlib
import threading
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Dict, Any, List, Tuple, Optional, Union

import polars as pl


# ==============================================================================
# HELPER PARSER TIMESTAMP SAFE
# ==============================================================================
def _safe_parse_timestamp_ns(val: Any) -> int:
    """Mengonversi nilai timestamp (int, float, atau ISO string) ke integer nanodetik secara aman."""
    if val is None:
        return time.time_ns()
    if isinstance(val, (int, float)):
        # Deteksi epoch detik (< 1e11)
        if val < 1e11:
            return int(val * 1e9)
        # Deteksi epoch milidetik (< 1e14)
        elif val < 1e14:
            return int(val * 1e6)
        # Deteksi epoch mikrodetik (< 1e17)
        elif val < 1e17:
            return int(val * 1e3)
        return int(val)
    if isinstance(val, str):
        val_str = val.strip()
        if val_str.isdigit():
            return int(val_str)
        try:
            # Parse ISO Format String
            dt = datetime.fromisoformat(val_str.replace("Z", "+00:00"))
            return int(dt.timestamp() * 1e9)
        except Exception:
            pass
    return time.time_ns()


# ==============================================================================
# CUSTOM EXCEPTIONS
# ==============================================================================
class StorageError(Exception):
    """Base Exception untuk seluruh kesalahan operasional pada modul storage.py."""
    pass

class DatabaseConnectionError(StorageError):
    """Pengecualian untuk kegagalan koneksi database SQLite."""
    pass

class SchemaMigrationError(StorageError):
    """Pengecualian untuk kegagalan inisialisasi atau migrasi skema tabel."""
    pass


# ==============================================================================
# 1. SQLITE ENGINE BASE
# ==============================================================================
class SQLiteEngine:
    """Engine Manajemen Koneksi Database SQLite Berbasis Thread-Local."""

    def __init__(self, db_path: str = "data/forex_storage_dryrun.db") -> None:
        self.db_path = db_path
        parent_dir = os.path.dirname(self.db_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        self._local = threading.local()
        self.init_db()

    def get_connection(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(self.db_path, timeout=30.0)
            self._local.conn.execute("PRAGMA foreign_keys = ON;")
            self._local.conn.execute("PRAGMA journal_mode = WAL;")
        return self._local.conn

    def init_db(self) -> None:
        conn = self.get_connection()
        with conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS signal_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    instrument TEXT NOT NULL,
                    signal INTEGER NOT NULL,
                    confidence REAL,
                    probability REAL,
                    timestamp_ns INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS prediction_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    instrument TEXT NOT NULL,
                    predicted_value REAL NOT NULL,
                    model_id TEXT NOT NULL,
                    timestamp_ns INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                );
            """)


# ==============================================================================
# 2. SIGNAL HISTORY STORE
# ==============================================================================
class SignalHistory:
    """Layanan Penyimpanan Log Sinyal Trading Kuantitatif Forex."""

    def __init__(self, sqlite_engine: SQLiteEngine) -> None:
        self.sqlite_engine = sqlite_engine

    def persist_signals(self, signals_payload: Union[pl.DataFrame, List[Dict[str, Any]], Dict[str, Any], None]) -> int:
        if signals_payload is None:
            return 0

        if isinstance(signals_payload, pl.DataFrame):
            records = signals_payload.to_dicts()
        elif isinstance(signals_payload, dict):
            if "orders" in signals_payload and isinstance(signals_payload["orders"], list):
                records = signals_payload["orders"]
            else:
                records = [signals_payload]
        elif isinstance(signals_payload, list):
            records = list(signals_payload)
        else:
            return 0

        if not records:
            return 0

        conn = self.sqlite_engine.get_connection()
        now_iso = datetime.now(timezone.utc).isoformat()
        inserted_count = 0

        with conn:
            cursor = conn.cursor()
            for sig in records:
                if not isinstance(sig, dict):
                    continue
                
                # Resolusi nama instrumen Forex (e.g. EUR_USD, GBP_USD)
                instrument = str(sig.get("instrument", sig.get("pair", sig.get("ticker", sig.get("symbol", "UNKNOWN")))))
                
                # Handling Sinyal Dual-Directional Forex (BUY=1, SELL=-1, HOLD=0)
                raw_sig = sig.get("signal", sig.get("signal_direction", sig.get("direction", sig.get("side", 0))))
                if isinstance(raw_sig, str):
                    raw_sig_upper = raw_sig.upper()
                    if raw_sig_upper in ["BUY", "LONG", "1"]:
                        signal_val = 1
                    elif raw_sig_upper in ["SELL", "SHORT", "-1"]:
                        signal_val = -1
                    else:
                        signal_val = 0
                else:
                    try:
                        signal_val = int(raw_sig)
                    except (ValueError, TypeError):
                        signal_val = 0

                conf = float(sig.get("confidence", sig.get("confidence_score", sig.get("signal_confidence", 0.0))))
                prob = float(sig.get("probability", sig.get("calibrated_prob", sig.get("signal_probability", 0.0))))
                
                raw_ts = sig.get("timestamp_ns", sig.get("timestamp", sig.get("date", None)))
                ts_ns = _safe_parse_timestamp_ns(raw_ts)

                cursor.execute("""
                    INSERT INTO signal_history (instrument, signal, confidence, probability, timestamp_ns, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (instrument, signal_val, conf, prob, ts_ns, now_iso))
                inserted_count += 1

        return inserted_count


# ==============================================================================
# 3. PREDICTION HISTORY STORE
# ==============================================================================
class PredictionHistory:
    """Layanan Audit Forensics Forecasting Lineage Forex."""

    def __init__(self, sqlite_engine: SQLiteEngine) -> None:
        self.sqlite_engine = sqlite_engine

    def persist_predictions(self, predictions_payload: Union[pl.DataFrame, List[Dict[str, Any]], None]) -> int:
        if predictions_payload is None:
            return 0

        if isinstance(predictions_payload, pl.DataFrame):
            records = predictions_payload.to_dicts()
        elif isinstance(predictions_payload, list):
            records = list(predictions_payload)
        else:
            return 0

        if not records:
            return 0

        conn = self.sqlite_engine.get_connection()
        now_iso = datetime.now(timezone.utc).isoformat()
        inserted_count = 0

        with conn:
            cursor = conn.cursor()
            for pred in records:
                if not isinstance(pred, dict):
                    continue
                
                instrument = str(pred.get("instrument", pred.get("pair", pred.get("asset_id", pred.get("asset", "UNKNOWN")))))
                pred_val = float(pred.get("predicted_value", pred.get("predicted_return", pred.get("prediction", 0.0))))
                model_id = str(pred.get("model_id", "DEFAULT_FOREX_MODEL"))
                
                raw_ts = pred.get("timestamp_ns", pred.get("timestamp", pred.get("date", None)))
                ts_ns = _safe_parse_timestamp_ns(raw_ts)

                cursor.execute("""
                    INSERT INTO prediction_history (instrument, predicted_value, model_id, timestamp_ns, created_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (instrument, pred_val, model_id, ts_ns, now_iso))
                inserted_count += 1

        return inserted_count


# ==============================================================================
# 4. UNIFIED STORAGE ENGINE (FACADE)
# ==============================================================================
class UnifiedStorageEngine:
    """
    Facade Utama Komponen Persistensi & Audit Database Storage OANDA Forex.
    Mendukung Pemisahan Database Terisolasi berdasarkan Mode Execution (Live vs Dry-Run).
    """

    def __init__(self, db_path: Optional[str] = None, mode: Optional[str] = None) -> None:
        self.mode = str(mode or os.getenv("EXECUTION_MODE", os.getenv("TRADING_MODE", "dry-run"))).lower().strip()
        self.is_live = self.mode in ["live", "force-rebalance"]
        self.state_suffix = "live" if self.is_live else "dryrun"

        if db_path is None:
            db_path = f"data/forex_storage_{self.state_suffix}.db"

        self.db_path = db_path
        self.sqlite_engine = SQLiteEngine(self.db_path)
        self.signal_store = SignalHistory(self.sqlite_engine)
        self.prediction_store = PredictionHistory(self.sqlite_engine)

    def persist_signals(self, signals_payload: Union[pl.DataFrame, List[Dict[str, Any]], Dict[str, Any], None]) -> int:
        """Alias langsung untuk persistensi sinyal Forex."""
        return self.signal_store.persist_signals(signals_payload)

    def persist_predictions(self, predictions_payload: Union[pl.DataFrame, List[Dict[str, Any]], None]) -> int:
        """Alias langsung untuk persistensi prediksi Forex."""
        return self.prediction_store.persist_predictions(predictions_payload)

    def persist_all(self, signals_df: Optional[Union[pl.DataFrame, List[Dict[str, Any]]]] = None, 
                    predictions_df: Optional[Union[pl.DataFrame, List[Dict[str, Any]]]] = None) -> Dict[str, int]:
        """Persistensi terpadu untuk sinyal dan prediksi sekaligus."""
        sig_count = self.persist_signals(signals_df) if signals_df is not None else 0
        pred_count = self.persist_predictions(predictions_df) if predictions_df is not None else 0
        return {"signals_persisted": sig_count, "predictions_persisted": pred_count}


if __name__ == "__main__":
    storage_engine = UnifiedStorageEngine()
    
    # Simple Smoke Test
    mock_signals = [
        {"instrument": "EUR_USD", "signal": "BUY", "confidence": 0.85, "probability": 0.72},
        {"instrument": "GBP_USD", "signal": "SELL", "confidence": 0.78, "probability": 0.65}
    ]
    
    res = storage_engine.persist_signals(mock_signals)
    print(f"Smoke Test Storage Engine Completed. Persisted {res} signal records into {storage_engine.db_path}.")
