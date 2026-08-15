"""
================================================================================
PROJECT     : FXP - Forex Autonomous Trading Bot
REPOSITORY  : https://github.com/raidvoltus/FX
MODULE      : persistence/state_store.py
VERSION     : 2026.Q3.v4.4.0 (Atomic File Swap & Remote Snapshot Sync)
PYTHON      : 3.11+
================================================================================
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

logger = logging.getLogger("FXP.StateStore")
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('[%(asctime)s] [%(levelname)s] [StateStore] %(message)s'))
    logger.addHandler(ch)
    logger.setLevel(logging.INFO)


class StateStore:
    """
    Penyimpanan State Lokal Terpusat Berbasis Atomic JSON Write.
    Menjamin integritas data dari korupsi file saat crash/power loss,
    serta mendukung rekonsiliasi snapshot remote OANDA secara lengkap.
    """

    def __init__(self, store_dir: str = "data") -> None:
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.store_dir / "system_state.json"
        self.tmp_file = self.store_dir / "system_state.json.tmp"

    def _atomic_write(self, data: Dict[str, Any]) -> bool:
        """
        Menulis data JSON ke file sementara (.tmp) lalu melakukan pertukaran atomis (os.replace).
        Mencegah korupsi data jika proses mati di tengah operasi penulisan.
        """
        try:
            data["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
            with open(self.tmp_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            
            # Atomic swap menggantikan file lama secara aman pada level OS
            os.replace(self.tmp_file, self.state_file)
            return True
        except Exception as e:
            logger.error("💥 Gagal melakukan penulisan atomis ke %s: %s", self.state_file, e)
            if self.tmp_file.exists():
                try:
                    self.tmp_file.unlink()
                except Exception:
                    pass
            return False

    def save_state(self, state_data: Dict[str, Any]) -> bool:
        """Menyimpan snapshot status sistem ke file JSON secara atomis."""
        if not isinstance(state_data, dict):
            logger.error("State data harus berupa dictionary.")
            return False
        
        success = self._atomic_write(state_data)
        if success:
            logger.info("💾 [STATE_SAVED] System state tersimpan atomis ke %s", self.state_file)
        return success

    def load_state(self) -> Optional[Dict[str, Any]]:
        """Memuat status sistem terakhir dari disk."""
        if not self.state_file.exists():
            return None

        try:
            with open(self.state_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            logger.info("⚡ [STATE_LOADED] State dimuat dari %s", self.state_file)
            return data
        except Exception as e:
            logger.error("⚠️ [STATE_LOAD_ERROR] Gagal memuat state sistem: %s", e)
            return None

    def update_snapshot(self, remote_state: Dict[str, Any]) -> bool:
        """
        Memperbarui snapshot state lokal dengan data authoritative dari OANDA
        (GET /v3/accounts/{accountID} result) saat startup reconciliation.
        """
        current_state = self.load_state() or {}

        current_state["remote_snapshot"] = {
            "positions": remote_state.get("positions", []),
            "trades": remote_state.get("trades", []),
            "orders": remote_state.get("orders", []),
            "last_transaction_id": remote_state.get("last_transaction_id"),
            "oanda_request_id": remote_state.get("oanda_request_id", "N/A"),
            "synced_at_utc": datetime.now(timezone.utc).isoformat()
        }

        return self.save_state(current_state)

    def update_positions_snapshot(self, positions_data: List[Dict[str, Any]]) -> bool:
        """Fallback helper untuk sinkronisasi posisi terbuka saja."""
        current_state = self.load_state() or {}
        if "remote_snapshot" not in current_state:
            current_state["remote_snapshot"] = {}
        
        current_state["remote_snapshot"]["positions"] = positions_data
        current_state["remote_snapshot"]["synced_at_utc"] = datetime.now(timezone.utc).isoformat()
        return self.save_state(current_state)
