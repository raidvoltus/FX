"""
================================================================================
MODULE: persistence/state_store.py
DESCRIPTION: System State Persistence & Historical Audit Logging for Forex Bot.
VERSION: 2026.1.0 (State Persistence Edition)
PYTHON VERSION: 3.11+ / 3.12+
ARCHITECTURE: Clean Architecture, Persistence Layer
================================================================================
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional

logger = logging.getLogger("Forex.StateStore")


class StateStore:
    """
    Penyimpanan State Sistem Terpusat untuk Menyimpan Snapshot Eksekusi,
    Jejak Log Audit, dan Status Portofolio.
    """

    def __init__(self, store_dir: str = "data"):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.store_dir / "system_state.json"

    def save_state(self, state_data: Dict[str, Any]) -> bool:
        """
        Menyimpan snapshot status sistem ke file JSON.
        """
        try:
            state_data["saved_at_utc"] = datetime.now(timezone.utc).isoformat()
            with open(self.state_file, "w") as f:
                json.dump(state_data, f, indent=2)
            logger.info(f"💾 [STATE_STORE_SUCCESS] System state saved to {self.state_file}")
            return True
        except Exception as e:
            logger.error(f"❌ [STATE_STORE_ERROR] Gagal menyimpan state sistem: {e}")
            return False

    def load_state(self) -> Optional[Dict[str, Any]]:
        """
        Memuat status sistem terakhir dari disk.
        """
        if not self.state_file.exists():
            return None

        try:
            with open(self.state_file, "r") as f:
                data = json.load(f)
            logger.info(f"⚡ [STATE_LOADED] Loaded state from {self.state_file}")
            return data
        except Exception as e:
            logger.error(f"⚠️ [STATE_LOAD_ERROR] Gagal memuat state sistem: {e}")
            return None
