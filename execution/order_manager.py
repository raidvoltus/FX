"""
================================================================================
PROJECT     : FXP - Forex Autonomous Trading Bot
REPOSITORY  : https://github.com/raidvoltus/FX
MODULE      : execution/order_manager.py
VERSION     : 2026.Q3.v4.4.0 (Idempotent Protected Execution Edition)
PYTHON      : 3.11+
================================================================================
"""

import time
import uuid
import logging
from typing import Dict, Any, Optional

from broker.oanda_client import OandaClient
from config import Config

logger = logging.getLogger("FXP.OrderManager")
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('[%(asctime)s] [%(levelname)s] [OrderManager] %(message)s'))
    logger.addHandler(ch)
    logger.setLevel(logging.INFO)


class OrderManager:
    """
    Manajer Eksekusi Order Autonomis FXP.
    Menghubungkan AutonomousEngine dengan OandaClient menggunakan eksekusi
    idempotent terproteksi (Client Order ID, SL/TP mandatory, & status mapping).
    """

    def __init__(self, oanda_client: OandaClient) -> None:
        self.broker = oanda_client
        self.config = Config.load()

    def execute_order(
        self, 
        instrument: str, 
        units: float, 
        order_type: str, 
        stop_loss: float, 
        take_profit: float,
        client_order_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Mengeksekusi order terproteksi ke OANDA REST API via OandaClient.
        
        Param:
        - instrument: Symbol pair Forex (contoh: 'EUR_USD')
        - units: Jumlah unit/lot (float)
        - order_type: 'BUY' / 'LONG' atau 'SELL' / 'SHORT'
        - stop_loss: Harga batas rugi (Stop Loss)
        - take_profit: Harga batas untung (Take Profit)
        - client_order_id: ID unik opsional (dibuat otomatis jika None)
        """
        if abs(units) < 1e-8:
            logger.warning("[%s] Unit bernilai 0 atau di bawah ambang batas. Eksekusi dibatalkan.", instrument)
            return {"status": "SKIPPED", "reason": "ZERO_UNITS", "instrument": instrument}

        # Generate Idempotent Client Order ID jika belum disediakan oleh caller
        if not client_order_id:
            clean_inst = instrument.replace("_", "").upper()
            client_order_id = f"FXP-{clean_inst}-{int(time.time())}-{uuid.uuid4().hex[:6]}"

        logger.info(
            "🚀 [ORDER SUBMIT] %s | Side: %s | Units: %.2f | SL: %.5f | TP: %.5f | ClientID: %s",
            instrument, order_type.upper(), units, stop_loss, take_profit, client_order_id
        )

        try:
            # Memanggil eksekusi idempotent authoritative dari OandaClient v4.4.0
            result = self.broker.execute_idempotent_order(
                instrument=instrument,
                units=units,
                order_type=order_type,
                stop_loss=stop_loss,
                take_profit=take_profit,
                client_order_id=client_order_id
            )

            status = result.get("status", "UNKNOWN")
            if status == "EXECUTED":
                logger.info(
                    "✔ [%s] Order TERISI | Fill Price: %.5f | ReqID: %s", 
                    instrument, result.get("price", 0.0), result.get("oanda_request_id")
                )
            elif status == "REJECTED":
                logger.warning(
                    "🚫 [%s] Order DITOLAK OANDA | Reason: %s", 
                    instrument, result.get("reason") or result.get("error_msg")
                )

            return result

        except Exception as e:
            logger.error("💥 [%s] Anomali fatal saat memanggil OandaClient: %s", instrument, e, exc_info=True)
            return {
                "status": "UNKNOWN",
                "client_order_id": client_order_id,
                "instrument": instrument,
                "reason": str(e),
                "timestamp": time.time()
            }
