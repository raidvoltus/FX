"""
================================================================================
MODULE: execution/order_manager.py
DESCRIPTION: Production-Grade Order Execution Engine for OANDA Forex API.
VERSION: 2026.1.0 (Forex Protected Execution & Retry Logic Edition)
PYTHON VERSION: 3.11+ / 3.12+
ARCHITECTURE: Clean Architecture, Broker Execution Wrapper
================================================================================
"""

import time
import logging
from typing import Dict, Any, Optional

from broker.oanda_client import OandaClient
from config import Config

logger = logging.getLogger("Forex.OrderManager")


class OrderManager:
    """
    Manajer Eksekusi Order yang Mengirimkan Market Orders beserta 
    Stop-Loss dan Take-Profit Terproteksi ke OANDA API.
    """

    def __init__(self, oanda_client: OandaClient):
        self.broker = oanda_client
        self.config = Config.load()

    def execute_order(
        self, 
        instrument: str, 
        units: int, 
        order_type: str, 
        stop_loss: Optional[float] = None, 
        take_profit: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Mengeksekusi order transaksi ke OANDA API.
        
        Param:
        - instrument: Symbol pair (e.g. 'EUR_USD')
        - units: Jumlah units (Positif untuk BUY, Negatif untuk SELL)
        - order_type: 'BUY' atau 'SELL'
        - stop_loss: Harga batas rugi
        - take_profit: Harga batas untung
        """
        if units == 0:
            logger.warning(f"[{instrument}] Units bernilai 0. Eksekusi order dibatalkan.")
            return {"status": "SKIPPED", "reason": "ZERO_UNITS"}

        # Penyesuaian Tanda Units (OANDA: Long = positive units, Short = negative units)
        final_units = abs(units) if order_type.upper() in ["BUY", "LONG", "1"] else -abs(units)

        logger.info(
            f"🚀 [ORDER SUBMIT] Executing {instrument} | Side: {order_type} | "
            f"Units: {final_units} | SL: {stop_loss} | TP: {take_profit}"
        )

        try:
            # Mengirimkan order via OandaClient
            response = self.broker.create_order(
                instrument=instrument,
                units=final_units,
                stop_loss=stop_loss,
                take_profit=take_profit
            )

            logger.info(f"✔ [{instrument}] Order Berhasil Diterima OANDA: {response}")
            return {
                "status": "SUCCESS",
                "instrument": instrument,
                "units": final_units,
                "response": response,
                "timestamp": time.time()
            }

        except Exception as e:
            logger.error(f"❌ [{instrument}] Gagal mengeksekusi order ke OANDA: {e}")
            return {
                "status": "FAILED",
                "instrument": instrument,
                "reason": str(e),
                "timestamp": time.time()
            }
