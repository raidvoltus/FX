"""
================================================================================
MODULE: risk/position_sizer.py
DESCRIPTION: Adaptive ATR & Risk-Budgeting Position Sizing Engine for Forex.
VERSION: 2026.1.0 (OANDA Volatility-Adjusted Sizing Edition)
PYTHON VERSION: 3.11+ / 3.12+
ARCHITECTURE: Clean Architecture, Dynamic Risk Budgeting
================================================================================
"""

import math
import logging
from typing import Dict, Any

from config import Config

logger = logging.getLogger("Forex.PositionSizer")


class PositionSizer:
    """
    Menghitung jumlah unit transaksi Forex secara otomatis berdasarkan:
    1. Account Equity (Modal Akun)
    2. Risk Budget % (Persentase Toleransi Risiko per Trade, e.g. 1%)
    3. Stop Loss Distance (Jarak SL dalam Pips/Harga)
    4. Quoted Instrument Factor (Direct Pairs vs JPY Pairs)
    """

    def __init__(self, account_risk_pct: Optional[float] = None):
        self.config = Config.load()
        self.risk_pct = account_risk_pct or self.config.MAX_RISK_PER_TRADE_PCT

    def calculate_units(
        self, 
        account_balance: float, 
        entry_price: float, 
        stop_loss_price: float, 
        instrument: str
    ) -> int:
        """
        Kalkulasi Jumlah Units Adaptif:
        Risk Amount ($) = Account Balance * Risk_PCT
        Stop Loss Distance ($) = |Entry Price - Stop Loss Price|
        Raw Units = Risk Amount / Stop Loss Distance
        """
        if account_balance <= 0 or entry_price <= 0 or stop_loss_price <= 0:
            logger.warning(f"[{instrument}] Parameter input Position Sizer tidak valid. Multiplier dibatalkan.")
            return 0

        sl_distance = abs(entry_price - stop_loss_price)
        if sl_distance == 0:
            logger.warning(f"[{instrument}] Jarak Stop Loss 0. Units di-set ke 0 untuk mencegah pembagian dengan nol.")
            return 0

        # Risk budget dalam Dolar ($)
        risk_budget_dollars = account_balance * self.risk_pct

        # Memperhitungkan pasangan JPY (Skala 100) vs Non-JPY (Skala 10000)
        raw_units = risk_budget_dollars / sl_distance

        # Pembulatan integer units (OANDA menerima integer units)
        units = int(math.floor(raw_units))

        # Batasan Pengaman: Minimal 1 unit jika valid
        if units < 1:
            logger.info(f"[{instrument}] Units hasil kalkulasi ({raw_units:.2f}) di bawah 1. Position sizer dibatalkan.")
            return 0

        logger.info(
            f"📊 [{instrument}] Position Sizing Result -> Balance: ${account_balance:.2f} | "
            f"Risk Budget ({self.risk_pct*100}%): ${risk_budget_dollars:.2f} | "
            f"SL Distance: {sl_distance:.5f} -> Calculated Units: {units}"
        )

        return units
