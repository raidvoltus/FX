# broker/oanda_client.py
import time
import logging
import pandas as pd
import tpqoa
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class OandaClient:
    """
    Wrapper aman di atas tpqoa dengan penanganan retry automatatis, 
    rate-limiting, dan manajemen error API.
    """

    def __init__(self, config_path: str = "oanda.cfg", max_retries: int = 3, retry_delay: int = 5):
        self.config_path = config_path
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.api: Optional[tpqoa.tpqoa] = None
        self._connect()

    def _connect(self) -> None:
        """Membuka koneksi ke OANDA API dengan skenario retry."""
        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(f"Mencoba menghubungkan ke OANDA API (Percobaan {attempt}/{self.max_retries})...")
                self.api = tpqoa.tpqoa(self.config_path)
                logger.info("Berhasil terhubung ke OANDA API.")
                return
            except Exception as e:
                logger.error(f"Gagal terhubung ke OANDA: {e}")
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay)
                else:
                    raise ConnectionError("Gagal menghubungkan ke OANDA API setelah beberapa kali percobaan.")

    def _execute_with_retry(self, func, *args, **kwargs):
        """Helper internal untuk mengeksekusi panggilan API dengan aman."""
        for attempt in range(1, self.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"OANDA API error pada {func.__name__}: {e}. Retry {attempt}/{self.max_retries}...")
                if attempt == self.max_retries:
                    logger.error(f"Panggilan {func.__name__} gagal secara permanen.")
                    raise e
                time.sleep(self.retry_delay)
                self._connect()  # Coba reconnect jika session terputus

    def get_instruments(self) -> List[tuple]:
        """Mengambil daftar instrumen yang tersedia dari OANDA."""
        return self._execute_with_retry(self.api.get_instruments)

    def get_historical_data(self, instrument: str, start: str, end: str, granularity: str, price: str = "M") -> pd.DataFrame:
        """
        Mengambil data candlestick histori.
        """
        df = self._execute_with_retry(
            self.api.get_history,
            instrument=instrument,
            start=start,
            end=end,
            granularity=granularity,
            price=price
        )
        return df

    def get_current_ask_bid(self, instrument: str) -> Dict[str, float]:
        """
        Mengambil harga Ask, Bid, dan Spread terkini untuk kalkulasi kuotasi & filter.
        """
        # tpqoa stream / getLastPrice wrapper logic
        try:
            # Menggunakan get_history baris terakhir untuk harga terkini yang stabil
            df = self.api.get_history(instrument=instrument, count=1, granularity="S5", price="BA")
            if not df.empty:
                ask = float(df['askc'].iloc[-1])
                bid = float(df['bidc'].iloc[-1])
                spread = ask - bid
                return {"ask": ask, "bid": bid, "spread": spread}
        except Exception as e:
            logger.error(f"Gagal mengambil harga Ask/Bid untuk {instrument}: {e}")
            raise e
        return {"ask": 0.0, "bid": 0.0, "spread": 0.0}

    def create_order(self, instrument: str, units: int, stop_loss: Optional[float] = None, take_profit: Optional[float] = None) -> Dict[str, Any]:
        """
        Mengeksekusi order pasar (Market Order) beserta pengaman SL/TP.
        """
        logger.info(f"Mengirim Order: {instrument} | Units: {units} | SL: {stop_loss} | TP: {take_profit}")
        try:
            order_response = self._execute_with_retry(
                self.api.create_order,
                instrument=instrument,
                units=units,
                sl_price=stop_loss,
                tp_price=take_profit
            )
            return order_response
        except Exception as e:
            logger.error(f"Gagal mengeksekusi order {instrument}: {e}")
            raise e

    def get_account_summary(self) -> Dict[str, Any]:
        """Mengambil ringkasan akun (Balance, NAV, Open Positions, Margin)."""
        return {
            "balance": float(self.api.balance),
            "nav": float(self.api.NAV),
            "unrealized_pnl": float(self.api.unrealized_profit),
            "open_positions": self.api.get_positions()
        }
