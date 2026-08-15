"""
================================================================================
PROJECT     : FXP - Forex Autonomous Trading Bot
REPOSITORY  : https://github.com/raidvoltus/FX
MODULE      : broker/oanda_client.py
VERSION     : 2026.Q3.v4.3.0 (Production OANDA REST v20 Client)
PYTHON      : 3.11+
================================================================================
"""

import logging
from typing import Dict, Any, List, Tuple, Optional

import polars as pl
import oandapyV20
import oandapyV20.endpoints.accounts as accounts
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.positions as positions
from oandapyV20.exceptions import V20Error

logger = logging.getLogger("FXP.OandaClient")
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('[%(asctime)s] [%(levelname)s] [OandaClient]: %(message)s'))
    logger.addHandler(ch)
    logger.setLevel(logging.INFO)


class OandaClient:
    """
    Client Wrapper Resmi OANDA REST API v20.
    Menyediakan antarmuka terisolasi untuk otentikasi akun, validasi kredensial,
    penarikan candle, rekonsiliasi posisi, dan eksekusi order.
    """

    def __init__(self, api_url: str, api_token: str, account_id: str) -> None:
        self.api_url = api_url.strip().rstrip("/")
        self.api_token = api_token.strip()
        self.account_id = account_id.strip()

        # Membersihkan protokol URL untuk format hostname oandapyV20
        cleaned_hostname = self.api_url.replace("https://", "").replace("http://", "")

        try:
            self.api = oandapyV20.API(
                access_token=self.api_token,
                hostname=cleaned_hostname,
                headers={"Content-Type": "application/json"}
            )
            logger.info("OandaClient terinisialisasi pada hostname: %s", cleaned_hostname)
        except Exception as err:
            logger.critical("Gagal menginisialisasi oandapyV20 API Context: %s", err)
            raise

    def validate_account_credentials(self) -> Tuple[bool, str]:
        """
        Validasi Otentikasi Remote Akun OANDA:
        1. Menguji keabsahan API Bearer Token.
        2. Mengonfirmasi OANDA_ACCOUNT_ID berada dalam daftar akun terotorisasi.
        3. Memeriksa ketersediaan margin dan status aktif akun.
        """
        try:
            # Step 1: Request daftar akun terotorisasi
            req_list = accounts.AccountList()
            res_list = self.api.request(req_list)
            
            authorized_accounts = [acc.get("id") for acc in res_list.get("accounts", [])]
            if self.account_id not in authorized_accounts:
                return (
                    False, 
                    f"Account ID '{self.account_id}' tidak ditemukan pada token API ini. Authorized list: {authorized_accounts}"
                )

            # Step 2: Verify account summary status
            req_summary = accounts.AccountSummary(accountID=self.account_id)
            res_summary = self.api.request(req_summary)
            acc_info = res_summary.get("account", {})

            if acc_info.get("marginRate") is None:
                return False, "Akun OANDA tidak mengembalikan marginRate yang valid."

            balance = acc_info.get("balance", "0.0")
            margin_avail = acc_info.get("marginAvailable", "0.0")

            return True, f"Account Validated | Balance: {balance} | Margin Available: {margin_avail}"

        except V20Error as err:
            return False, f"OANDA v20 API Error [{err.code}]: {err.msg}"
        except Exception as err:
            return False, f"Gagal mengeksekusi request validasi OANDA: {str(err)}"

    def get_instruments(self) -> List[Dict[str, Any]]:
        """Mengambil daftar instrumen mata uang yang tersedia untuk akun ini."""
        try:
            req = accounts.AccountInstruments(accountID=self.account_id)
            res = self.api.request(req)
            return res.get("instruments", [])
        except Exception as err:
            logger.error("Gagal mengambil daftar instrumen dari OANDA: %s", err)
            return []

    def get_candles(self, instrument: str, granularity: str = "M5", count: int = 500) -> pl.DataFrame:
        """
        Menarik data candlestick historis dari OANDA dan mengonversinya ke Polars DataFrame.
        """
        params = {
            "granularity": granularity,
            "count": count,
            "price": "M"  # Midpoint Prices
        }
        try:
            req = instruments.InstrumentsCandles(instrument=instrument, params=params)
            res = self.api.request(req)
            candles_raw = res.get("candles", [])

            if not candles_raw:
                return pl.DataFrame()

            records = []
            for c in candles_raw:
                if not c.get("complete", False):
                    continue
                mid = c.get("mid", {})
                records.append({
                    "timestamp": c.get("time"),
                    "open": float(mid.get("o", 0.0)),
                    "high": float(mid.get("h", 0.0)),
                    "low": float(mid.get("l", 0.0)),
                    "close": float(mid.get("c", 0.0)),
                    "volume": int(c.get("volume", 0))
                })

            if not records:
                return pl.DataFrame()

            return pl.DataFrame(records)

        except Exception as err:
            logger.error("Gagal menarik candle data [%s]: %s", instrument, err)
            return pl.DataFrame()

    def fetch_open_positions(self) -> List[Dict[str, Any]]:
        """
        Mengambil daftar seluruh posisi terbuka (Open Positions) dari OANDA Server
        untuk keperluan Startup Reconciliation.
        """
        try:
            req = positions.OpenPositions(accountID=self.account_id)
            res = self.api.request(req)
            return res.get("positions", [])
        except Exception as err:
            logger.error("Gagal mengambil posisi terbuka dari OANDA: %s", err)
            return []

    def execute_order(
        self, 
        instrument: str, 
        units: float, 
        order_type: str, 
        stop_loss: float, 
        take_profit: float
    ) -> Dict[str, Any]:
        """
        Mengirimkan Market Order berpasangan Stop Loss dan Take Profit langsung ke OANDA server.
        order_type: "BUY" / "LONG" (unit positif) atau "SELL" / "SHORT" (unit negatif).
        """
        is_buy = order_type.upper() in ["BUY", "LONG", "1"]
        signed_units = abs(units) if is_buy else -abs(units)

        # Formatting presisi string unit dan harga untuk OANDA REST API
        units_str = str(int(signed_units)) if float(signed_units).is_integer() else f"{signed_units:.2f}"
        
        order_payload = {
            "order": {
                "units": units_str,
                "instrument": instrument,
                "timeInForce": "FOK",
                "type": "MARKET",
                "positionFill": "DEFAULT",
                "stopLossOnFill": {
                    "price": f"{stop_loss:.5f}",
                    "timeInForce": "GTC"
                },
                "takeProfitOnFill": {
                    "price": f"{take_profit:.5f}",
                    "timeInForce": "GTC"
                }
            }
        }

        try:
            req = orders.OrderCreate(accountID=self.account_id, data=order_payload)
            res = self.api.request(req)

            fill_trans = res.get("orderFillTransaction", {})
            if fill_trans:
                exec_price = float(fill_trans.get("price", 0.0))
                logger.info(
                    "🎉 Market Order TERISI [%s] %s | Units: %s | Price: %.5f",
                    instrument, order_type.upper(), units_str, exec_price
                )
                return {
                    "status": "EXECUTED",
                    "order_id": fill_trans.get("id"),
                    "instrument": instrument,
                    "direction": order_type.upper(),
                    "units": signed_units,
                    "price": exec_price,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "raw_response": res
                }
            else:
                logger.warning("Order terkirim namun tidak langsung terisi: %s", res)
                return {"status": "PENDING_OR_CANCELLED", "raw_response": res}

        except V20Error as err:
            logger.error("💥 OANDA Order Execution V20Error [%s]: %s", err.code, err.msg)
            return {"status": "REJECTED", "error_code": err.code, "error_msg": str(err.msg)}
        except Exception as err:
            logger.error("💥 Gagal mengeksekusi order [%s]: %s", instrument, err)
            return {"status": "ERROR", "error_msg": str(err)}
