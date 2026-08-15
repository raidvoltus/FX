"""
================================================================================
PROJECT     : FXP - Forex Autonomous Trading Bot
REPOSITORY  : https://github.com/raidvoltus/FX
MODULE      : broker/oanda_client.py
VERSION     : 2026.Q3.v4.4.0 (Authoritative Account State & Fail-Safe Client)
PYTHON      : 3.11+
================================================================================
"""

import os
import logging
from typing import Dict, Any, List, Tuple, Optional
from urllib.parse import urlparse

import polars as pl
import oandapyV20
import oandapyV20.endpoints.accounts as accounts
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.positions as positions
import oandapyV20.endpoints.trades as trades
from oandapyV20.exceptions import V20Error

logger = logging.getLogger("FXP.OandaClient")
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('[%(asctime)s] [%(levelname)s] [OandaClient] %(message)s'))
    logger.addHandler(ch)
    logger.setLevel(logging.INFO)


class OandaClient:
    """
    Authoritative OANDA REST API v20 Client.
    Menerapkan fail-safe exceptions, presisi dinamik per instrumen,
    ekstraksi OANDA RequestID untuk audit trail, dan rekonsiliasi total akun.
    """

    def __init__(self, api_url: str, api_token: str, account_id: str) -> None:
        self.api_url = api_url.strip().rstrip("/")
        self.api_token = api_token.strip()
        self.account_id = account_id.strip()

        # Pembersihan hostname untuk oandapyV20 Context
        parsed = urlparse(self.api_url)
        cleaned_hostname = (parsed.netloc or parsed.path).lower().rstrip("/")

        try:
            self.api = oandapyV20.API(
                access_token=self.api_token,
                hostname=cleaned_hostname,
                headers={"Content-Type": "application/json"}
            )
            logger.info("OandaClient terinisialisasi pada hostname: %s", cleaned_hostname)
        except Exception as err:
            logger.critical("Gagal menginisialisasi oandapyV20 API Context: %s", err)
            raise RuntimeError(f"OandaClient initialization failure: {err}") from err

        # Cache metadata spesifikasi instrumen (displayPrecision & tradeUnitsPrecision)
        self.instrument_specs: Dict[str, Dict[str, Any]] = {}

    # =========================================================================
    # METADATA & PRECISION FORMATTING HELPERS
    # =========================================================================

    def load_instrument_specs(self) -> None:
        """Memuat spesifikasi presisi seluruh instrumen dari OANDA."""
        try:
            req = accounts.AccountInstruments(accountID=self.account_id)
            res = self.api.request(req)
            raw_instruments = res.get("instruments", [])

            for inst in raw_instruments:
                name = inst.get("name")
                if name:
                    self.instrument_specs[name] = {
                        "displayPrecision": int(inst.get("displayPrecision", 5)),
                        "tradeUnitsPrecision": int(inst.get("tradeUnitsPrecision", 0)),
                        "pipLocation": int(inst.get("pipLocation", -4)),
                        "marginRate": float(inst.get("marginRate", 0.05))
                    }
            logger.info("Spesifikasi %d instrumen OANDA berhasil dimuat ke cache.", len(self.instrument_specs))
        except Exception as err:
            logger.error("Gagal memuat spesifikasi instrumen OANDA: %s", err)
            raise RuntimeError(f"Gagal memuat metadata instrumen: {err}") from err

    def format_price(self, instrument: str, price: float) -> str:
        """Memformat harga sesuai displayPrecision spesifik instrumen."""
        spec = self.instrument_specs.get(instrument, {})
        precision = spec.get("displayPrecision", 5)
        return f"{price:.{precision}f}"

    def format_units(self, instrument: str, units: float) -> str:
        """Memformat jumlah unit sesuai tradeUnitsPrecision spesifik instrumen."""
        spec = self.instrument_specs.get(instrument, {})
        precision = spec.get("tradeUnitsPrecision", 0)
        if precision == 0:
            return str(int(round(units)))
        return f"{units:.{precision}f}"

    def _extract_request_id(self, req: Any) -> str:
        """Mengekstrak OANDA RequestID dari HTTP Response Header untuk audit trail."""
        try:
            if hasattr(req, "response") and req.response is not None:
                headers = getattr(req.response, "headers", {})
                return headers.get("RequestID") or headers.get("x-request-id") or "UNKNOWN"
        except Exception:
            pass
        return "N/A"

    # =========================================================================
    # AUTHENTICATION & ACCOUNT HEALTH VALIDATION
    # =========================================================================

    def validate_account_credentials(self) -> Tuple[bool, str]:
        """
        Validasi OANDA Account + Authentication + Authorization:
        1. GET /v3/accounts (Verifikasi Token Bearer)
        2. Match OANDA_ACCOUNT_ID pada daftar terotorisasi
        3. GET /v3/accounts/{accountID}/summary (Margin Available, Balance, Margin Closeout Status)
        """
        try:
            # Step 1: Request Account List
            req_list = accounts.AccountList()
            res_list = self.api.request(req_list)
            
            authorized_accounts = [acc.get("id") for acc in res_list.get("accounts", [])]
            if self.account_id not in authorized_accounts:
                return False, f"Account ID '{self.account_id}' tidak terdaftar pada token API ini."

            # Step 2: Request Account Summary
            req_summary = accounts.AccountSummary(accountID=self.account_id)
            res_summary = self.api.request(req_summary)
            acc_info = res_summary.get("account", {})

            # Strict Account Checks
            if acc_info.get("id") != self.account_id:
                return False, "Remote Account ID mismatch."
            if acc_info.get("marginRate") is None:
                return False, "Akun OANDA tidak mengembalikan marginRate yang valid."
            if acc_info.get("marginAvailable") is None:
                return False, "marginAvailable tidak tersedia."
            if acc_info.get("balance") is None:
                return False, "balance tidak tersedia."

            margin_closeout_pct = float(acc_info.get("marginCloseoutPercent", 0.0))
            if margin_closeout_pct >= 1.0:
                return False, f"CRITICAL: Account berada dalam kondisi margin closeout ({margin_closeout_pct * 100:.2f}%)."

            # Load Instrument Specs
            self.load_instrument_specs()

            balance = acc_info.get("balance")
            margin_avail = acc_info.get("marginAvailable")
            return True, f"Account Terverifikasi Aktif | Balance: {balance} | Margin Available: {margin_avail}"

        except V20Error as err:
            return False, f"OANDA v20 API Error [{err.code}]: {err.msg}"
        except Exception as err:
            return False, f"Gagal otentikasi REST API OANDA: {str(err)}"

    # =========================================================================
    # AUTHORITATIVE STATE & RECONCILIATION
    # =========================================================================

    def get_account_state(self) -> Dict[str, Any]:
        """
        Mengambil Snapshot Authoritative Akun (GET /v3/accounts/{accountID}).
        Mengembalikan Positions, Trades, Pending Orders, dan lastTransactionID dalam 1 call.
        """
        try:
            req = accounts.AccountDetails(accountID=self.account_id)
            res = self.api.request(req)
            account = res.get("account")

            if not isinstance(account, dict):
                raise RuntimeError("OANDA mengembalikan account state kosong atau tidak valid.")

            request_id = self._extract_request_id(req)

            return {
                "account": account,
                "positions": account.get("positions", []),
                "trades": account.get("trades", []),
                "orders": account.get("orders", []),
                "last_transaction_id": res.get("lastTransactionID"),
                "oanda_request_id": request_id
            }
        except V20Error as err:
            raise RuntimeError(f"OANDA account state error [{err.code}]: {err.msg}") from err
        except Exception as err:
            raise RuntimeError(f"Gagal mengambil remote account state: {err}") from err

    def fetch_open_positions(self) -> List[Dict[str, Any]]:
        """Mengambil posisi terbuka aktif dari OANDA Server (Fail-Safe: melempar Exception jika API gagal)."""
        try:
            req = positions.OpenPositions(accountID=self.account_id)
            res = self.api.request(req)
            return res.get("positions", [])
        except V20Error as err:
            raise RuntimeError(f"OANDA position API error [{err.code}]: {err.msg}") from err
        except Exception as err:
            raise RuntimeError(f"Gagal mengambil posisi terbuka OANDA: {err}") from err

    def fetch_open_trades(self) -> List[Dict[str, Any]]:
        """Mengambil trade individual aktif dari OANDA Server (Fail-Safe)."""
        try:
            req = trades.OpenTrades(accountID=self.account_id)
            res = self.api.request(req)
            return res.get("trades", [])
        except V20Error as err:
            raise RuntimeError(f"OANDA open trades API error [{err.code}]: {err.msg}") from err
        except Exception as err:
            raise RuntimeError(f"Gagal mengambil trade aktif OANDA: {err}") from err

    def fetch_pending_orders(self) -> List[Dict[str, Any]]:
        """Mengambil pending order aktif dari OANDA Server (Fail-Safe)."""
        try:
            req = orders.Orders(accountID=self.account_id, params={"state": "PENDING"})
            res = self.api.request(req)
            return res.get("orders", [])
        except V20Error as err:
            raise RuntimeError(f"OANDA pending orders API error [{err.code}]: {err.msg}") from err
        except Exception as err:
            raise RuntimeError(f"Gagal mengambil pending order OANDA: {err}") from err

    # =========================================================================
    # MARKET DATA (CANDLES & INSTRUMENTS)
    # =========================================================================

    def get_instruments(self) -> List[Dict[str, Any]]:
        """Mengambil daftar instrumen terdaftar dari OANDA (Fail-Safe)."""
        try:
            req = accounts.AccountInstruments(accountID=self.account_id)
            res = self.api.request(req)
            return res.get("instruments", [])
        except V20Error as err:
            raise RuntimeError(f"OANDA instruments API error [{err.code}]: {err.msg}") from err
        except Exception as err:
            raise RuntimeError(f"Gagal mengambil daftar instrumen OANDA: {err}") from err

    def get_candles(self, instrument: str, granularity: str = "M5", count: int = 500) -> pl.DataFrame:
        """
        Menarik data candlestick historis dari OANDA (Fail-Safe).
        Melempar Exception jika terjadi kegagalan jaringan/API. Empty DataFrame HANYA dikembalikan
        jika OANDA secara eksplisit merespons dengan 0 baris candle yang selesai.
        """
        params = {
            "granularity": granularity,
            "count": count,
            "price": "M"
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

            return pl.DataFrame(records)

        except V20Error as err:
            raise RuntimeError(f"OANDA candle API error [{err.code}]: {err.msg}") from err
        except Exception as err:
            raise RuntimeError(f"Gagal mengambil candle data {instrument}: {err}") from err

    # =========================================================================
    # ORDER EXECUTION ENGINE
    # =========================================================================

    def execute_order(
        self, 
        instrument: str, 
        units: float, 
        order_type: str, 
        stop_loss: float, 
        take_profit: float
    ) -> Dict[str, Any]:
        """
        Mengirim Market Order dengan presisi dinamik dan audit ID terstruktur.
        Mengurai orderFillTransaction, orderCancelTransaction, dan orderCreateTransaction.
        """
        is_buy = order_type.upper() in ["BUY", "LONG", "1"]
        signed_units = abs(units) if is_buy else -abs(units)

        # Menggunakan presisi dinamis per instrumen
        formatted_units = self.format_units(instrument, signed_units)
        formatted_sl = self.format_price(instrument, stop_loss)
        formatted_tp = self.format_price(instrument, take_profit)

        order_payload = {
            "order": {
                "units": formatted_units,
                "instrument": instrument,
                "timeInForce": "FOK",
                "type": "MARKET",
                "positionFill": "DEFAULT",
                "stopLossOnFill": {
                    "price": formatted_sl,
                    "timeInForce": "GTC"
                },
                "takeProfitOnFill": {
                    "price": formatted_tp,
                    "timeInForce": "GTC"
                }
            }
        }

        try:
            req = orders.OrderCreate(accountID=self.account_id, data=order_payload)
            res = self.api.request(req)
            request_id = self._extract_request_id(req)

            fill_trans = res.get("orderFillTransaction", {})
            cancel_trans = res.get("orderCancelTransaction", {})
            create_trans = res.get("orderCreateTransaction", {})

            # Debug flag sanitasi raw response
            debug_raw = os.getenv("FXP_DEBUG_RAW_OANDA", "false").lower() == "true"
            raw_data = res if debug_raw else None

            if fill_trans:
                exec_price = float(fill_trans.get("price", 0.0))
                logger.info(
                    "🎉 Order FILLED [%s] %s %s Units @ %.5f | ReqID: %s",
                    instrument, order_type.upper(), formatted_units, exec_price, request_id
                )
                return {
                    "status": "EXECUTED",
                    "order_id": create_trans.get("id"),
                    "fill_transaction_id": fill_trans.get("id"),
                    "oanda_request_id": request_id,
                    "instrument": instrument,
                    "direction": order_type.upper(),
                    "units": float(formatted_units),
                    "price": exec_price,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "raw_response": raw_data
                }

            if cancel_trans:
                reason = cancel_trans.get("reason", "UNKNOWN_CANCEL_REASON")
                logger.warning("🚫 Order CANCELLED/REJECTED [%s]: %s | ReqID: %s", instrument, reason, request_id)
                return {
                    "status": "REJECTED",
                    "order_id": create_trans.get("id"),
                    "cancel_transaction_id": cancel_trans.get("id"),
                    "oanda_request_id": request_id,
                    "instrument": instrument,
                    "reason": reason,
                    "raw_response": raw_data
                }

            return {
                "status": "UNKNOWN_RESULT",
                "order_id": create_trans.get("id"),
                "oanda_request_id": request_id,
                "instrument": instrument,
                "raw_response": raw_data
            }

        except V20Error as err:
            logger.error("💥 OANDA Order V20Error [%s]: %s", err.code, err.msg)
            return {
                "status": "REJECTED",
                "error_code": err.code,
                "error_msg": str(err.msg),
                "instrument": instrument
            }
        except Exception as err:
            logger.error("💥 Gagal mengeksekusi order [%s]: %s", instrument, err)
            raise RuntimeError(f"Order execution internal error [{instrument}]: {err}") from err
