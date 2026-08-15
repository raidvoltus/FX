"""
================================================================================
PROJECT     : FXP - Forex Autonomous Trading Bot
REPOSITORY  : https://github.com/raidvoltus/FX
MODULE      : broker/oanda_client.py
VERSION     : 2026.Q3.v4.6.0 (Strict Fail-Closed & Exception Enforcement)
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
import oandapyV20.endpoints.pricing as pricing
from oandapyV20.exceptions import V20Error

logger = logging.getLogger("FXP.OandaClient")


class BrokerError(Exception):
    pass

class BrokerStateError(BrokerError):
    pass

class MarketDataUnavailableError(BrokerError):
    pass


class OandaClient:

    ALLOWED_HOSTNAMES = {
        "api-fxpractice.oanda.com",
        "api-fxtrade.oanda.com"
    }

    def __init__(self, api_url: str, api_token: str, account_id: str) -> None:
        self.api_url = api_url.strip().rstrip("/")
        self.api_token = api_token.strip()
        self.account_id = account_id.strip()

        parsed = urlparse(self.api_url)
        hostname = (parsed.hostname or "").lower().rstrip(".")

        if hostname not in self.ALLOWED_HOSTNAMES:
            raise ValueError(f"CRITICAL: Hostname API '{hostname}' tidak berada dalam whitelist {self.ALLOWED_HOSTNAMES}")

        try:
            self.api = oandapyV20.API(
                access_token=self.api_token,
                hostname=hostname,
                headers={"Content-Type": "application/json"}
            )
        except Exception as err:
            raise BrokerError(f"Gagal menginisialisasi oandapyV20 Context: {err}") from err

        self.instrument_specs: Dict[str, Dict[str, Any]] = {}

    def load_instrument_specs(self) -> None:
        try:
            req = accounts.AccountInstruments(accountID=self.account_id)
            res = self.api.request(req)
            for inst in res.get("instruments", []):
                name = inst.get("name")
                if name:
                    self.instrument_specs[name] = {
                        "displayPrecision": int(inst.get("displayPrecision", 5)),
                        "tradeUnitsPrecision": int(inst.get("tradeUnitsPrecision", 0)),
                        "pipLocation": int(inst.get("pipLocation", -4)),
                        "minimumTradeSize": float(inst.get("minimumTradeSize", 1.0)),
                        "marginRate": float(inst.get("marginRate", 0.05))
                    }
        except Exception as err:
            raise BrokerStateError(f"Gagal memuat metadata instrumen dari OANDA: {err}") from err

    def format_price(self, instrument: str, price: float) -> str:
        if instrument not in self.instrument_specs:
            raise BrokerStateError(f"Metadata instrumen tidak tersedia di cache: {instrument}")
        precision = self.instrument_specs[instrument]["displayPrecision"]
        return f"{price:.{precision}f}"

    def format_units(self, instrument: str, units: float) -> str:
        if instrument not in self.instrument_specs:
            raise BrokerStateError(f"Metadata instrumen tidak tersedia di cache: {instrument}")
        precision = self.instrument_specs[instrument]["tradeUnitsPrecision"]
        if precision == 0:
            return str(int(round(units)))
        return f"{units:.{precision}f}"

    def get_executable_price(self, instrument: str, direction: str) -> float:
        params = {"instruments": instrument}
        try:
            req = pricing.PricingInfo(accountID=self.account_id, params=params)
            res = self.api.request(req)
            prices = res.get("prices", [])
            if not prices:
                raise MarketDataUnavailableError(f"Harga eksekusi [{instrument}] tidak ditemukan.")

            p_data = prices[0]
            if direction.upper() in ["BUY", "LONG", "1"]:
                asks = p_data.get("asks", [])
                if asks:
                    return float(asks[0]["price"])
            else:
                bids = p_data.get("bids", [])
                if bids:
                    return float(bids[0]["price"])

            raise MarketDataUnavailableError(f"Kedalaman pasar [{instrument}] tidak memuat harga {direction}.")
        except Exception as err:
            raise MarketDataUnavailableError(f"Gagal mengambil harga eksekusi [{instrument}]: {err}") from err

    def validate_account_credentials(self) -> Tuple[bool, str]:
        try:
            req_list = accounts.AccountList()
            res_list = self.api.request(req_list)
            
            authorized = [acc.get("id") for acc in res_list.get("accounts", [])]
            if self.account_id not in authorized:
                return False, f"Account ID '{self.account_id}' tidak terdaftar pada token API ini."

            req_summary = accounts.AccountSummary(accountID=self.account_id)
            res_summary = self.api.request(req_summary)
            acc_info = res_summary.get("account", {})

            if acc_info.get("id") != self.account_id:
                return False, "Remote Account ID mismatch."
            if acc_info.get("marginRate") is None or acc_info.get("marginAvailable") is None:
                return False, "Metadata margin akun tidak valid."

            margin_closeout_pct = float(acc_info.get("marginCloseoutPercent", 0.0))
            if margin_closeout_pct >= 1.0:
                return False, f"CRITICAL: Akun dalam posisi margin closeout ({margin_closeout_pct * 100:.2f}%)."

            self.load_instrument_specs()
            return True, f"Account OK | Balance: {acc_info.get('balance')} | Margin Available: {acc_info.get('marginAvailable')}"

        except V20Error as err:
            return False, f"OANDA API Error [{err.code}]: {err.msg}"
        except Exception as err:
            return False, f"Gagal mengeksekusi validasi OANDA: {str(err)}"

    def get_account_state(self) -> Dict[str, Any]:
        try:
            req = accounts.AccountDetails(accountID=self.account_id)
            res = self.api.request(req)
            account = res.get("account")

            if not isinstance(account, dict):
                raise BrokerStateError("Response AccountDetails OANDA mengembalikan struktur kosong.")

            headers = getattr(req.response, "headers", {}) if hasattr(req, "response") else {}
            req_id = headers.get("RequestID") or headers.get("x-request-id") or "N/A"

            return {
                "account": account,
                "positions": account.get("positions", []),
                "trades": account.get("trades", []),
                "orders": account.get("orders", []),
                "last_transaction_id": res.get("lastTransactionID"),
                "oanda_request_id": req_id
            }
        except V20Error as err:
            raise BrokerStateError(f"OANDA AccountDetails error [{err.code}]: {err.msg}") from err
        except Exception as err:
            raise BrokerStateError(f"Gagal mengambil remote account state: {err}") from err

    def get_candles(self, instrument: str, granularity: str = "M5", count: int = 500) -> pl.DataFrame:
        params = {"granularity": granularity, "count": count, "price": "M"}
        try:
            req = instruments.InstrumentsCandles(instrument=instrument, params=params)
            res = self.api.request(req)
            candles_raw = res.get("candles", [])

            if not candles_raw:
                raise MarketDataUnavailableError(f"OANDA mengembalikan data candle kosong untuk {instrument}")

            records = [
                {
                    "timestamp": c.get("time"),
                    "open": float(c["mid"]["o"]),
                    "high": float(c["mid"]["h"]),
                    "low": float(c["mid"]["l"]),
                    "close": float(c["mid"]["c"]),
                    "volume": int(c.get("volume", 0))
                }
                for c in candles_raw if c.get("complete", False)
            ]

            if not records:
                raise MarketDataUnavailableError(f"Tidak ada completed candle valid untuk {instrument}")

            return pl.DataFrame(records)

        except V20Error as err:
            raise MarketDataUnavailableError(f"OANDA Candle API error [{err.code}]: {err.msg}") from err
        except Exception as err:
            raise MarketDataUnavailableError(f"Gagal mengambil candle {instrument}: {err}") from err

    def execute_idempotent_order(
        self,
        instrument: str,
        units: float,
        order_type: str,
        stop_loss: float,
        take_profit: float,
        client_order_id: str
    ) -> Dict[str, Any]:
        if instrument not in self.instrument_specs:
            raise BrokerStateError(f"Metadata instrumen tidak tersedia di cache: {instrument}")

        min_size = self.instrument_specs[instrument]["minimumTradeSize"]
        if abs(units) < min_size:
            return {"status": "REJECTED", "client_order_id": client_order_id, "reason": f"UNITS_BELOW_MINIMUM ({abs(units)} < {min_size})"}

        is_buy = order_type.upper() in ["BUY", "LONG", "1"]
        signed_units = abs(units) if is_buy else -abs(units)

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
                "clientExtensions": {
                    "id": client_order_id,
                    "tag": "FXP_AUTONOMOUS_ENGINE"
                },
                "stopLossOnFill": {"price": formatted_sl, "timeInForce": "GTC"},
                "takeProfitOnFill": {"price": formatted_tp, "timeInForce": "GTC"}
            }
        }

        try:
            req = orders.OrderCreate(accountID=self.account_id, data=order_payload)
            res = self.api.request(req)
            
            headers = getattr(req.response, "headers", {}) if hasattr(req, "response") else {}
            request_id = headers.get("RequestID") or headers.get("x-request-id") or "N/A"

            fill_trans = res.get("orderFillTransaction", {})
            cancel_trans = res.get("orderCancelTransaction", {})
            create_trans = res.get("orderCreateTransaction", {})

            if fill_trans:
                return {
                    "status": "EXECUTED",
                    "client_order_id": client_order_id,
                    "order_id": create_trans.get("id"),
                    "fill_transaction_id": fill_trans.get("id"),
                    "oanda_request_id": request_id,
                    "instrument": instrument,
                    "direction": order_type.upper(),
                    "units": float(formatted_units),
                    "price": float(fill_trans.get("price", 0.0)),
                    "stop_loss": stop_loss,
                    "take_profit": take_profit
                }

            if cancel_trans:
                return {
                    "status": "REJECTED",
                    "client_order_id": client_order_id,
                    "reason": cancel_trans.get("reason", "ORDER_CANCELLED"),
                    "oanda_request_id": request_id,
                    "instrument": instrument
                }

            return {"status": "UNKNOWN", "client_order_id": client_order_id, "oanda_request_id": request_id}

        except V20Error as err:
            return {"status": "REJECTED", "client_order_id": client_order_id, "error_code": err.code, "error_msg": str(err.msg)}
        except Exception as err:
            logger.error("💥 Network/Timeout error saat POST order [%s]: %s", client_order_id, err)
            return {"status": "UNKNOWN", "client_order_id": client_order_id, "error_msg": str(err)}
