"""
================================================================================
PROJECT     : FXP - Forex Autonomous Trading Bot
REPOSITORY  : https://github.com/raidvoltus/FX
MODULE      : broker/oanda_client.py
VERSION     : 2026.Q3.v4.4.0 (Fail-Closed & Dynamic Precision Client)
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


class OandaClient:

    def __init__(self, api_url: str, api_token: str, account_id: str) -> None:
        self.api_url = api_url.strip().rstrip("/")
        self.api_token = api_token.strip()
        self.account_id = account_id.strip()

        parsed = urlparse(self.api_url)
        cleaned_hostname = (parsed.netloc or parsed.path).lower().rstrip("/")

        try:
            self.api = oandapyV20.API(
                access_token=self.api_token,
                hostname=cleaned_hostname,
                headers={"Content-Type": "application/json"}
            )
        except Exception as err:
            raise RuntimeError(f"OandaClient Context Error: {err}") from err

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
                        "marginRate": float(inst.get("marginRate", 0.05))
                    }
        except Exception as err:
            raise RuntimeError(f"Gagal memuat metadata instrumen: {err}") from err

    def format_price(self, instrument: str, price: float) -> str:
        precision = self.instrument_specs.get(instrument, {}).get("displayPrecision", 5)
        return f"{price:.{precision}f}"

    def format_units(self, instrument: str, units: float) -> str:
        precision = self.instrument_specs.get(instrument, {}).get("tradeUnitsPrecision", 0)
        return str(int(round(units))) if precision == 0 else f"{units:.{precision}f}"

    def _extract_request_id(self, req: Any) -> str:
        try:
            if hasattr(req, "response") and req.response is not None:
                headers = getattr(req.response, "headers", {})
                return headers.get("RequestID") or headers.get("x-request-id") or "N/A"
        except Exception:
            pass
        return "N/A"

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
                return False, "Margin metadata tidak valid."

            margin_closeout_pct = float(acc_info.get("marginCloseoutPercent", 0.0))
            if margin_closeout_pct >= 1.0:
                return False, f"CRITICAL: Account margin closeout ({margin_closeout_pct * 100:.2f}%)."

            self.load_instrument_specs()
            return True, f"Account OK | Balance: {acc_info.get('balance')} | Margin Avail: {acc_info.get('marginAvailable')}"

        except V20Error as err:
            return False, f"OANDA API Error [{err.code}]: {err.msg}"
        except Exception as err:
            return False, f"Gagal otentikasi API OANDA: {str(err)}"

    def get_account_state(self) -> Dict[str, Any]:
        """Fail-Closed Remote State Reader (Account, Positions, Trades, Orders, lastTransactionID)."""
        try:
            req = accounts.AccountDetails(accountID=self.account_id)
            res = self.api.request(req)
            account = res.get("account")

            if not isinstance(account, dict):
                raise RuntimeError("Response AccountDetails tidak valid.")

            return {
                "account": account,
                "positions": account.get("positions", []),
                "trades": account.get("trades", []),
                "orders": account.get("orders", []),
                "last_transaction_id": res.get("lastTransactionID"),
                "oanda_request_id": self._extract_request_id(req)
            }
        except V20Error as err:
            raise RuntimeError(f"OANDA AccountDetails error [{err.code}]: {err.msg}") from err
        except Exception as err:
            raise RuntimeError(f"Gagal mengambil remote account state: {err}") from err

    def get_candles(self, instrument: str, granularity: str = "M5", count: int = 500) -> pl.DataFrame:
        """Fail-Closed Candle Data Fetcher."""
        params = {"granularity": granularity, "count": count, "price": "M"}
        try:
            req = instruments.InstrumentsCandles(instrument=instrument, params=params)
            res = self.api.request(req)
            candles_raw = res.get("candles", [])

            if not candles_raw:
                return pl.DataFrame()

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
            return pl.DataFrame(records)

        except V20Error as err:
            raise RuntimeError(f"OANDA Candle API error [{err.code}]: {err.msg}") from err
        except Exception as err:
            raise RuntimeError(f"Gagal menarik candle {instrument}: {err}") from err

    def execute_idempotent_order(
        self,
        instrument: str,
        units: float,
        order_type: str,
        stop_loss: float,
        take_profit: float,
        client_order_id: str
    ) -> Dict[str, Any]:
        """Eksekusi Order dengan Idempotent Client Extensions & Presisi Dinamis."""
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
            request_id = self._extract_request_id(req)

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
            raise RuntimeError(f"Network error saat POST order [{client_order_id}]: {err}") from err
