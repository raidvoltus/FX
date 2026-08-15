"""
=============================================================================
Module      : reporting.py (v2026.Q3 OANDA Forex Synchronized Edition)
Directory   : Flat Directory (Root Level with main.py)
Version     : 2026.Q3.v4.6.0 (Strict Audit & Data Integrity Edition)
Compliance  : OANDA v20 REST API Rules (Forex Pairs, Pip Precision, Dual-Side)
=============================================================================
"""

import os
import re
import time
import json
import hashlib
import logging
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse

import numpy as np
import polars as pl
import requests

logger = logging.getLogger("FXP.ReportingEngine")

INDONESIAN_MONTHS = [
    "Januari", "Februari", "Maret", "April", "Mei", "Juni",
    "Juli", "Agustus", "September", "Oktober", "November", "Desember"
]


def escape_markdown_v2(text: str) -> str:
    """Mengamankan sintaks MarkdownV2 Telegram dari karakter khusus."""
    if not text:
        return ""
    escape_chars = r"_*[]()~`>#+-=|{}.!"
    return re.sub(r"([" + re.escape(escape_chars) + r"])", r"\\\1", str(text))


def normalize_forex_instrument(symbol: Any) -> str:
    if not symbol:
        return "EUR_USD"
    cleaned = str(symbol).upper().strip()
    cleaned = re.sub(r"[^A-Z0-9_]", "", cleaned)
    if "_" not in cleaned and len(cleaned) == 6:
        cleaned = f"{cleaned[:3]}_{cleaned[3:]}"
    return cleaned


def format_forex_price(symbol: str, price: Optional[float], precision: Optional[int] = None) -> str:
    """Format harga dinamis tanpa membuat asumsi hardcoded."""
    if price is None or price <= 0:
        return "N/A"
    
    if precision is not None:
        return f"{price:.{precision}f}"
    
    symbol_upper = str(symbol).upper()
    dec = 3 if "JPY" in symbol_upper else 5
    return f"{price:.{dec}f}"


def mask_account_id(account_id: str) -> str:
    if not account_id or len(account_id) < 6:
        return "***"
    return f"{account_id[:3]}-***-***-{account_id[-3:]}"


class PerformanceSummarizer:
    """Kalkulator performa kuantitatif Forex berbasis 252 hari perdagangan."""

    def __init__(self, risk_free_rate: float = 0.02) -> None:
        self.risk_free_rate = float(risk_free_rate)

    def calculate_metrics(self, df: pl.DataFrame, return_col: str = "daily_return") -> Dict[str, float]:
        if df.is_empty() or return_col not in df.columns:
            return {"Annualized Return": 0.0, "Annualized Volatility": 0.0, "Sharpe Ratio": 0.0, "Max Drawdown": 0.0}

        returns = df.select(return_col).to_numpy().flatten()
        returns = returns[~np.isnan(returns) & ~np.isinf(returns)]
        if len(returns) == 0:
            return {"Annualized Return": 0.0, "Annualized Volatility": 0.0, "Sharpe Ratio": 0.0, "Max Drawdown": 0.0}

        n_obs = len(returns)
        cum_returns = np.cumprod(1.0 + returns)
        total_return = float(cum_returns[-1] - 1.0)
        ann_return = float((1.0 + total_return) ** (252.0 / max(n_obs, 1)) - 1.0)
        ann_vol = float(np.std(returns, ddof=1) * np.sqrt(252)) if n_obs > 1 else 0.0
        sharpe = float((ann_return - self.risk_free_rate) / ann_vol) if ann_vol > 0 else 0.0

        peak = np.maximum.accumulate(cum_returns)
        drawdowns = (cum_returns - peak) / peak
        max_dd = float(np.abs(np.min(drawdowns))) if len(drawdowns) > 0 else 0.0

        return {"Annualized Return": ann_return, "Annualized Volatility": ann_vol, "Sharpe Ratio": sharpe, "Max Drawdown": max_dd}


class SignalSummaryGenerator:

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.min_confidence = float(self.config.get("REPORTING_MIN_CONFIDENCE", 0.60))
        self.min_probability = float(self.config.get("REPORTING_MIN_PROBABILITY", 0.50))
        self.max_signals = int(self.config.get("REPORTING_MAX_SIGNALS", 10))
        self.framework_version = str(self.config.get("FRAMEWORK_VERSION", "2026.Q3.v4.6.0"))

    def _get_mode_header(self) -> str:
        api_url = str(self.config.get("OANDA_API_URL", os.environ.get("OANDA_API_URL", "")))
        parsed = urlparse(api_url.strip().rstrip("/"))
        hostname = (parsed.hostname or "").lower().rstrip(".")

        if hostname == "api-fxtrade.oanda.com":
            return "🚨 *[OANDA LIVE - REAL ACCOUNT]*"
        elif hostname == "api-fxpractice.oanda.com":
            return "🧪 *[OANDA PRACTICE - DEMO ACCOUNT]*"
        return "⚡ *[OANDA FOREX ENGINE]*"

    def load_portfolio_state(self) -> Tuple[Dict[str, Any], bool]:
        file_path = os.path.join("data", "system_state.json")
        default_state = {"equity": None, "balance": None, "margin_available": None, "active_positions_count": 0}

        if not os.path.exists(file_path):
            return default_state, True

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            updated_str = data.get("updated_at_utc") or data.get("remote_snapshot", {}).get("synced_at_utc")
            is_stale = True
            if updated_str:
                updated_dt = datetime.fromisoformat(updated_str.replace("Z", "+00:00"))
                age_seconds = (datetime.now(timezone.utc) - updated_dt).total_seconds()
                is_stale = age_seconds > 300

            remote = data.get("remote_snapshot", {})
            default_state["active_positions_count"] = len(remote.get("positions", [])) if remote else 0

            acc_sum = data.get("account_summary", {})
            if acc_sum and not is_stale:
                default_state["equity"] = float(acc_sum.get("NAV", acc_sum.get("balance", 0.0)))
                default_state["balance"] = float(acc_sum.get("balance", 0.0))
                default_state["margin_available"] = float(acc_sum.get("marginAvailable", 0.0))

            return default_state, is_stale
        except Exception as e:
            logger.warning("Gagal membaca berkas state: %s", e)
            return default_state, True

    def build_telegram_message(self, summary_payload: Dict[str, Any], portfolio_data: Optional[Dict[str, Any]] = None) -> str:
        p_state, is_stale = self.load_portfolio_state()
        if portfolio_data:
            p_state.update(portfolio_data)

        now_wib = datetime.now(timezone.utc) + timedelta(hours=7)
        month_str = INDONESIAN_MONTHS[now_wib.month - 1]
        header_time = f"{now_wib.day} {month_str} {now_wib.year} pukul {now_wib.strftime('%H:%M')} WIB"

        raw_acc_id = str(self.config.get("OANDA_ACCOUNT_ID", os.environ.get("OANDA_ACCOUNT_ID", "")))
        masked_acc = mask_account_id(raw_acc_id)

        state_status_str = "⚠️ *[STALE / UNKNOWN STATE]*" if is_stale else "🟢 *[LIVE SINKRON]*"

        equity_str = f"${p_state['equity']:,.2f} USD" if p_state.get('equity') is not None else "N/A"
        balance_str = f"${p_state['balance']:,.2f} USD" if p_state.get('balance') is not None else "N/A"
        margin_str = f"${p_state['margin_available']:,.2f} USD" if p_state.get('margin_available') is not None else "N/A"

        md = [
            "📊 *DASHBOARD PORTOFOLIO OANDA FOREX*",
            f"📍 Mode: {self._get_mode_header()}",
            f"🔑 Account: `{masked_acc}`",
            f"🗓️ {header_time}",
            f"📡 Status Data: {state_status_str}",
            "══════════════════════════════",
            "📌 *RINGKASAN AKUN & MARGIN*",
            f"💰 Total Ekuitas : `{equity_str}`",
            f"🏦 Balance       : `{balance_str}`",
            f"💵 Margin Avail  : `{margin_str}`",
            f"💼 Posisi Aktif  : `{p_state.get('active_positions_count', 0)} Pasangan`",
            "══════════════════════════════",
            "🔹 *SINYAL PERDAGANGAN FOREX PERIODE INI*"
        ]

        signals = summary_payload.get("signals", []) if summary_payload else []
        if not signals:
            md.append("⚠️ *Tidak ada sinyal aktif yang memenuhi standar kualifikasi periode ini.*")
        else:
            for sig in signals:
                inst_raw = normalize_forex_instrument(sig.get("instrument", ""))
                dir_str = str(sig.get("direction", "BUY")).upper()
                action_text = "BELI (LONG)" if dir_str in ["BUY", "LONG", "1"] else "JUAL (SHORT)"

                entry_val = float(sig.get("entry_price", 0.0))
                tp_val = sig.get("tp_price")
                sl_val = sig.get("sl_price")

                entry_str = format_forex_price(inst_raw, entry_val)
                tp_str = format_forex_price(inst_raw, float(tp_val) if tp_val is not None else None)
                sl_str = format_forex_price(inst_raw, float(sl_val) if sl_val is not None else None)

                md.extend([
                    f"🔸 *{inst_raw}*",
                    f"   💰 Harga Entry : `{entry_str}`",
                    f"   🎯 Target TP   : `{tp_str}`",
                    f"   🛑 Stop Loss   : `{sl_str}`",
                    f"   ✅ Probabilitas: `{float(sig.get('probability', 0.0))*100:.1f}%` | ❇️ Conf: `{float(sig.get('confidence', 0.0))*100:.1f}%`",
                    f"   📌 Rekomendasi : *{action_text}*",
                    ""
                ])

        return "\n".join(md)


class TelegramReporter:
    """Reporter Adapter tunggal dengan pemotongan aman berbasis baris (Line-Boundary Chunking)."""

    def __init__(self, config: Any = None) -> None:
        self.config = config
        self.token = os.environ.get("TELEGRAM_BOT_TOKEN") or os.environ.get("TELEGRAM_TOKEN")
        self.chat_id = os.environ.get("TELEGRAM_CHAT_ID") or os.environ.get("CHAT_ID")

    def send_message(self, message_text: str, parse_mode: str = "Markdown") -> bool:
        if not self.token or not self.chat_id:
            logger.warning("[TELEGRAM_REPORTER] Token/Chat ID tidak ditemukan. Transmisi dibatalkan.")
            return False

        lines = message_text.split("\n")
        chunks: List[str] = []
        current_chunk: List[str] = []
        current_len = 0

        for line in lines:
            if current_len + len(line) + 1 > 3900:
                chunks.append("\n".join(current_chunk))
                current_chunk = [line]
                current_len = len(line)
            else:
                current_chunk.append(line)
                current_len += len(line) + 1

        if current_chunk:
            chunks.append("\n".join(current_chunk))

        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        all_success = True

        for chunk in chunks:
            payload = {"chat_id": self.chat_id, "text": chunk, "parse_mode": parse_mode}
            try:
                res = requests.post(url, json=payload, timeout=15)
                if res.status_code != 200:
                    all_success = False
            except Exception as err:
                logger.error("[TELEGRAM_REPORTER] Transmisi gagal: %s", err)
                all_success = False

        return all_success

    def broadcast_signals(self, orders: Optional[List[Dict[str, Any]]] = None, portfolio_data: Optional[Dict[str, Any]] = None) -> bool:
        generator = SignalSummaryGenerator(config=self.config if isinstance(self.config, dict) else {})
        msg = generator.build_telegram_message({"signals": orders or []}, portfolio_data=portfolio_data)
        return self.send_message(msg, parse_mode="Markdown")
