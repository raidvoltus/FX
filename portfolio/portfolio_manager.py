"""
================================================================================
MODULE: portfolio/portfolio_manager.py
DESCRIPTION: Production Quantitative Portfolio Construction & Tracker Engine for OANDA Forex.
VERSION: 2026.1.0 (OANDA Forex Hierarchical Risk Parity & State Isolation Edition)
PYTHON VERSION: 3.11+ / 3.12+
ARCHITECTURE: Clean Architecture, Decoupled Portfolio Pipeline, HRP Allocation
================================================================================
"""

import os
import time
import json
import math
import logging
import threading
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import polars as pl
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform

from broker.oanda_client import OandaClient
from config import Config

logger = logging.getLogger("Forex.PortfolioManager")


# =============================================================================
# 1. HIERARCHICAL RISK PARITY (HRP) ALLOCATOR FOR FOREX
# =============================================================================
class ForexHRPAllocator:
    """
    Mesin Hierarchical Risk Parity (HRP) untuk Alokasi Bobot Risiko Forex.
    Menghitung bobot optimal tanpa memerlukan inversi matriks kovarians langsung.
    """

    def __init__(self, cov_reg: float = 1e-8):
        self.cov_reg = cov_reg

    def compute_hrp_weights(self, returns_df: pl.DataFrame) -> Dict[str, float]:
        """
        Menghitung bobot HRP dari DataFrame Return historis aset.
        """
        if returns_df.height < 10 or len(returns_df.columns) < 2:
            # Fallback ke Equal Weight jika data belum mencukupi
            asset_cols = [c for c in returns_df.columns if c not in ["datetime", "date", "timestamp"]]
            n = len(asset_cols)
            return {asset: 1.0 / max(n, 1) for asset in asset_cols}

        asset_cols = [c for c in returns_df.columns if c not in ["datetime", "date", "timestamp"]]
        matrix_data = returns_df.select(asset_cols).to_numpy()

        # 1. Matriks Kovarians & Korelasi
        cov = np.cov(matrix_data, rowvar=False)
        cov = (cov + cov.T) / 2.0
        cov += np.eye(cov.shape[0]) * self.cov_reg

        diag_std = np.sqrt(np.clip(np.diag(cov), 1e-8, None))
        outer_std = np.outer(diag_std, diag_std)
        corr = np.clip(cov / outer_std, -1.0, 1.0)
        np.fill_diagonal(corr, 1.0)

        # 2. Matriks Jarak Geometris (Distance Matrix)
        dist = np.sqrt(np.clip((1.0 - corr) / 2.0, 0.0, 1.0))
        np.fill_diagonal(dist, 0.0)

        # 3. Clustering Dendrogram (Hierarchical Linkage)
        condensed_dist = squareform(dist, checks=False)
        linkage_mat = sch.linkage(condensed_dist, method='single')
        tree_root = sch.to_tree(linkage_mat, rd=False)

        def get_leaves(node) -> List[int]:
            if node.is_leaf():
                return [node.id]
            return get_leaves(node.get_left()) + get_leaves(node.get_right())

        # 4. Recursive Bisection untuk Alokasi Bobot Risk Parity
        weights = np.ones(len(asset_cols), dtype=np.float64)
        queue = [(tree_root, 1.0)]

        while len(queue) > 0:
            node, curr_w = queue.pop(0)
            if node.is_leaf():
                weights[node.id] = curr_w
                continue

            left_leaves = get_leaves(node.get_left())
            right_leaves = get_leaves(node.get_right())

            # Variance Cluster Left vs Right
            cov_left = cov[np.ix_(left_leaves, left_leaves)]
            cov_right = cov[np.ix_(right_leaves, right_leaves)]

            inv_var_left = 1.0 / np.sum(np.diag(cov_left))
            inv_var_right = 1.0 / np.sum(np.diag(cov_right))

            alpha = inv_var_left / (inv_var_left + inv_var_right + 1e-15)

            queue.append((node.get_left(), curr_w * alpha))
            queue.append((node.get_right(), curr_w * (1.0 - alpha)))

        # Normalisasi Total Bobot = 1.0
        weights = weights / np.sum(weights)
        return {asset_cols[i]: float(weights[i]) for i in range(len(asset_cols))}


# =============================================================================
# 2. UNIFIED FOREX PORTFOLIO MANAGER & TRACKER
# =============================================================================
class PortfolioManager:
    """
    Pemantau Terpusat Portofolio & Eksekusi Konstruksi Alokasi Risiko HRP OANDA.
    Menangani isolasi state file (*_live_state.json vs *_dryrun_state.json).
    """

    def __init__(self, oanda_client: Optional[OandaClient] = None, mode: Optional[str] = None):
        self.config = Config.load()
        self.broker = oanda_client
        self._lock = threading.RLock()

        # Mode Resolution (Live vs Dry-Run)
        self.mode = str(mode or os.getenv("EXECUTION_MODE", "dry-run")).lower().strip()
        self.is_live = self.mode == "live"
        self.state_suffix = "live" if self.is_live else "dryrun"
        self.state_file = f"portfolio_{self.state_suffix}_state.json"

        self.hrp_allocator = ForexHRPAllocator()

    def get_account_balance(self) -> float:
        """Mengambil Saldo Kas (Balance) dari Broker OANDA."""
        if self.broker:
            try:
                summary = self.broker.get_account_summary()
                return float(summary.get("balance", 0.0))
            except Exception as e:
                logger.error(f"Gagal mengambil balance dari OANDA: {e}")
        return 10000.0 if not self.is_live else 0.0

    def get_account_nav(self) -> float:
        """Mengambil Net Asset Value (NAV / Equity) termasuk Floating PnL."""
        if self.broker:
            try:
                summary = self.broker.get_account_summary()
                return float(summary.get("nav", 0.0))
            except Exception as e:
                logger.error(f"Gagal mengambil NAV dari OANDA: {e}")
        return 10000.0 if not self.is_live else 0.0

    def get_open_positions(self) -> List[Dict[str, Any]]:
        """Mengambil daftar posisi aktif yang sedang terbuka dari OANDA."""
        if self.broker:
            try:
                summary = self.broker.get_account_summary()
                positions = summary.get("open_positions", [])
                return positions if isinstance(positions, list) else []
            except Exception as e:
                logger.error(f"Gagal mengambil posisi aktif dari OANDA: {e}")
        return []

    def get_summary(self) -> Dict[str, Any]:
        """Menyusun ringkasan lengkap portofolio terkini."""
        balance = self.get_account_balance()
        nav = self.get_account_nav()
        positions = self.get_open_positions()

        return {
            "mode": self.mode,
            "balance": balance,
            "nav": nav,
            "unrealized_pnl": nav - balance,
            "open_positions_count": len(positions),
            "open_positions": positions,
            "last_updated": datetime.now(timezone.utc).isoformat()
        }

    def save_state(self, portfolio_summary: Optional[Dict[str, Any]] = None) -> bool:
        """Menyimpan data state portofolio secara permanen ke file JSON."""
        with self._lock:
            try:
                data = portfolio_summary or self.get_summary()
                data["last_updated"] = datetime.now(timezone.utc).isoformat()
                
                with open(self.state_file, "w") as f:
                    json.dump(data, f, indent=2)
                logger.info(f"💾 [PORTFOLIO STATE STORED] Saved state to {self.state_file}")
                return True
            except Exception as e:
                logger.error(f"❌ Gagal menyimpan {self.state_file}: {e}")
                return False

    def load_state(self) -> Dict[str, Any]:
        """Membaca data state portofolio dari file JSON terisolasi."""
        with self._lock:
            if os.path.exists(self.state_file):
                try:
                    with open(self.state_file, "r") as f:
                        return json.load(f)
                except Exception as e:
                    logger.warning(f"⚠️ Gagal membaca {self.state_file}: {e}")

            return self.get_summary()

    def optimize_portfolio_weights(self, returns_df: pl.DataFrame) -> Dict[str, float]:
        """Menghitung alokasi bobot optimal menggunakan Hierarchical Risk Parity (HRP)."""
        logger.info("📊 Menjalankan Alokasi Bobot Portofolio Hierarchical Risk Parity (HRP)...")
        return self.hrp_allocator.compute_hrp_weights(returns_df)
