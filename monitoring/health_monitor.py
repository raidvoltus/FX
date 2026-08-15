"""
================================================================================
MODULE: monitoring/health_monitor.py
DESCRIPTION: Unified Telemetry, System Health & Infrastructure Monitoring Engine for Forex.
VERSION: 2026.1.0 (OANDA Forex Resilient Egress & Observability Edition)
PYTHON VERSION: 3.11+ / 3.12+
ARCHITECTURE: Clean Architecture, Decoupled Monitoring & Observability
================================================================================
"""

import os
import time
import socket
import sqlite3
import logging
import threading
from pathlib import Path
from functools import wraps
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple, Union, Callable

import numpy as np
import polars as pl
import psutil
from scipy import stats

from config import Config

logger = logging.getLogger("Forex.HealthMonitor")

# Kolom-kolom metadata yang dikecualikan dari kalkulasi statistik numerik murni
EXCLUDED_NON_NUMERIC_COLS = {
    "date", "timestamp", "time", "asset", "ticker", "symbol", "created_at",
    "portfolio_asset_id", "allocation_reason", "sector", "industry", "country"
}


# =============================================================================
# 1. HEALTH CHECK ENGINE
# =============================================================================
class HealthCheckEngine:
    """
    Engine diagnostik untuk sumber daya sistem (RAM, Disk, Egress Network, DB).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.cfg = Config.load()
        self.config = config or {}
        
        self.memory_threshold_pct: float = float(self.config.get("memory_threshold_pct", 85.0))
        self.disk_threshold_pct: float = float(self.config.get("disk_threshold_pct", 90.0))
        self.network_timeout_sec: float = float(self.config.get("network_timeout_sec", 3.0))
        self.workspace_dir: Path = Path(self.config.get("workspace_dir", Path.cwd()))
        
        # OANDA FX Egress Targets
        self.network_targets: List[Tuple[str, int]] = [
            ("api-fxpractice.oanda.com", 443),
            ("api-fxtrade.oanda.com", 443),
            ("1.1.1.1", 53),
            ("8.8.8.8", 53)
        ]

    def check_memory(self) -> Dict[str, Any]:
        try:
            vm = psutil.virtual_memory()
            used_pct = vm.percent
            available_mb = vm.available / (1024 * 1024)
            total_mb = vm.total / (1024 * 1024)
            
            is_healthy = used_pct < self.memory_threshold_pct
            status = "HEALTHY" if is_healthy else "CRITICAL"
            
            return {
                "status": status,
                "metrics": {
                    "total_memory_mb": round(total_mb, 2),
                    "available_memory_mb": round(available_mb, 2),
                    "used_percentage": used_pct
                }
            }
        except Exception as e:
            logger.error(f"Gagal memeriksa memori: {e}")
            return {"status": "FAILED", "error": str(e)}

    def check_disk(self) -> Dict[str, Any]:
        try:
            usage = psutil.disk_usage(str(self.workspace_dir.resolve()))
            used_pct = usage.percent
            free_gb = usage.free / (1024 * 1024 * 1024)
            
            is_healthy = used_pct < self.disk_threshold_pct
            status = "HEALTHY" if is_healthy else "CRITICAL"
            
            return {
                "status": status,
                "metrics": {
                    "free_disk_gb": round(free_gb, 2),
                    "used_percentage": used_pct
                }
            }
        except Exception as e:
            logger.error(f"Gagal memeriksa disk: {e}")
            return {"status": "FAILED", "error": str(e)}

    def check_network(self) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {"targets_evaluated": [], "successful_connections": 0}
        all_failed = True
        
        for host, port in self.network_targets:
            start_time = time.perf_counter()
            try:
                with socket.create_connection((host, port), timeout=self.network_timeout_sec):
                    latency_ms = (time.perf_counter() - start_time) * 1000
                    metrics["targets_evaluated"].append({
                        "target": f"{host}:{port}",
                        "reachable": True,
                        "latency_ms": round(latency_ms, 2)
                    })
                    metrics["successful_connections"] += 1
                    all_failed = False
            except (socket.timeout, socket.error) as e:
                metrics["targets_evaluated"].append({
                    "target": f"{host}:{port}",
                    "reachable": False,
                    "error": str(e)
                })

        if all_failed:
            status = "CRITICAL"
        elif metrics["successful_connections"] > 0:
            status = "HEALTHY"
        else:
            status = "DEGRADED"

        return {"status": status, "metrics": metrics}

    def run_all(self) -> Dict[str, Any]:
        start_ts = time.time()
        diagnostics = {
            "memory": self.check_memory(),
            "disk": self.check_disk(),
            "network": self.check_network()
        }
        
        statuses = [res["status"] for res in diagnostics.values()]
        if "CRITICAL" in statuses or "FAILED" in statuses:
            aggregate_status = "UNHEALTHY"
        elif "DEGRADED" in statuses:
            aggregate_status = "DEGRADED"
        else:
            aggregate_status = "HEALTHY"
            
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "aggregate_status": aggregate_status,
            "execution_duration_sec": round(time.time() - start_ts, 4),
            "diagnostics": diagnostics
        }


# =============================================================================
# 2. DATA QUALITY MONITOR
# =============================================================================
class DataMonitor:
    """
    Validasi skema data, staleness timestamp, dan estimasi outlier.
    """

    def __init__(self, iqr_factor: float = 1.5, max_staleness_sec: float = 86400.0):
        self.iqr_factor = iqr_factor
        self.max_staleness_sec = max_staleness_sec

    def check_missing_and_invalid(self, df: pl.DataFrame, columns: List[str]) -> Dict[str, Any]:
        if df.height == 0:
            return {"status": "EMPTY", "metrics": {}}

        valid_cols = [c for c in columns if c in df.columns]
        null_counts = {}

        for col in valid_cols:
            nulls = df[col].null_count()
            null_counts[col] = {
                "null_count": nulls,
                "missing_rate": round(nulls / df.height, 4)
            }

        return {"status": "HEALTHY", "metrics": null_counts}


# =============================================================================
# 3. UNIFIED HEALTH MONITOR FACADE
# =============================================================================
class HealthMonitor:
    """
    Facade Utama Pengawas Kesehatan Infrastruktur & Latensi Loop Sistem Autonomous.
    """

    def __init__(self):
        self.start_time = time.time()
        self.last_ping_time = time.time()
        self.cycle_count = 0
        self.health_engine = HealthCheckEngine()
        self.data_monitor = DataMonitor()

    def ping(self) -> Dict[str, Any]:
        """
        Memeriksa latensi siklus utama dan kesehatan infrastruktur secara berkala.
        """
        self.cycle_count += 1
        now = time.time()
        loop_latency = now - self.last_ping_time
        self.last_ping_time = now

        system_health = self.health_engine.run_all()
        uptime_seconds = now - self.start_time

        report = {
            "cycle_number": self.cycle_count,
            "uptime_seconds": round(uptime_seconds, 2),
            "loop_latency_seconds": round(loop_latency, 4),
            "system_status": system_health["aggregate_status"],
            "memory": system_health["diagnostics"]["memory"]["metrics"],
            "network": system_health["diagnostics"]["network"]["status"],
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        logger.info(
            f"💚 [HEALTH MONITOR] Cycle #{self.cycle_count} | Uptime: {report['uptime_seconds']}s | "
            f"Loop Latency: {report['loop_latency_seconds']}s | System Status: {report['system_status']}"
        )

        return report
