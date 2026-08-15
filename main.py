"""
================================================================================
PROJECT     : FXP (Forex Autonomous Trading Bot)
REPOSITORY  : https://github.com/raidvoltus/FXP
MODULE      : main.py
DESCRIPTION : Production Entry Point & Autonomous Loop Orchestrator.
VERSION     : 2026.Q3.v3.0.0 (FXP Production Synchronized Edition)
PYTHON      : 3.11+ / 3.12+
ARCHITECTURE: Pure Env-Vars Driven, Fully Integrated Quantitative Pipeline
================================================================================
"""

import sys
import time
import signal
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional

import polars as pl

# Import Seluruh Modul Kuantitatif Terintegrasi Repositori FXP
from config import Config
from autonomous_engine import AutonomousEngine
from signal_forex import UnifiedForexSignalEngine
from evaluation import UnifiedEvaluationEngine
from validation import UnifiedValidationEngine
from self_learning import UnifiedSelfLearningEngine
from storage import UnifiedStorageEngine

# Logging Setup
logger = logging.getLogger("FXP.MainOrchestrator")
if not logger.handlers:
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter('[%(asctime)s] [%(levelname)s] [FXP.Orchestrator]: %(message)s'))
    logger.addHandler(ch)
    logger.setLevel(logging.INFO)


class FXPAutonomousOrchestrator:
    """
    Pengendali Utama (Main Orchestrator) Repositori FXP.
    Mengoordinasikan alur eksekusi autonomis OANDA Forex, penanganan sinyal 9-Gateway,
    evaluasi performa, validasi risiko, serta persistensi data terisolasi.
    """

    def __init__(self):
        self.is_running: bool = False

        # 1. Muat Konfigurasi Utama (Single Source of Truth dari Env Vars)
        try:
            self.config = Config.load()
            self.account_type = str(getattr(self.config, "OANDA_ACCOUNT_TYPE", "demo")).upper()
            self.environment = str(getattr(self.config, "OANDA_ENVIRONMENT", "practice")).lower()
            self.poll_interval = int(getattr(self.config, "POLL_INTERVAL_SECONDS", 300))
        except Exception as err:
            logger.critical(f"❌ Gagal memuat variabel lingkungan (Config/Env Error): {err}")
            sys.exit(1)

        logger.info("==================================================")
        logger.info("🚀 MEMULAI FXP AUTONOMOUS TRADING ENGINE")
        logger.info(f"🔗 Repositori      : https://github.com/raidvoltus/FXP")
        logger.info(f"🔑 Tipe Akun OANDA : {self.account_type}")
        logger.info(f"🌐 Environment API  : {self.environment.upper()}")
        logger.info(f"⏱️ Loop Interval    : {self.poll_interval} Detik")
        logger.info(f"🕒 Waktu Sistem     : {datetime.now(timezone.utc).isoformat()}")
        logger.info("==================================================")

        # 2. Inisialisasi Seluruh Engine Kuantitatif
        try:
            self.autonomous_engine = AutonomousEngine()
            self.signal_engine = UnifiedForexSignalEngine()
            self.evaluation_engine = UnifiedEvaluationEngine()
            self.validation_engine = UnifiedValidationEngine()
            self.self_learning_engine = UnifiedSelfLearningEngine()
            self.storage_engine = UnifiedStorageEngine()
            
            logger.info("✅ Seluruh Sub-Engine FXP berhasil terhubung dan aktif.")
        except Exception as err:
            logger.critical(f"❌ Gagal menginisialisasi modul sistem FXP: {err}", exc_info=True)
            sys.exit(1)

        # 3. Handling Sinyal OS untuk Graceful Shutdown (Docker/Linux SIGINT/SIGTERM)
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

    def _handle_shutdown(self, signum: int, frame: Any) -> None:
        """Prosedur penghentian aman saat menerima sinyal OS."""
        logger.warning(f"\n⚠️ Sinyal shutdown OS ({signum}) diterima. Menghentikan alur FXP Engine...")
        self.is_running = False

    def run_single_cycle(self) -> Dict[str, Any]:
        """
        Menjalankan 1 alur siklus eksekusi autonomis penuh.
        (Market Ingestion -> ML Signal Generation -> Risk Sizing -> Execution -> 9-Gateway Filtering -> Storage)
        """
        cycle_start_time = time.perf_counter()
        logger.info(f"\n⚡ Executing FXP Autonomous Cycle @ {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")

        try:
            # Step 1: Eksekusi Siklus Utama via Autonomous Engine
            cycle_result = self.autonomous_engine.run_cycle()

            if cycle_result.get("status") != "SUCCESS":
                logger.warning(f"⚠️ Siklus dilewati/dihentikan. Alasan: {cycle_result.get('reason')}")
                return cycle_result

            executed_orders = cycle_result.get("executed_orders", [])

            # Step 2: Validasi 9-Gateway Signal Engine & Persistensi Storage
            if executed_orders:
                df_orders = pl.DataFrame(executed_orders)
                
                # Pemrosesan Sinyal melalui 9-Gateway Pipeline Forex
                df_validated_signals = self.signal_engine.process_signals(df_orders)
                
                # Persistensi Sinyal Terfilter ke SQLite Storage Terisolasi
                persisted_counts = self.storage_engine.persist_signals(df_validated_signals)
                logger.info(f"💾 Persistensi Database: {persisted_counts} sinyal tersimpan ke {self.storage_engine.db_path}.")

            elapsed_time_ms = (time.perf_counter() - cycle_start_time) * 1000.0
            logger.info(f"✅ Siklus FXP selesai dalam {elapsed_time_ms:.2f} ms.")

            return {
                "status": "SUCCESS",
                "executed_orders_count": len(executed_orders),
                "latency_ms": elapsed_time_ms
            }

        except Exception as err:
            logger.error(f"💥 Anomali fatal pada siklus eksekusi FXP: {err}", exc_info=True)
            return {"status": "ERROR", "reason": str(err)}

    def start_loop(self, run_once: bool = False) -> None:
        """
        Memutar siklus eksekusi autonomis secara berkelanjutan.
        """
        if run_once:
            logger.info("🏃 Menjalankan eksekusi 1 siklus tunggal (Run-Once Mode)...")
            self.run_single_cycle()
            self._cleanup()
            return

        self.is_running = True
        logger.info(f"🔄 FXP Autonomous Loop Aktif. Polling interval: {self.poll_interval} detik.")

        cycle_count = 0
        while self.is_running:
            cycle_count += 1
            logger.info(f"\n=== SIKLUS AUTONOMOUS FXP #{cycle_count} ===")

            self.run_single_cycle()

            # Listener jeda sleep non-blocking agar responsif terhadap sinyal interrupt
            for _ in range(self.poll_interval):
                if not self.is_running:
                    break
                time.sleep(1)

        self._cleanup()

    def _cleanup(self) -> None:
        """Prosedur pembersihan resource saat engine dimatikan."""
        logger.info("🧹 Memulai pembersihan resource dan penghentian koneksi engine...")
        try:
            self.signal_engine.deactivate_all()
            logger.info("✅ UnifiedForexSignalEngine berhasil dinonaktifkan.")
        except Exception as err:
            logger.error(f"Gagal mematikan ForexSignalEngine: {err}")

        logger.info("🛑 FXP Autonomous Engine dimatikan secara bersih.")


def main():
    """Entry Point Utama Aplikasi Repositori FXP."""
    import argparse

    parser = argparse.ArgumentParser(description="FXP Forex Autonomous Trading Bot Orchestrator")
    parser.add_argument("--once", action="store_true", help="Jalankan 1 siklus saja lalu keluar (Single Execution Test)")
    args = parser.parse_args()

    orchestrator = FXPAutonomousOrchestrator()
    orchestrator.start_loop(run_once=args.once)


if __name__ == "__main__":
    main()
