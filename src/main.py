"""
main.py
-------
Orquestador del pipeline ETL: ingesta -> preprocesamiento -> guardado.

Uso:
    python -m src.main
(ejecutar parado en la raíz del proyecto, uber-data-analytics/)
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.ingestion import load_config, load_raw_data
from src.preprocessing import build_master_trips_table

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "Uber Data" / "Driver"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def run_pipeline() -> None:
    logger.info("=== Iniciando pipeline ETL ===")

    cfg = load_config(PROJECT_ROOT / "config.yaml")
    dfs = load_raw_data(cfg, raw_dir=RAW_DIR)

    if "driver_lifetime_trips" not in dfs:
        raise RuntimeError(
            "No se pudo cargar driver_lifetime_trips-0.csv. "
            "Sin este archivo no hay tabla maestra posible. Revisá RAW_DIR."
        )

    master = build_master_trips_table(dfs)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = PROCESSED_DIR / "trips_master.csv"
    parquet_path = PROCESSED_DIR / "trips_master.parquet"

    master.to_csv(csv_path, index=False)
    try:
        master.to_parquet(parquet_path, index=False)
    except ImportError:
        logger.warning("pyarrow no está instalado, se omite el guardado en Parquet.")

    logger.info("=== Pipeline terminado ===")
    logger.info("Tabla maestra: %d viajes, %d columnas", *master.shape)
    logger.info("Guardado en: %s", csv_path)

    print("\n--- Preview de la tabla maestra (primeras 5 filas) ---")
    print(master.head().to_string())
    print(f"\nColumnas ({len(master.columns)}): {list(master.columns)}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    run_pipeline()