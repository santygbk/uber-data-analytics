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
from src.metrics import compute_all_metrics

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

    logger.info("=== Tabla maestra lista ===")
    logger.info("Tabla maestra: %d viajes, %d columnas", *master.shape)
    logger.info("Guardado en: %s", csv_path)

    # --- Métricas -----------------------------------------------------------
    logger.info("=== Calculando métricas ===")
    metricas = compute_all_metrics(master)

    trips_enriquecido_path = PROCESSED_DIR / "trips_enriquecido.csv"
    metricas["trips_enriquecido"].to_csv(trips_enriquecido_path, index=False)

    tabla_referencia_path = PROCESSED_DIR / "tabla_referencia_dia_hora.csv"
    metricas["tabla_referencia_dia_hora"].to_csv(tabla_referencia_path, index=False)

    heatmap_path = PROCESSED_DIR / "heatmap_ganancia_hora.csv"
    metricas["heatmap_ganancia_hora"].to_csv(heatmap_path)

    sesiones_path = PROCESSED_DIR / "sesiones_trabajo.csv"
    metricas["sesiones_trabajo"].to_csv(sesiones_path, index=False)

    logger.info("Tabla enriquecida guardada en: %s", trips_enriquecido_path)
    logger.info("Tabla de referencia día/hora guardada en: %s", tabla_referencia_path)
    logger.info("Heatmap guardado en: %s", heatmap_path)

    logger.info("=== Pipeline terminado ===")

    print("\n--- Cobertura de datos ---")
    for k, v in metricas["cobertura"].items():
        print(f"  {k}: {v}")

    print("\n--- Resumen general (tasa activa: solo tiempo con pasajero) ---")
    for k, v in metricas["resumen_general"].items():
        print(f"  {k}: {v}")

    print("\n--- Resumen de sesiones de trabajo (incluye tiempo de espera entre viajes) ---")
    for k, v in metricas["resumen_sesiones"].items():
        print(f"  {k}: {v}")

    print("\n--- Ganancia por día de la semana (activa: solo con pasajero) ---")
    print(metricas["ganancia_por_dia_semana"].to_string(index=False))

    print("\n--- Ganancia por día de la semana (CONECTADO: incluye espera entre viajes) ---")
    print(metricas["ganancia_por_dia_semana_conectado"].to_string(index=False))

    print("\n--- Ganancia por franja horaria (CONECTADO: incluye espera entre viajes) ---")
    print(metricas["ganancia_por_franja_horaria_conectado"].to_string(index=False))

    print("\n--- Preview de la tabla maestra (primeras 5 filas) ---")
    print(master.head().to_string())
    print(f"\nColumnas de la tabla maestra ({len(master.columns)}): {list(master.columns)}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    run_pipeline()