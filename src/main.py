"""
main.py
-------
Orquestador del pipeline ETL: ingesta -> preprocesamiento -> guardado.

Uso:
    python -m src.main
    python -m src.main --ultimos-n-viajes 200
    python -m src.main --ultimos-dias 30
(ejecutar parado en la raíz del proyecto, uber-data-analytics/)

--ultimos-n-viajes y --ultimos-dias son mutuamente excluyentes. Si no se
pasa ninguno, se usa el historial completo (comportamiento de siempre).
La tabla maestra completa (trips_master.csv) SIEMPRE se guarda entera,
sin filtrar -> es el registro histórico completo. El filtro solo afecta
qué subconjunto se usa para calcular métricas (trips_enriquecido,
tabla_referencia_dia_hora, heatmap, sesiones) y qué se imprime en consola.
Si se filtra, esos archivos se guardan con un sufijo (ej:
trips_enriquecido_ultimos200.csv) para no pisar la versión de histórico
completo.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.ingestion import load_config, load_raw_data
from src.preprocessing import build_master_trips_table
from src.metrics import compute_all_metrics

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "Uber Data" / "Driver"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def filtrar_periodo_reciente(
    master: pd.DataFrame,
    ultimos_n_viajes: int | None = None,
    ultimos_dias: int | None = None,
) -> tuple[pd.DataFrame, str, str]:
    """
    Recorta la tabla maestra a un período reciente, ordenando de más nuevo
    a más antiguo por begintrip_timestamp_local.

    Viajes sin begintrip_timestamp_local válido se excluyen del recorte
    (no se puede saber qué tan "recientes" son) -> se pierden sea cual sea
    el filtro elegido, salvo que no se filtre nada.

    Returns
    -------
    (df_filtrado, descripcion_legible, sufijo_para_nombres_de_archivo)
    """
    if ultimos_n_viajes is None and ultimos_dias is None:
        return master, "historial completo", ""

    if ultimos_n_viajes is not None and ultimos_dias is not None:
        raise ValueError("--ultimos-n-viajes y --ultimos-dias son mutuamente excluyentes, usá solo uno.")
    # eliminamos las columnas de begin NaN
    m = master.dropna(subset=["begintrip_timestamp_local"]).sort_values(
        "begintrip_timestamp_local", ascending=False
    )
    excluidos = len(master) - len(m)
    if excluidos:
        logger.warning(
            "filtrar_periodo_reciente: %d viajes sin begintrip_timestamp_local "
            "válido quedan excluidos del recorte (no se puede saber qué tan recientes son).",
            excluidos,
        )

    if ultimos_dias is not None:
        # fecha mas reciente - ultimos_dias
        corte = m["begintrip_timestamp_local"].max() - pd.Timedelta(days=ultimos_dias)
        # filtrado por mascara booleana, filtramos m[] con una condicion dentro de los corchetes, en este caso, que la fecha sea mayor o igual a corte (lista de booleans)
        m = m[m["begintrip_timestamp_local"] >= corte]
        descripcion = f"últimos {ultimos_dias} días"
        sufijo = f"_ultimos{ultimos_dias}dias"
    else:
        # filtrado por cantidad de viajes, tomamos los ultimos n viajes
        m = m.head(ultimos_n_viajes)
        descripcion = f"últimos {ultimos_n_viajes} viajes"
        sufijo = f"_ultimos{ultimos_n_viajes}"

    m = m.sort_values("begintrip_timestamp_local")
    logger.info(
        "filtrar_periodo_reciente: %s -> %d viajes seleccionados (de %s a %s).",
        descripcion, len(m),
        m["begintrip_timestamp_local"].min(), m["begintrip_timestamp_local"].max(),
    )
    return m, descripcion, sufijo


def run_pipeline(
    ultimos_n_viajes: int | None = None,
    ultimos_dias: int | None = None,
) -> None:
    logger.info("=== Iniciando pipeline ETL ===")

    cfg = load_config(PROJECT_ROOT / "config.yaml")
    dfs = load_raw_data(cfg, raw_dir=RAW_DIR)

    if "driver_lifetime_trips" not in dfs:
        raise RuntimeError(
            "No se pudo cargar driver_lifetime_trips-0.csv. "
            "Sin este archivo no hay tabla maestra posible. Revisá RAW_DIR."
        )

    master = build_master_trips_table(dfs)

    filtered_master, periodo_descripcion, sufijo = filtrar_periodo_reciente(
        master,
        ultimos_n_viajes=ultimos_n_viajes,
        ultimos_dias=ultimos_dias,
    )

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

    logger.info("=== Calculando métricas para %s ===", periodo_descripcion)
    metricas = compute_all_metrics(filtered_master)

    trips_enriquecido_path = PROCESSED_DIR / f"trips_enriquecido{sufijo}.csv"
    metricas["trips_enriquecido"].to_csv(trips_enriquecido_path, index=False)

    tabla_referencia_path = PROCESSED_DIR / f"tabla_referencia_dia_hora{sufijo}.csv"
    metricas["tabla_referencia_dia_hora"].to_csv(tabla_referencia_path, index=False)

    heatmap_path = PROCESSED_DIR / f"heatmap_ganancia_hora{sufijo}.csv"
    metricas["heatmap_ganancia_hora"].to_csv(heatmap_path)

    sesiones_path = PROCESSED_DIR / f"sesiones_trabajo{sufijo}.csv"
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

    print("--- Preview de la tabla maestra filtrada (primeras 5 filas) ---")
    print(filtered_master.head().to_string())
    print(
        f"\nColumnas de la tabla maestra filtrada ({len(filtered_master.columns)}): "
        f"{list(filtered_master.columns)}"
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    parser = argparse.ArgumentParser(description="Pipeline ETL de Uber Driver Analytics")
    grupo = parser.add_mutually_exclusive_group()
    grupo.add_argument(
        "--ultimos-n-viajes",
        type=int,
        help="Filtra los datos a los últimos N viajes según begintrip_timestamp_local",
    )
    grupo.add_argument(
        "--ultimos-dias",
        type=int,
        help="Filtra los datos a los últimos D días según begintrip_timestamp_local",
    )
    args = parser.parse_args()

    run_pipeline(
        ultimos_n_viajes=args.ultimos_n_viajes,
        ultimos_dias=args.ultimos_dias,
    )