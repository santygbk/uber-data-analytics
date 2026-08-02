"""
ingestion.py
------------
Módulo de ingesta del pipeline ETL de Uber Driver Analytics.

Responsabilidad única: leer el archivo de configuración (config.yaml),
localizar los CSVs crudos en `data/raw/` y cargarlos en memoria como
DataFrames de pandas, conservando únicamente las columnas declaradas
en la configuración.

No realiza limpieza, casteo de tipos ni reglas de negocio: eso vive
en `preprocessing.py`. Este módulo solo se encarga de EXTRACT.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import pandas as pd
import yaml

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)

# Raíz del proyecto = una carpeta arriba de src/ (donde vive este archivo).
# Sirve para construir defaults que funcionan sin importar desde qué
# directorio se ejecute el script (evita el clásico bug de rutas relativas
# rotas al correr "python ingestion.py" parado dentro de src/).
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(config_path: str | Path = "config.yaml") -> dict:
    """
    Lee y parsea el archivo YAML de configuración.

    Parameters
    ----------
    config_path : str | Path
        Ruta al archivo config.yaml.

    Returns
    -------
    dict
        Diccionario con la estructura completa del YAML
        (clave raíz esperada: "archivos_uber").

    Raises
    ------
    FileNotFoundError
        Si el archivo de configuración no existe.
    ValueError
        Si el YAML no contiene la clave "archivos_uber".
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de configuración: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not config or "archivos_uber" not in config:
        raise ValueError(
            f"El archivo {config_path} no contiene la clave raíz 'archivos_uber'."
        )

    logger.info("Configuración cargada: %d archivos definidos.", len(config["archivos_uber"]))
    return config


def _clean_key(filename: str) -> str:
    """
    Convierte 'driver_lifetime_trips-0.csv' en la clave 'driver_lifetime_trips'.

    Elimina la extensión .csv y el sufijo numérico tipo '-0' que Uber
    agrega a sus exports, para que las claves del diccionario resultante
    sean estables y legibles en el resto del pipeline.
    """
    stem = Path(filename).stem  # quita .csv -> 'driver_lifetime_trips-0'
    if "-" in stem and stem.rsplit("-", 1)[-1].isdigit():
        stem = stem.rsplit("-", 1)[0]
    return stem


def _load_single_csv(file_path: Path, columns: list[str]) -> pd.DataFrame:
    """
    Carga un único CSV filtrando solo las columnas solicitadas.

    Si alguna columna definida en el config.yaml no existe en el CSV real,
    se registra un warning y se continúa solo con las columnas disponibles,
    en lugar de romper todo el pipeline por un desajuste de config.
    """
    # Primero leemos solo el header para validar qué columnas existen de verdad.
    header = pd.read_csv(file_path, nrows=0).columns.tolist()
    columnas_disponibles = [c for c in columns if c in header]
    columnas_faltantes = [c for c in columns if c not in header]

    if columnas_faltantes:
        logger.warning(
            "%s: columnas del config no encontradas en el CSV real y serán ignoradas: %s",
            file_path.name,
            columnas_faltantes,
        )

    if not columnas_disponibles:
        raise ValueError(
            f"Ninguna de las columnas configuradas para {file_path.name} existe en el archivo."
        )

    df = pd.read_csv(file_path, usecols=columnas_disponibles, low_memory=False)
    logger.info("%s: %d filas, %d columnas cargadas.", file_path.name, len(df), df.shape[1])
    return df


def load_raw_data(
    config: dict,
    raw_dir: str | Path = "data/raw",
) -> Dict[str, pd.DataFrame]:
    """
    Itera sobre 'archivos_uber' en el config, carga cada CSV desde raw_dir
    filtrando columnas, y devuelve un diccionario de DataFrames.

    Parameters
    ----------
    config : dict
        Diccionario devuelto por `load_config()`.
    raw_dir : str | Path
        Carpeta donde viven los CSVs originales de Uber.

    Returns
    -------
    Dict[str, pd.DataFrame]
        Ej: {"driver_lifetime_trips": df, "driver_app_analytics": df, ...}
        Los archivos que no se encuentren en disco se omiten con un warning,
        en vez de detener todo el pipeline.
    """
    raw_dir = Path(raw_dir)
    dataframes: Dict[str, pd.DataFrame] = {}

    for filename, columns in config["archivos_uber"].items():
        file_path = raw_dir / filename

        if not file_path.exists():
            logger.warning("Archivo no encontrado, se omite: %s", file_path)
            continue

        key = _clean_key(filename)
        try:
            dataframes[key] = _load_single_csv(file_path, columns)
        except ValueError as e:
            logger.error("Error cargando %s: %s", filename, e)

    logger.info("Ingesta finalizada: %d/%d archivos cargados.",
                len(dataframes), len(config["archivos_uber"]))
    return dataframes


if __name__ == "__main__":
    # Ahora se puede correr con "python -m src.ingestion" O "python src/ingestion.py"
    # O incluso "python ingestion.py" parado dentro de src/ -> las rutas ya no
    # dependen del directorio desde el que se lanza el script.
    cfg = load_config(PROJECT_ROOT / "config.yaml")
    dfs = load_raw_data(cfg, raw_dir=PROJECT_ROOT / "data" / "Uber Data" / "Driver")

    for nombre, df in dfs.items():
        print(f"\n=== {nombre} ===")
        print(df.head(3))
        print(df.dtypes)