"""
preprocessing.py
-----------------
Módulo de transformación (TRANSFORM) del pipeline ETL de Uber Driver Analytics.

Responsabilidades:
1. Limpiar y castear tipos (timestamps, booleanos) de cada DataFrame crudo.
2. Asignar un trip_id sintético a cada viaje (Uber no nos da uno en las
   columnas configuradas).
3. Unir driver_payments y driver_app_analytics a driver_lifetime_trips
   usando proximidad temporal, ya que no existe una clave común explícita.

=================================================================
SUPUESTOS CRÍTICOS — REVISAR ANTES DE CONFIAR EN LOS RESULTADOS
=================================================================
1. TRIP_ID: no hay un identificador de viaje compartido entre los 3 CSVs
   (según las columnas definidas en config.yaml). Si tus CSVs reales de
   Uber sí incluyen 'trip_uuid' (a veces viene fuera de lo que uno filtra
   a simple vista), avisame el nombre exacto de columna y reemplazamos
   todo este join aproximado por un join directo por ID, que es muchísimo
   más confiable.

2. ZONA HORARIA: 'Event Time (UTC)' en driver_app_analytics está en UTC,
   mientras que los timestamps de driver_lifetime_trips y driver_payments
   tienen sufijo "_local"/"Local". Se asume Argentina => UTC-3 fijo
   (sin horario de verano, Argentina no lo usa desde 2009). Si tus datos
   son de otro país o corresponden a un viaje mientras viajabas, ajustá
   UTC_OFFSET_HOURS más abajo.

3. TOLERANCIA DE MATCHEO: los joins por tiempo usan una ventana de
   tolerancia (PAYMENT_MATCH_TOLERANCE). Un pago que caiga fuera de esa
   ventana respecto al viaje más cercano queda como NaN en vez de
   asignarse a un viaje equivocado. Preferí perder datos a ensuciarlos.
=================================================================
"""

from __future__ import annotations

import logging
from typing import Dict

import pandas as pd

logger = logging.getLogger(__name__)

# ---- Supuestos configurables (ver docstring del módulo) --------------------
UTC_OFFSET_HOURS = -3  # Argentina, fijo todo el año
PAYMENT_MATCH_TOLERANCE = pd.Timedelta("30min")  # margen para casar pago <-> viaje (ver diagnose_unmatched.py)
TELEMETRY_EXTRA_MARGIN = pd.Timedelta("2min")     # margen extra sobre el rango del viaje


# =============================================================================
# 1. LIMPIEZA POR DATASET
# =============================================================================

def clean_trips(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia driver_lifetime_trips: castea timestamps, ordena cronológicamente
    y asigna un trip_id sintético (0, 1, 2, ...) basado en el orden de inicio
    del viaje. Este trip_id es la clave que usan todos los joins posteriores.
    """
    df = df.copy()

    timestamp_cols = [
        "request_timestamp_local",
        "begintrip_timestamp_local",
        "dropoff_timestamp_local",
    ]
    for col in timestamp_cols:
        if col in df.columns:
            # IMPORTANTE: estas columnas vienen con sufijo "Z" (ej.
            # "2025-01-14T07:59:40.000Z") a pesar de llamarse "_local".
            # Confirmado contra las columnas *_utc hermanas: el "Z" es un
            # error de etiquetado de Uber, NO significa UTC real -> los
            # dígitos ya son la hora local correcta. Si dejáramos que
            # pd.to_datetime interprete el "Z", nos devuelve un datetime
            # tz-aware en UTC (corriendo el valor), lo que además rompe
            # cualquier merge contra columnas naive como "Local Timestamp"
            # de driver_payments. Por eso se saca el "Z" ANTES de parsear.
            df[col] = pd.to_datetime(
                df[col].astype(str).str.replace("Z$", "", regex=True),
                errors="coerce",
            )

    bool_cols = ["is_completed", "is_cash_trip", "is_airport_trip", "is_scheduled_trip"]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype("boolean")

    # Columnas financieras: castear a numérico. Uber a veces deja estos campos
    # vacíos (no NaN explícito) cuando no aplican (ej: sin surge -> surge_fare
    # vacío), así que rellenamos con 0.0 en vez de dejar NaN, que rompería sumas.
    fare_cols = [
        "driver_upfront_fare_local", "original_fare_local", "base_fare_local",
        "surge_fare_local", "per_mile_fare_local", "per_minute_fare_local",
        "wait_time_fare_local", "minimum_fare_roundup_local", "booking_fee_local",
        "service_fee_local", "toll_amount_local", "cancellation_fee_local",
        "promotion_local", "credits_local", "rounding_down_amount_local",
        "long_distance_surcharge_local",
    ]
    for col in fare_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # Sin begintrip_timestamp_local no hay forma confiable de ordenar ni de
    # casar telemetría/pagos -> se descartan esas filas explícitamente
    # (en vez de dejarlas flotando con NaT y romper los merges silenciosamente).
    filas_antes = len(df)
    df = df.dropna(subset=["begintrip_timestamp_local"])
    if len(df) < filas_antes:
        logger.warning(
            "clean_trips: se descartaron %d filas sin begintrip_timestamp_local válido.",
            filas_antes - len(df),
        )

    df = df.sort_values("begintrip_timestamp_local").reset_index(drop=True)
    df["trip_id"] = df.index

    logger.info("clean_trips: %d viajes procesados.", len(df))
    return df


def clean_payments(df: pd.DataFrame) -> pd.DataFrame:
    """Limpia driver_payments: castea timestamp y tipos numéricos."""
    df = df.copy()
    if "Local Timestamp" in df.columns:
        df["Local Timestamp"] = pd.to_datetime(df["Local Timestamp"], errors="coerce")
    if "Local Amount" in df.columns:
        df["Local Amount"] = pd.to_numeric(df["Local Amount"], errors="coerce")

    df = df.dropna(subset=["Local Timestamp"]).sort_values("Local Timestamp").reset_index(drop=True)
    logger.info("clean_payments: %d pagos procesados.", len(df))
    return df


def clean_telemetry(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia driver_app_analytics: castea timestamp UTC y lo convierte a hora
    local usando UTC_OFFSET_HOURS, para poder cruzarlo con los viajes.
    """
    df = df.copy()
    if "Event Time (UTC)" in df.columns:
        df["Event Time (UTC)"] = pd.to_datetime(df["Event Time (UTC)"], errors="coerce", utc=True)
        df["event_time_local"] = (
            df["Event Time (UTC)"] + pd.Timedelta(hours=UTC_OFFSET_HOURS)
        ).dt.tz_localize(None)

    df = df.dropna(subset=["event_time_local"]).sort_values("event_time_local").reset_index(drop=True)
    logger.info("clean_telemetry: %d registros de telemetría procesados.", len(df))
    return df


# =============================================================================
# 2. JOIN POR PROXIMIDAD TEMPORAL
# =============================================================================

# Uber usa "Category" como agrupación de negocio de sus ~30 códigos de ledger
# internos (Classification). Mapeo a nombres legibles para las columnas finales.
CATEGORY_LABELS = {
    "driver_payment_fares": "tarifa_viaje",       # base + distancia + tiempo + surge + ajustes
    "commission": "comision_uber",
    "commission_adjustment": "ajuste_comision",
    "cash_collected": "efectivo_cobrado",
    "tip": "propina",
    "driver_payment_tolls": "peaje",
    "driver_payment_charges": "cargos_espera",
    "safe_rides_fee": "tarifa_seguridad",
    "rider_fares": "ajuste_pasajero",
    "existing_driver_incentive": "incentivo",
}


def _aggregate_payments_by_uuid(payments: pd.DataFrame) -> pd.DataFrame:
    """
    Consolida las ~11 líneas de ledger por viaje (una por cada componente:
    tarifa base, distancia, tiempo, comisión, etc.) en UNA fila por Trip UUID,
    con los montos pivoteados por Category.

    Esto es clave para que el join contra `trips` sea confiable: en vez de
    matchear 18.000+ líneas sueltas por proximidad temporal (donde varias
    líneas del mismo viaje pueden "flotar" y colarse en el viaje vecino),
    matcheamos ~1.800 transacciones ya consolidadas por un ID exacto.
    """
    class_col = "Category" if "Category" in payments.columns else "Classification"

    # Chequeo de calidad: dentro de un mismo Trip UUID, todas las líneas
    # deberían compartir (aprox) el mismo Local Timestamp, porque son
    # fragmentos de la misma liquidación. Si el rango es grande, algo raro
    # está pasando con ese viaje puntual -> se loguea, no se rompe nada.
    rango_por_uuid = payments.groupby("Trip UUID")["Local Timestamp"].agg(lambda s: s.max() - s.min())
    inconsistentes = rango_por_uuid[rango_por_uuid > pd.Timedelta("5min")]
    if len(inconsistentes):
        logger.warning(
            "_aggregate_payments_by_uuid: %d Trip UUID tienen líneas de pago "
            "dispersas en más de 5 min entre sí (revisar manualmente).",
            len(inconsistentes),
        )

    pivot = payments.pivot_table(
        index="Trip UUID",
        columns=class_col,
        values="Local Amount",
        aggfunc="sum",
        fill_value=0.0,
    )
    pivot.columns = [
        f"monto_{CATEGORY_LABELS.get(str(c), str(c).lower().strip().replace(' ', '_'))}"
        for c in pivot.columns
    ]

    anchor_timestamp = payments.groupby("Trip UUID")["Local Timestamp"].first()
    pivot["payment_timestamp"] = anchor_timestamp
    pivot = pivot.reset_index()

    logger.info(
        "_aggregate_payments_by_uuid: %d líneas de pago consolidadas en %d transacciones (Trip UUID).",
        len(payments), len(pivot),
    )
    return pivot


def merge_payments_to_trips(
    trips: pd.DataFrame,
    payments: pd.DataFrame,
    tolerance: pd.Timedelta = PAYMENT_MATCH_TOLERANCE,
) -> pd.DataFrame:
    """
    Asigna cada viaje su transacción de pago correspondiente.

    Estrategia en 2 pasos:
    1. Consolidar todas las líneas de ledger de driver_payments por Trip UUID
       (exacto, sin ambigüedad) en una fila por transacción.
    2. Matchear esas transacciones consolidadas contra los viajes por
       proximidad temporal (dropoff_timestamp_local más cercano).
    """
    pagos_consolidados = _aggregate_payments_by_uuid(payments)

    # merge_asof no acepta nulos en la clave de unión. Viajes sin
    # dropoff_timestamp_local válido (ej: cancelados, incompletos) no pueden
    # usarse como ancla temporal -> se excluyen SOLO de este merge, pero
    # siguen en la tabla final (el merge con `trips` completo es un left
    # join más abajo).
    trips_con_dropoff = trips.dropna(subset=["dropoff_timestamp_local"])
    excluidos = len(trips) - len(trips_con_dropoff)
    if excluidos:
        logger.warning(
            "merge_payments_to_trips: %d viajes sin dropoff_timestamp_local válido "
            "quedan sin pagos asignados (no se pueden usar como ancla temporal).",
            excluidos,
        )

    trips_sorted = trips_con_dropoff[["trip_id", "dropoff_timestamp_local"]].sort_values("dropoff_timestamp_local")
    pagos_sorted = pagos_consolidados.sort_values("payment_timestamp")

    pagos_con_trip = pd.merge_asof(
        pagos_sorted,
        trips_sorted,
        left_on="payment_timestamp",
        right_on="dropoff_timestamp_local",
        direction="nearest",
        tolerance=tolerance,
    )

    sin_match = pagos_con_trip["trip_id"].isna().sum()
    if sin_match:
        logger.warning(
            "merge_payments_to_trips: %d transacciones de pago no matchearon con "
            "ningún viaje dentro de la tolerancia de %s.",
            sin_match, tolerance,
        )
    pagos_con_trip = pagos_con_trip.dropna(subset=["trip_id"])

    # Ahora que cada fila YA es una transacción única (1 Trip UUID), la única
    # ambigüedad posible es que dos transacciones distintas hayan matcheado
    # al MISMO viaje (viajes muy pegados en el tiempo, ambos "compitiendo"
    # por el mismo vecino más cercano). merge_asof con direction='nearest'
    # NO garantiza una asignación 1 a 1 -> hay que resolverla a mano acá:
    # nos quedamos con la transacción temporalmente más cercana a ese viaje
    # y la otra queda SIN asignar (mejor perder el dato que sumarle a un
    # viaje una ganancia que probablemente pertenece al vecino).
    pagos_con_trip["_diff_tiempo"] = (
        pagos_con_trip["payment_timestamp"] - pagos_con_trip["dropoff_timestamp_local"]
    ).abs()

    conteo_por_trip = pagos_con_trip.groupby("trip_id")["Trip UUID"].nunique()
    trips_ambiguos = conteo_por_trip[conteo_por_trip > 1]
    if len(trips_ambiguos):
        logger.warning(
            "merge_payments_to_trips: %d viajes recibieron más de una transacción "
            "candidata (viajes muy próximos en el tiempo). Se conserva solo la más "
            "cercana por viaje; el resto queda sin asignar: trip_id=%s",
            len(trips_ambiguos), trips_ambiguos.index.tolist()[:20],
        )

    antes = len(pagos_con_trip)
    pagos_con_trip = pagos_con_trip.sort_values("_diff_tiempo").drop_duplicates(
        subset="trip_id", keep="first"
    )
    descartadas_por_duplicado = antes - len(pagos_con_trip)
    if descartadas_por_duplicado:
        logger.warning(
            "merge_payments_to_trips: %d transacciones descartadas por perder la "
            "competencia contra otra más cercana al mismo viaje.",
            descartadas_por_duplicado,
        )

    pagos_con_trip = pagos_con_trip.rename(columns={"Trip UUID": "trip_uuid_uber"})
    pagos_con_trip = pagos_con_trip.drop(columns=["dropoff_timestamp_local", "payment_timestamp", "_diff_tiempo"])

    resultado = trips.merge(pagos_con_trip, on="trip_id", how="left")
    logger.info("merge_payments_to_trips: %d/%d viajes con una transacción de pago asignada.",
                pagos_con_trip["trip_id"].nunique(), len(trips))
    return resultado


def attach_telemetry_to_trips(
    trips: pd.DataFrame,
    telemetry: pd.DataFrame,
    extra_margin: pd.Timedelta = TELEMETRY_EXTRA_MARGIN,
) -> pd.DataFrame:
    """
    Asigna cada punto de telemetría al viaje en curso en ese momento
    (begintrip_timestamp_local <= evento <= dropoff_timestamp_local + margen),
    y agrega por viaje: cantidad de registros y posición promedio.

    Usa merge_asof (direction='backward') para encontrar el último viaje
    que empezó antes del evento, y después filtra los eventos que quedaron
    fuera de la ventana de ese viaje (es decir, tiempo muerto entre viajes).
    """
    trips_validos = trips.dropna(subset=["begintrip_timestamp_local", "dropoff_timestamp_local"])
    excluidos = len(trips) - len(trips_validos)
    if excluidos:
        logger.warning(
            "attach_telemetry_to_trips: %d viajes sin begintrip/dropoff válidos "
            "quedan sin telemetría asignada.",
            excluidos,
        )

    trips_sorted = trips_validos[["trip_id", "begintrip_timestamp_local", "dropoff_timestamp_local"]] \
        .sort_values("begintrip_timestamp_local")
    telemetry_sorted = telemetry.sort_values("event_time_local")

    telemetry_con_trip = pd.merge_asof(
        telemetry_sorted,
        trips_sorted,
        left_on="event_time_local",
        right_on="begintrip_timestamp_local",
        direction="backward",
    )

    dentro_de_ventana = (
        telemetry_con_trip["event_time_local"]
        <= telemetry_con_trip["dropoff_timestamp_local"] + extra_margin
    )
    telemetry_en_viaje = telemetry_con_trip[dentro_de_ventana].dropna(subset=["trip_id"])

    logger.info(
        "attach_telemetry_to_trips: %d/%d registros de telemetría cayeron dentro de un viaje.",
        len(telemetry_en_viaje), len(telemetry_con_trip),
    )

    agg = telemetry_en_viaje.groupby("trip_id").agg(
        telemetry_puntos=("event_time_local", "count"),
        lat_promedio=("Latitude", "mean"),
        lon_promedio=("Longitude", "mean"),
    ).reset_index()

    resultado = trips.merge(agg, on="trip_id", how="left")
    return resultado


# =============================================================================
# 3. ORQUESTADOR
# =============================================================================

def build_master_trips_table(dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Limpia los 3 DataFrames y arma la tabla maestra: un registro por viaje,
    con pagos pivoteados y telemetría agregada.

    Parameters
    ----------
    dfs : dict
        Diccionario devuelto por `ingestion.load_raw_data()`, con claves
        'driver_lifetime_trips', 'driver_payments', 'driver_app_analytics'.
    """
    trips = clean_trips(dfs["driver_lifetime_trips"])

    if "driver_payments" in dfs:
        trips = merge_payments_to_trips(trips, clean_payments(dfs["driver_payments"]))
    else:
        logger.warning("build_master_trips_table: no se encontró driver_payments, se omite el merge.")

    if "driver_app_analytics" in dfs:
        trips = attach_telemetry_to_trips(trips, clean_telemetry(dfs["driver_app_analytics"]))
    else:
        logger.warning("build_master_trips_table: no se encontró driver_app_analytics, se omite el merge.")

    return trips