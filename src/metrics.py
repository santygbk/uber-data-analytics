"""
metrics.py
----------
Módulo de KPIs del pipeline ETL de Uber Driver Analytics.

Toma la tabla maestra de `preprocessing.build_master_trips_table()` y calcula
métricas de rendimiento como conductor.

=================================================================
DECISIÓN CLAVE: cómo se calcula la "ganancia neta" de un viaje
=================================================================
Se suman las columnas `monto_*` que representan plata que efectivamente
te queda a VOS como conductor (tarifa liquidada, comisión de Uber -ya viene
en negativo-, propina, peaje, cargos de espera, incentivos). Se EXCLUYEN
a propósito:
  - monto_tarifa_seguridad: fee que Uber le cobra al pasajero, no es tuyo.
  - monto_ajuste_pasajero: ajuste del lado del pasajero, no tuyo.
  - monto_efectivo_cobrado: OJO, esto NO es un gasto tuyo a pesar de venir
    en negativo. Es un ajuste contable interno: cuando el pasajero te paga
    en efectivo, Uber DESCUENTA ese monto de tu transferencia digital
    (porque ya tenés esa plata en la mano). Sumar este campo directo a la
    ganancia neta resta el efectivo DOS VECES (una porque ya está restado
    de la transferencia digital, y otra porque lo volvíamos a restar acá).
    Confirmado con un caso real: la fórmula vieja daba ganancia negativa en
    una semana donde el conductor había ganado 71.060 ARS reales -> excluir
    este campo reprodujo el número correcto. `monto_tarifa_viaje +
    monto_comision_uber` YA representa la ganancia real del viaje, sea cash
    o tarjeta. Queda disponible como columna informativa aparte (para saber
    cuánto efectivo llevás encima), pero no se suma a la ganancia.

Si un viaje no tiene NINGUNA columna monto_* disponible (no se pudo asignar
una transacción de pago durante el merge en preprocessing.py), la ganancia
neta queda en NaN -> se EXCLUYE de los promedios en vez de contar como $0,
para no sesgar las métricas hacia abajo. Ver `reporte_cobertura()` para
saber qué % de viajes tiene este problema.

=================================================================
OJO: per_mile_fare_local y per_minute_fare_local son TARIFAS (rate),
no montos ya multiplicados por distancia/duración. Confirmado con datos
reales: el mismo valor se repite en viajes de distinta distancia/duración.
Por eso acá se calculan `ingreso_por_distancia_calc` e
`ingreso_por_tiempo_calc` multiplicando explícitamente.
=================================================================
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Columnas de driver_payments que representan plata real para el conductor
# (ver decisión documentada arriba). monto_efectivo_cobrado NO va acá.
DRIVER_EARNING_COLS = [
    "monto_tarifa_viaje",
    "monto_comision_uber",       # ya viene negativo
    "monto_ajuste_comision",
    "monto_propina",
    "monto_peaje",
    "monto_cargos_espera",
    "monto_incentivo",
]

DIAS_ORDEN = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
DIAS_ES = {
    "Monday": "Lunes", "Tuesday": "Martes", "Wednesday": "Miércoles",
    "Thursday": "Jueves", "Friday": "Viernes", "Saturday": "Sábado", "Sunday": "Domingo",
}


# =============================================================================
# 1. ENRIQUECIMIENTO: agrega columnas calculadas a nivel de viaje
# =============================================================================

def compute_trip_financials(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega al DataFrame de viajes las columnas calculadas necesarias para
    todos los KPIs de este módulo. No modifica el original.

    Columnas nuevas:
    - duracion_horas, duracion_minutos
    - ingreso_por_distancia_calc, ingreso_por_tiempo_calc (tarifa x cantidad real)
    - ganancia_neta_local (suma de DRIVER_EARNING_COLS disponibles; NaN si
      ninguna está disponible)
    - ganancia_por_hora, ganancia_por_milla (NaN si falta el denominador o
      la ganancia)
    - dia_semana (en español), hora_del_dia (0-23), franja_horaria
    - tiene_datos_pago (bool, para filtrar/reportar cobertura)
    """
    df = df.copy()

    df["duracion_horas"] = df["trip_duration_seconds"] / 3600
    df["duracion_minutos"] = df["trip_duration_seconds"] / 60

    if "per_mile_fare_local" in df.columns:
        df["ingreso_por_distancia_calc"] = df["per_mile_fare_local"] * df["trip_distance_miles"]
    if "per_minute_fare_local" in df.columns:
        df["ingreso_por_tiempo_calc"] = df["per_minute_fare_local"] * df["duracion_minutos"]

    cols_disponibles = [c for c in DRIVER_EARNING_COLS if c in df.columns]
    if not cols_disponibles:
        raise ValueError(
            "Ninguna columna de ganancia (DRIVER_EARNING_COLS) está presente. "
            "¿Corriste merge_payments_to_trips antes de esto?"
        )
    df["tiene_datos_pago"] = df[cols_disponibles].notna().any(axis=1)
    # min_count=1 -> si TODAS las columnas son NaN en una fila, el resultado
    # es NaN (no 0). Si hay al menos una, las NaN individuales se tratan
    # como 0 dentro de la suma (así quedaron desde el pivot en preprocessing).
    df["ganancia_neta_local"] = df[cols_disponibles].sum(axis=1, min_count=1)

    # Informativo, NO se suma a la ganancia (ver docstring del módulo):
    # cuánto efectivo tuviste que cobrar/llevar encima en el viaje.
    if "monto_efectivo_cobrado" in df.columns:
        df["efectivo_en_mano"] = df["monto_efectivo_cobrado"].abs()

    duracion_valida = df["duracion_horas"] > 0
    df["ganancia_por_hora"] = np.where(
        duracion_valida, df["ganancia_neta_local"] / df["duracion_horas"], np.nan
    )
    distancia_valida = df["trip_distance_miles"] > 0
    df["ganancia_por_milla"] = np.where(
        distancia_valida, df["ganancia_neta_local"] / df["trip_distance_miles"], np.nan
    )

    df["dia_semana_en"] = df["begintrip_timestamp_local"].dt.day_name()
    df["dia_semana"] = df["dia_semana_en"].map(DIAS_ES)
    df["hora_del_dia"] = df["begintrip_timestamp_local"].dt.hour

    bins = [-1, 5, 11, 17, 21, 24]
    labels = ["Madrugada (0-5h)", "Mañana (6-11h)", "Tarde (12-17h)", "Noche (18-21h)", "Noche tardía (22-23h)"]
    df["franja_horaria"] = pd.cut(df["hora_del_dia"], bins=bins, labels=labels)

    n_sin_pago = (~df["tiene_datos_pago"]).sum()
    logger.info(
        "compute_trip_financials: %d/%d viajes con datos de pago disponibles (%.1f%%).",
        df["tiene_datos_pago"].sum(), len(df), 100 * df["tiene_datos_pago"].mean(),
    )
    return df


# =============================================================================
# 2. REPORTE DE COBERTURA (transparencia sobre qué % de datos falta)
# =============================================================================

def reporte_cobertura(df: pd.DataFrame) -> dict:
    """
    Resume qué fracción de los viajes tiene cada tipo de dato disponible.
    Fundamental para no reportar promedios engañosos calculados solo sobre
    el subconjunto de viajes con datos completos.
    """
    total = len(df)
    return {
        "total_viajes": total,
        "con_datos_pago": int(df["tiene_datos_pago"].sum()),
        "pct_con_datos_pago": round(100 * df["tiene_datos_pago"].mean(), 1),
        "con_telemetria": int(df["telemetry_puntos"].notna().sum()) if "telemetry_puntos" in df.columns else None,
        "pct_con_telemetria": round(100 * df["telemetry_puntos"].notna().mean(), 1) if "telemetry_puntos" in df.columns else None,
    }


# =============================================================================
# 3. RESUMEN GENERAL
# =============================================================================

def resumen_general(df: pd.DataFrame) -> dict:
    """
    KPIs agregados de todo el período cubierto por la tabla maestra.

    IMPORTANTE: la ganancia por hora se calcula como
    SUM(ganancia_neta_local) / SUM(duracion_horas), NO como el promedio de
    la tasa $/hora de cada viaje individual. Promediar tasas por viaje
    sobreestima sistemáticamente: un viaje de 5 minutos con tarifa mínima
    da una tasa proyectada absurda si se lo estira a una hora completa
    (ej: $4.395 en 5 min -> "52.217 $/h"), y esos outliers dominan un
    promedio simple aunque representen poquísimo tiempo real trabajado.
    Confirmado con un caso real: el promedio simple daba +15% más alto que
    el total/total para la misma semana.
    """
    con_pago = df[df["tiene_datos_pago"]]
    ganancia_total = con_pago["ganancia_neta_local"].sum()
    horas_activas_total = con_pago["duracion_horas"].sum()

    return {
        "periodo_desde": df["begintrip_timestamp_local"].min(),
        "periodo_hasta": df["dropoff_timestamp_local"].max(),
        "total_viajes": len(df),
        "total_viajes_con_datos_pago": len(con_pago),
        "ganancia_neta_total_local": round(ganancia_total, 2),
        "ganancia_total_periodo": round(ganancia_total, 2),
        "ganancia_promedio_por_viaje": round(con_pago["ganancia_neta_local"].mean(), 2),
        "ganancia_mediana_por_viaje": round(con_pago["ganancia_neta_local"].median(), 2),
        "ganancia_por_hora_activa": round(ganancia_total / horas_activas_total, 2) if horas_activas_total else None,
        "horas_activas_manejando": round(horas_activas_total, 1),
        "distancia_total_millas": round(df["trip_distance_miles"].sum(), 1),
        "kilometros_totales": round(df["trip_distance_miles"].sum() * 1.60934, 1),
        "propina_total": round(con_pago["monto_propina"].sum(), 2) if "monto_propina" in df.columns else None,
        "propina_promedio_por_viaje": round(con_pago["monto_propina"].mean(), 2) if "monto_propina" in df.columns else None,
        "comision_uber_total": round(con_pago["monto_comision_uber"].sum(), 2) if "monto_comision_uber" in df.columns else None,
        "pct_viajes_cash": round(100 * df["is_cash_trip"].mean(), 1) if "is_cash_trip" in df.columns else None,
        "pct_viajes_aeropuerto": round(100 * df["is_airport_trip"].mean(), 1) if "is_airport_trip" in df.columns else None,
        "surge_promedio": round(df["surge_multiplier"].mean(), 2) if "surge_multiplier" in df.columns else None,
        "pct_viajes_con_surge": round(100 * (df["surge_multiplier"] > 1.0).mean(), 1) if "surge_multiplier" in df.columns else None,
    }


# =============================================================================
# 3.5. SESIONES DE TRABAJO (incluye tiempo de espera entre viajes)
# =============================================================================

def construir_sesiones_trabajo(df: pd.DataFrame, gap_maximo_minutos: int = 60) -> pd.DataFrame:
    """
    `trip_duration_seconds` solo cuenta el tiempo CON el pasajero arriba
    (begintrip -> dropoff). NO incluye el tiempo esperando que te asignen
    un viaje, ni el traslado hasta el punto de encuentro. Por eso dividir
    ganancia por `duracion_horas` (basada en trip_duration_seconds)
    SUBESTIMA las horas reales trabajadas y sobreestima el $/hora.

    Esta función agrupa viajes consecutivos en "sesiones de trabajo": si el
    tiempo entre el dropoff de un viaje y el request/begintrip del
    siguiente es menor a `gap_maximo_minutos`, se consideran parte de la
    misma sesión (mismo "turno" manejando). La duración de la sesión se
    mide de punta a punta (primer request -> último dropoff), lo que SÍ
    incluye la espera entre viajes dentro del turno.

    Confirmado con un caso real: usando trip_duration_seconds la ganancia
    por hora daba muy por encima de lo que el conductor sabe que gana;
    usando la duración de sesión (con espera incluida) el número coincidió
    con su cálculo manual (ganancia total / tiempo total del turno).
    """
    con_pago = df[df["tiene_datos_pago"]].copy()
    ancla_inicio = con_pago["request_timestamp_local"].fillna(con_pago["begintrip_timestamp_local"])
    con_pago = con_pago.assign(_ancla_inicio=ancla_inicio).sort_values("_ancla_inicio")

    gap = con_pago["_ancla_inicio"] - con_pago["dropoff_timestamp_local"].shift(1)
    nueva_sesion = gap.isna() | (gap > pd.Timedelta(minutes=gap_maximo_minutos))
    con_pago["sesion_id"] = nueva_sesion.cumsum()

    sesiones = con_pago.groupby("sesion_id").agg(
        sesion_inicio=("_ancla_inicio", "min"),
        sesion_fin=("dropoff_timestamp_local", "max"),
        ganancia_sesion=("ganancia_neta_local", "sum"),
        n_viajes=("ganancia_neta_local", "count"),
    ).reset_index()

    sesiones["duracion_sesion_horas"] = (
        (sesiones["sesion_fin"] - sesiones["sesion_inicio"]).dt.total_seconds() / 3600
    )
    sesiones["ganancia_por_hora_trabajada"] = (
        sesiones["ganancia_sesion"] / sesiones["duracion_sesion_horas"]
    ).round(2)
    # Se etiqueta cada sesión por el día/hora de INICIO (una sesión larga
    # que cruza la medianoche o varias horas queda asignada a cuándo empezó
    # el turno, no se reparte proporcionalmente entre días/horas).
    sesiones["dia_semana_en"] = sesiones["sesion_inicio"].dt.day_name()
    sesiones["dia_semana"] = sesiones["dia_semana_en"].map(DIAS_ES)
    sesiones["hora_inicio"] = sesiones["sesion_inicio"].dt.hour

    logger.info(
        "construir_sesiones_trabajo: %d viajes agrupados en %d sesiones (gap máximo %d min).",
        len(con_pago), len(sesiones), gap_maximo_minutos,
    )
    return sesiones


def resumen_sesiones(df: pd.DataFrame, gap_maximo_minutos: int = 60) -> dict:
    """
    Resumen agregado a nivel de sesión: esta es la ganancia_por_hora que
    corresponde comparar contra "cuánto gano por hora de laburo real"
    (incluye espera). Usar esta, no `ganancia_por_hora_activa` de
    `resumen_general()`, para responder "¿cuánto gano por hora trabajada?".
    """
    sesiones = construir_sesiones_trabajo(df, gap_maximo_minutos)
    ganancia_total = sesiones["ganancia_sesion"].sum()
    horas_total = sesiones["duracion_sesion_horas"].sum()
    return {
        "n_sesiones": len(sesiones),
        "horas_trabajadas_totales": round(horas_total, 1),
        "ganancia_total": round(ganancia_total, 2),
        "ganancia_por_hora_trabajada": round(ganancia_total / horas_total, 2) if horas_total else None,
        "ganancia_por_hora_trabajada_mediana_sesion": round(sesiones["ganancia_por_hora_trabajada"].median(), 2),
        "viajes_por_sesion_promedio": round(sesiones["n_viajes"].mean(), 1),
        "duracion_sesion_promedio_horas": round(sesiones["duracion_sesion_horas"].mean(), 2),
    }


def ganancia_por_dia_semana_conectado(df: pd.DataFrame, gap_maximo_minutos: int = 60) -> pd.DataFrame:
    """
    ★ La versión "por hora CONECTADA" de ganancia_por_dia_semana ★

    En vez de usar trip_duration_seconds (solo tiempo con pasajero arriba),
    agrupa por sesión de trabajo (turno completo, con espera incluida) y
    después por día de la semana en que arrancó cada turno. Esto responde
    "¿cuánto gano por hora de estar conectado/trabajando ese día?", no
    "por hora de viaje activo".
    """
    sesiones = construir_sesiones_trabajo(df, gap_maximo_minutos)
    resultado = sesiones.groupby("dia_semana_en").agg(
        ganancia_total=("ganancia_sesion", "sum"),
        horas_total=("duracion_sesion_horas", "sum"),
        n_sesiones=("ganancia_sesion", "count"),
        n_viajes=("n_viajes", "sum"),
    )
    resultado = resultado.reindex(DIAS_ORDEN)
    resultado["ganancia_hora_conectada"] = (resultado["ganancia_total"] / resultado["horas_total"]).round(2)
    resultado = resultado.reset_index()
    resultado["dia_semana"] = resultado["dia_semana_en"].map(DIAS_ES)
    return resultado[["dia_semana", "ganancia_hora_conectada", "ganancia_total", "horas_total", "n_sesiones", "n_viajes"]]


def ganancia_por_franja_horaria_conectado(df: pd.DataFrame, gap_maximo_minutos: int = 60) -> pd.DataFrame:
    """Igual que arriba pero por franja horaria de INICIO del turno."""
    sesiones = construir_sesiones_trabajo(df, gap_maximo_minutos)
    bins = [-1, 5, 11, 17, 21, 24]
    labels = ["Madrugada (0-5h)", "Mañana (6-11h)", "Tarde (12-17h)", "Noche (18-21h)", "Noche tardía (22-23h)"]
    sesiones["franja_horaria"] = pd.cut(sesiones["hora_inicio"], bins=bins, labels=labels)

    resultado = sesiones.groupby("franja_horaria", observed=True).agg(
        ganancia_total=("ganancia_sesion", "sum"),
        horas_total=("duracion_sesion_horas", "sum"),
        n_sesiones=("ganancia_sesion", "count"),
    ).reset_index()
    resultado["ganancia_hora_conectada"] = (resultado["ganancia_total"] / resultado["horas_total"]).round(2)
    return resultado[["franja_horaria", "ganancia_hora_conectada", "horas_total", "n_sesiones"]]



# =============================================================================
# 4. GANANCIA POR HORA DEL DÍA / DÍA DE LA SEMANA
# =============================================================================

def ganancia_por_hora_del_dia(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrupa por hora del día (0-23). `ganancia_hora_ponderada` = suma de
    ganancia / suma de horas activas del grupo (correcto). `ganancia_hora_mediana`
    queda como referencia del viaje "típico" en esa hora (median resiste
    outliers mejor que un promedio simple de tasas, pero sigue siendo
    una tasa por viaje individual, no la ponderada del grupo).
    """
    con_pago = df[df["tiene_datos_pago"]]
    resultado = con_pago.groupby("hora_del_dia").agg(
        ganancia_total=("ganancia_neta_local", "sum"),
        horas_total=("duracion_horas", "sum"),
        ganancia_hora_mediana=("ganancia_por_hora", "median"),
        n_viajes=("ganancia_por_hora", "count"),
    ).reset_index().sort_values("hora_del_dia")
    resultado["ganancia_hora_ponderada"] = (resultado["ganancia_total"] / resultado["horas_total"]).round(2)
    resultado["ganancia_hora_mediana"] = resultado["ganancia_hora_mediana"].round(2)
    return resultado[["hora_del_dia", "ganancia_hora_ponderada", "ganancia_hora_mediana", "n_viajes", "horas_total"]]


def ganancia_por_dia_semana(df: pd.DataFrame) -> pd.DataFrame:
    """Agrupa por día de la semana -> "¿qué día gano más?", con tasa ponderada."""
    con_pago = df[df["tiene_datos_pago"]]
    resultado = con_pago.groupby("dia_semana_en").agg(
        ganancia_total=("ganancia_neta_local", "sum"),
        horas_total=("duracion_horas", "sum"),
        ganancia_promedio_viaje=("ganancia_neta_local", "mean"),
        n_viajes=("ganancia_neta_local", "count"),
    )
    resultado = resultado.reindex(DIAS_ORDEN)
    resultado["ganancia_hora_ponderada"] = (resultado["ganancia_total"] / resultado["horas_total"]).round(2)
    resultado["ganancia_promedio_viaje"] = resultado["ganancia_promedio_viaje"].round(2)
    resultado = resultado.reset_index()
    resultado["dia_semana"] = resultado["dia_semana_en"].map(DIAS_ES)
    return resultado[["dia_semana", "ganancia_hora_ponderada", "ganancia_promedio_viaje", "ganancia_total", "horas_total", "n_viajes"]]


def tabla_referencia_dia_hora(df: pd.DataFrame, min_muestras: int = 3) -> pd.DataFrame:
    """
    ★ LA TABLA CLAVE PARA LA FUTURA FÓRMULA DE DECISIÓN EN TIEMPO REAL ★

    Agrupa por (día de la semana, hora del día). `ganancia_hora_ponderada`
    (suma de ganancia / suma de horas activas del grupo) es el número que
    conviene usar como baseline -> no está inflado por viajes cortos con
    tasa proyectada absurda, a diferencia de promediar tasas individuales.

    Columnas devueltas:
    - dia_semana, hora_del_dia
    - ganancia_hora_ponderada: la tasa $/hora "real" del grupo (usar esta).
    - ganancia_hora_mediana: tasa $/hora del viaje típico individual (para
      contrastar contra la ponderada; si difieren mucho, hay heterogeneidad
      de viajes cortos/largos en esa celda).
    - ganancia_milla_ponderada: análogo por milla.
    - n_viajes: tamaño de muestra -> úsalo para ponderar confianza.
    - dato_confiable: bool, True si n_viajes >= min_muestras
    """
    con_pago = df[df["tiene_datos_pago"]]
    resultado = con_pago.groupby(["dia_semana_en", "hora_del_dia"]).agg(
        ganancia_total=("ganancia_neta_local", "sum"),
        horas_total=("duracion_horas", "sum"),
        millas_total=("trip_distance_miles", "sum"),
        ganancia_hora_mediana=("ganancia_por_hora", "median"),
        n_viajes=("ganancia_por_hora", "count"),
    ).reset_index()

    resultado["ganancia_hora_ponderada"] = (resultado["ganancia_total"] / resultado["horas_total"]).round(2)
    resultado["ganancia_milla_ponderada"] = (resultado["ganancia_total"] / resultado["millas_total"]).round(2)
    resultado["ganancia_hora_mediana"] = resultado["ganancia_hora_mediana"].round(2)
    resultado["dia_semana"] = resultado["dia_semana_en"].map(DIAS_ES)
    resultado["dato_confiable"] = resultado["n_viajes"] >= min_muestras
    resultado["dia_semana_en"] = pd.Categorical(resultado["dia_semana_en"], categories=DIAS_ORDEN, ordered=True)
    resultado = resultado.sort_values(["dia_semana_en", "hora_del_dia"]).reset_index(drop=True)
    return resultado[[
        "dia_semana", "hora_del_dia", "ganancia_hora_ponderada", "ganancia_hora_mediana",
        "ganancia_milla_ponderada", "n_viajes", "dato_confiable",
    ]]


def heatmap_dia_hora(df: pd.DataFrame, metrica: str = "ganancia_hora_ponderada") -> pd.DataFrame:
    """
    Devuelve una tabla pivoteada (día x hora) lista para graficar como
    heatmap, calculada a partir de `tabla_referencia_dia_hora` (tasa
    ponderada, no promedio simple de tasas individuales).

    `metrica`: 'ganancia_hora_ponderada' (default) o 'ganancia_milla_ponderada'.
    """
    tabla = tabla_referencia_dia_hora(df)
    pivot = tabla.pivot_table(index="dia_semana", columns="hora_del_dia", values=metrica)
    pivot = pivot.reindex(DIAS_ES.values())
    return pivot


# =============================================================================
# 5. GANANCIA POR TIPO DE VIAJE
# =============================================================================

def ganancia_por_tipo_viaje(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compara ganancia_por_hora y ganancia_por_milla según distintas
    categorías binarias del viaje: aeropuerto, cash vs tarjeta, con/sin
    surge, franja horaria.
    """
    con_pago = df[df["tiene_datos_pago"]].copy()
    con_pago["tiene_surge"] = con_pago["surge_multiplier"] > 1.0

    filas = []
    for columna, etiquetas in [
        ("is_airport_trip", {True: "Aeropuerto", False: "No aeropuerto"}),
        ("is_cash_trip", {True: "Efectivo", False: "Tarjeta/App"}),
        ("tiene_surge", {True: "Con surge", False: "Sin surge"}),
        ("is_scheduled_trip", {True: "Programado", False: "No programado"}),
    ]:
        if columna not in con_pago.columns:
            continue
        grp = con_pago.groupby(columna).agg(
            ganancia_total=("ganancia_neta_local", "sum"),
            horas_total=("duracion_horas", "sum"),
            millas_total=("trip_distance_miles", "sum"),
            ganancia_promedio_viaje=("ganancia_neta_local", "mean"),
            n_viajes=("ganancia_neta_local", "count"),
        )
        grp["ganancia_hora_ponderada"] = (grp["ganancia_total"] / grp["horas_total"]).round(2)
        grp["ganancia_milla_ponderada"] = (grp["ganancia_total"] / grp["millas_total"]).round(2)
        grp["ganancia_promedio_viaje"] = grp["ganancia_promedio_viaje"].round(2)
        grp = grp.drop(columns=["ganancia_total", "horas_total", "millas_total"])
        for valor, fila in grp.iterrows():
            filas.append({
                "categoria": columna,
                "grupo": etiquetas.get(valor, str(valor)),
                **fila.to_dict(),
            })

    return pd.DataFrame(filas)


def ganancia_por_franja_horaria(df: pd.DataFrame) -> pd.DataFrame:
    """Agrupa por franja horaria (Madrugada/Mañana/Tarde/Noche/Noche tardía)."""
    con_pago = df[df["tiene_datos_pago"]]
    resultado = con_pago.groupby("franja_horaria", observed=True).agg(
        ganancia_total=("ganancia_neta_local", "sum"),
        horas_total=("duracion_horas", "sum"),
        ganancia_promedio_viaje=("ganancia_neta_local", "mean"),
        n_viajes=("ganancia_neta_local", "count"),
    ).reset_index()
    resultado["ganancia_hora_ponderada"] = (resultado["ganancia_total"] / resultado["horas_total"]).round(2)
    resultado["ganancia_promedio_viaje"] = resultado["ganancia_promedio_viaje"].round(2)
    return resultado[["franja_horaria", "ganancia_hora_ponderada", "ganancia_promedio_viaje", "n_viajes", "horas_total"]]


# =============================================================================
# 6. RELACIÓN SURGE <-> GANANCIA (para intuir cuánto "vale" el surge)
# =============================================================================

def impacto_surge(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrupa por multiplicador de surge redondeado (1.0, 1.1, 1.2, ...) para
    ver cómo escala la ganancia_por_hora real con el surge nominal. Útil
    para la fórmula futura: cuantifica si "surge 1.5x" realmente se traduce
    en ~50% más de ganancia por hora, o menos (por ejemplo, si esos viajes
    también tienden a ser más cortos).
    """
    con_pago = df[df["tiene_datos_pago"]].copy()
    con_pago["surge_redondeado"] = con_pago["surge_multiplier"].round(1)
    resultado = con_pago.groupby("surge_redondeado").agg(
        ganancia_total=("ganancia_neta_local", "sum"),
        horas_total=("duracion_horas", "sum"),
        ganancia_promedio_viaje=("ganancia_neta_local", "mean"),
        duracion_promedio_min=("duracion_minutos", "mean"),
        n_viajes=("ganancia_neta_local", "count"),
    ).reset_index()
    resultado["ganancia_hora_ponderada"] = (resultado["ganancia_total"] / resultado["horas_total"]).round(2)
    resultado["ganancia_promedio_viaje"] = resultado["ganancia_promedio_viaje"].round(2)
    resultado["duracion_promedio_min"] = resultado["duracion_promedio_min"].round(2)
    return resultado[["surge_redondeado", "ganancia_hora_ponderada", "ganancia_promedio_viaje", "duracion_promedio_min", "n_viajes"]]


# =============================================================================
# 7. ORQUESTADOR: corre todo y devuelve un diccionario de resultados
# =============================================================================

def compute_all_metrics(df: pd.DataFrame) -> dict:
    """
    Punto de entrada único: recibe la tabla maestra cruda (output de
    `preprocessing.build_master_trips_table`) y devuelve un diccionario con
    todos los reportes/tablas de este módulo, listos para exportar a
    Power BI/Tableau (cada valor es un dict o un DataFrame) o para
    imprimir/graficar directo en Python.
    """
    df = compute_trip_financials(df)
    return {
        "cobertura": reporte_cobertura(df),
        "resumen_general": resumen_general(df),
        "resumen_sesiones": resumen_sesiones(df),
        "sesiones_trabajo": construir_sesiones_trabajo(df),
        "ganancia_por_dia_semana_conectado": ganancia_por_dia_semana_conectado(df),
        "ganancia_por_franja_horaria_conectado": ganancia_por_franja_horaria_conectado(df),
        "ganancia_por_hora_del_dia": ganancia_por_hora_del_dia(df),
        "ganancia_por_dia_semana": ganancia_por_dia_semana(df),
        "tabla_referencia_dia_hora": tabla_referencia_dia_hora(df),
        "heatmap_ganancia_hora": heatmap_dia_hora(df, "ganancia_hora_ponderada"),
        "heatmap_ganancia_milla": heatmap_dia_hora(df, "ganancia_milla_ponderada"),
        "ganancia_por_tipo_viaje": ganancia_por_tipo_viaje(df),
        "ganancia_por_franja_horaria": ganancia_por_franja_horaria(df),
        "impacto_surge": impacto_surge(df),
        "trips_enriquecido": df,  # la tabla completa con todas las columnas calculadas
    }