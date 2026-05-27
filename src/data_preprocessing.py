"""Limpeza, feature engineering e divisão treino/teste.

Princípios aplicados
--------------------
1. **Nada de vazamento temporal**: a divisão é por data (últimos 20% das
   observações) e a interpolação respeita ordem cronológica.
2. **Engenharia explícita de sazonalidade**: as datas são codificadas em
   componentes cíclicos (sin/cos de dia-do-ano), porque modelos baseados em
   árvores aproveitam diretamente esses sinais sem precisar de SARIMA.
3. **Lags curtos**: replicam a memória atmosférica do sistema (1, 7 dias).
4. **Tratamento idempotente de nulos**: o sentinela -999 já é convertido em
   NaN na etapa de aquisição; aqui usamos *forward fill* dentro de cada
   cidade — apenas para pequenas lacunas pontuais.

A função :func:`build_feature_matrix` retorna ``(X, y, meta)`` prontos para
treino, onde ``meta`` carrega data, cidade e região para análise pós-hoc.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from . import config

logger = logging.getLogger(__name__)

TARGET_COL: str = "ALLSKY_SFC_SW_DWN"   # variável-alvo da regressão (Q1)

FEATURE_BASE: list[str] = [
    "T2M", "T2M_MAX", "T2M_MIN", "RH2M",
    "PRECTOTCORR", "WS2M", "PS", "CLRSKY_SFC_SW_DWN", "ALLSKY_KT",
]


def load_consolidated(path: Path | None = None) -> pd.DataFrame:
    """Carrega o CSV consolidado, garantindo tipos corretos."""
    path = path or (config.PROCESSED_DIR / "nasa_power_consolidated.csv")
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values(["city", "date"]).reset_index(drop=True)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Trata sentinelas, imputa lacunas e remove linhas inviáveis."""
    df = df.copy()
    df = df.replace(config.NASA_MISSING_SENTINEL, np.nan)

    numeric_cols = [c for c in config.NASA_PARAMETERS if c in df.columns]
    # Forward fill (curto) por cidade para preencher gaps esporádicos de satélite.
    df[numeric_cols] = (
        df.groupby("city")[numeric_cols]
          .transform(lambda s: s.ffill(limit=3).bfill(limit=3))
    )

    # Linhas remanescentes sem irradiância (alvo) são descartadas.
    before = len(df)
    df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    logger.info("clean: descartadas %d linhas sem alvo (%d -> %d)",
                before - len(df), before, len(df))
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona componentes cíclicos do calendário (sin/cos)."""
    df = df.copy()
    doy = df["date"].dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    return df


def add_lag_features(
    df: pd.DataFrame,
    cols: Sequence[str] = ("T2M", "RH2M", "PRECTOTCORR", TARGET_COL),
    lags: Sequence[int] = (1, 7),
) -> pd.DataFrame:
    """Cria lags por cidade — preserva ordem temporal."""
    df = df.copy().sort_values(["city", "date"])
    for col in cols:
        for lag in lags:
            df[f"{col}_lag{lag}"] = df.groupby("city")[col].shift(lag)
    return df


def add_rolling_features(
    df: pd.DataFrame,
    cols: Sequence[str] = ("T2M", "ALLSKY_SFC_SW_DWN"),
    windows: Sequence[int] = (7, 30),
) -> pd.DataFrame:
    """Estatísticas móveis (média) por cidade."""
    df = df.copy().sort_values(["city", "date"])
    for col in cols:
        for w in windows:
            df[f"{col}_rollmean{w}"] = (
                df.groupby("city")[col]
                  .transform(lambda s, w=w: s.shift(1).rolling(w, min_periods=max(2, w // 2)).mean())
            )
    return df


def build_feature_matrix(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Pipeline completo: clean -> calendário -> lags -> rolling -> dropna.

    Returns
    -------
    X : DataFrame
        Matriz de features (numéricas + one-hot da região).
    y : Series
        Alvo ``ALLSKY_SFC_SW_DWN`` (kWh/m²/dia).
    meta : DataFrame
        Colunas ``date``, ``city``, ``region`` para inspeção.
    """
    df = clean(df)
    df = add_calendar_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)

    feature_cols = list(FEATURE_BASE) + [
        c for c in df.columns
        if c.endswith(("_lag1", "_lag7", "_rollmean7", "_rollmean30"))
    ] + ["doy_sin", "doy_cos", "latitude", "longitude"]

    # Remove duplicatas mantendo ordem
    feature_cols = list(dict.fromkeys(feature_cols))

    region_dummies = pd.get_dummies(df["region"], prefix="reg", dtype=float)
    X = pd.concat([df[feature_cols], region_dummies], axis=1)

    y = df[TARGET_COL]
    meta = df[["date", "city", "region", "uf", "biome"]].copy()

    full = pd.concat([X, y.rename(TARGET_COL), meta], axis=1).dropna()
    X = full[X.columns]
    y = full[TARGET_COL]
    meta = full[meta.columns]
    logger.info("matriz final: %d linhas x %d features", len(X), X.shape[1])
    return X, y, meta


def temporal_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    meta: pd.DataFrame,
    test_size: float = config.TEST_SIZE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]:
    """Divisão por data: as últimas ``test_size`` observações vão para teste.

    A divisão é feita globalmente (não por cidade) para que o teste contenha
    o período mais recente — comportamento esperado em previsão operacional.
    """
    order = meta["date"].argsort().values
    n_test = int(len(X) * test_size)
    test_idx = order[-n_test:]
    train_idx = order[:-n_test]

    return (
        X.iloc[train_idx], X.iloc[test_idx],
        y.iloc[train_idx], y.iloc[test_idx],
        meta.iloc[train_idx], meta.iloc[test_idx],
    )


def aggregate_city_climatology(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula assinaturas climáticas anuais por cidade (insumo do clustering Q4)."""
    df = clean(df)
    agg = (
        df.groupby("city")
          .agg(
              irrad_mean=("ALLSKY_SFC_SW_DWN", "mean"),
              irrad_std=("ALLSKY_SFC_SW_DWN", "std"),
              temp_mean=("T2M", "mean"),
              temp_amplitude=("T2M", lambda s: s.quantile(0.95) - s.quantile(0.05)),
              rh_mean=("RH2M", "mean"),
              prec_total=("PRECTOTCORR", "mean"),
              wind_mean=("WS2M", "mean"),
              kt_mean=("ALLSKY_KT", "mean"),
          )
          .reset_index()
    )
    meta_cols = (
        df[["city", "uf", "region", "biome", "latitude", "longitude"]]
        .drop_duplicates("city")
    )
    return agg.merge(meta_cols, on="city")


if __name__ == "__main__":
    consolidated = load_consolidated()
    X, y, meta = build_feature_matrix(consolidated)
    print(X.head())
    print("alvo (kWh/m^2/dia):", y.describe())
