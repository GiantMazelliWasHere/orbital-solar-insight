"""Análise exploratória de dados (EDA) com geração de figuras.

Cada função recebe o DataFrame consolidado e salva um PNG em
``outputs/figures``. Todas retornam o ``Path`` da figura para facilitar
documentação e uso em notebooks.

A escolha das visualizações segue o princípio de respondê-las antes da
modelagem: existem distribuições assimétricas? há sazonalidade clara? há
diferença entre regiões? quais variáveis correlacionam com irradiância?
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from . import config

logger = logging.getLogger(__name__)


def _setup_style() -> None:
    plt.style.use(config.PLT_STYLE)
    sns.set_palette(config.PALETTE)


def plot_irradiance_distribution(df: pd.DataFrame) -> Path:
    """Distribuição da irradiância por região (violin + boxplot)."""
    _setup_style()
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.violinplot(
        data=df, x="region", y="ALLSKY_SFC_SW_DWN",
        order=["Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul"],
        inner="quartile", ax=ax, cut=0,
    )
    ax.set_title("Distribuição da irradiância solar diária por região (2015-2024)")
    ax.set_ylabel("Irradiância global (kWh/m²/dia)")
    ax.set_xlabel("Região")
    fig.tight_layout()
    out = config.FIGURES_DIR / "01_irradiance_by_region.png"
    fig.savefig(out, dpi=config.FIG_DPI)
    plt.close(fig)
    return out


def plot_monthly_seasonality(df: pd.DataFrame) -> Path:
    """Heatmap mês × cidade da irradiância média (assinatura sazonal)."""
    _setup_style()
    pivot = (
        df.assign(month=df["date"].dt.month)
          .groupby(["city", "month"])["ALLSKY_SFC_SW_DWN"]
          .mean()
          .unstack("month")
    )

    fig, ax = plt.subplots(figsize=(11, 6))
    sns.heatmap(
        pivot, cmap="YlOrRd", annot=True, fmt=".1f",
        cbar_kws={"label": "kWh/m²/dia"}, ax=ax,
    )
    ax.set_title("Sazonalidade da irradiância — média mensal por capital")
    ax.set_xlabel("Mês")
    ax.set_ylabel("Cidade")
    fig.tight_layout()
    out = config.FIGURES_DIR / "02_seasonality_heatmap.png"
    fig.savefig(out, dpi=config.FIG_DPI)
    plt.close(fig)
    return out


def plot_correlation_matrix(df: pd.DataFrame) -> Path:
    """Matriz de correlação Pearson entre variáveis NASA POWER."""
    _setup_style()
    cols = [c for c in config.NASA_PARAMETERS if c in df.columns]
    corr = df[cols].corr(method="pearson")

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, cmap="RdBu_r", center=0, annot=True, fmt=".2f",
                square=True, ax=ax, vmin=-1, vmax=1)
    ax.set_title("Correlação entre parâmetros NASA POWER")
    fig.tight_layout()
    out = config.FIGURES_DIR / "03_correlation_matrix.png"
    fig.savefig(out, dpi=config.FIG_DPI)
    plt.close(fig)
    return out


def plot_temperature_vs_irradiance(df: pd.DataFrame) -> Path:
    """Relação T2M vs irradiância colorida por região."""
    _setup_style()
    sample = df.sample(min(8000, len(df)), random_state=config.RANDOM_STATE)
    g = sns.lmplot(
        data=sample, x="T2M", y="ALLSKY_SFC_SW_DWN",
        hue="region", scatter_kws={"alpha": 0.25, "s": 10},
        line_kws={"linewidth": 2}, height=5, aspect=1.6, ci=None,
    )
    g.set_axis_labels("Temperatura média a 2 m (°C)", "Irradiância (kWh/m²/dia)")
    g.figure.suptitle("Temperatura vs irradiância por região (amostra)", y=1.02)
    out = config.FIGURES_DIR / "04_temp_vs_irradiance.png"
    g.savefig(out, dpi=config.FIG_DPI, bbox_inches="tight")
    plt.close(g.figure)
    return out


def plot_long_term_trend(df: pd.DataFrame) -> Path:
    """Tendência anual média da irradiância — Q10 e diagnóstico de mudança."""
    _setup_style()
    yearly = (
        df.assign(year=df["date"].dt.year)
          .groupby(["year", "region"])["ALLSKY_SFC_SW_DWN"]
          .mean()
          .reset_index()
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=yearly, x="year", y="ALLSKY_SFC_SW_DWN",
                 hue="region", marker="o", ax=ax)
    ax.set_title("Irradiância média anual por região (2015-2024)")
    ax.set_ylabel("kWh/m²/dia")
    ax.set_xlabel("Ano")
    fig.tight_layout()
    out = config.FIGURES_DIR / "05_yearly_trend.png"
    fig.savefig(out, dpi=config.FIG_DPI)
    plt.close(fig)
    return out


def plot_missing_summary(df: pd.DataFrame) -> Path:
    """Mapa de missing por variável e cidade."""
    _setup_style()
    cols = [c for c in config.NASA_PARAMETERS if c in df.columns]
    miss = (
        df.groupby("city")[cols]
          .apply(lambda g: g.isna().mean() * 100)
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(miss, cmap="rocket_r", annot=False, ax=ax,
                cbar_kws={"label": "% missing"})
    ax.set_title("Proporção de missing por variável e cidade")
    fig.tight_layout()
    out = config.FIGURES_DIR / "06_missing_heatmap.png"
    fig.savefig(out, dpi=config.FIG_DPI)
    plt.close(fig)
    return out


def descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Resumo descritivo agregado."""
    cols = [c for c in config.NASA_PARAMETERS if c in df.columns]
    desc = (
        df.groupby("region")[cols]
          .mean()
          .round(2)
    )
    return desc


def run_all(df: pd.DataFrame) -> dict[str, Path]:
    """Executa todas as visualizações e retorna mapa nome -> caminho."""
    return {
        "irradiance_by_region": plot_irradiance_distribution(df),
        "seasonality_heatmap": plot_monthly_seasonality(df),
        "correlation_matrix": plot_correlation_matrix(df),
        "temp_vs_irradiance": plot_temperature_vs_irradiance(df),
        "yearly_trend": plot_long_term_trend(df),
        "missing_heatmap": plot_missing_summary(df),
    }
