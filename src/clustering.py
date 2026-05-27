"""Modelagem não-supervisionada — Q4: agrupamento de cidades por perfil climático.

Pipeline
--------
1. Calcula assinaturas climáticas anuais por cidade (média de irradiância,
   amplitude térmica, umidade, precipitação, vento, índice de claridade).
2. Padroniza com ``StandardScaler``.
3. Avalia ``KMeans`` em k=2..8 usando silhueta e *inertia* (curva-cotovelo).
4. Ajusta o modelo final com o k escolhido (default 4) e devolve rótulos +
   métricas + projeção 2D via PCA para visualização.

Os clusters não dependem das séries diárias completas: a assinatura agregada
é suficiente para distinguir biomas e regimes climáticos no Brasil.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from . import config

logger = logging.getLogger(__name__)

CLUSTER_FEATURES: list[str] = [
    "irrad_mean", "irrad_std", "temp_mean", "temp_amplitude",
    "rh_mean", "prec_total", "wind_mean", "kt_mean",
]


@dataclass
class ClusterResult:
    k: int
    model: KMeans
    scaler: StandardScaler
    labels: np.ndarray
    silhouette: float
    calinski: float
    summary: pd.DataFrame


def evaluate_k_range(
    signatures: pd.DataFrame,
    k_range=config.KMEANS_RANGE,
) -> pd.DataFrame:
    """Calcula silhueta e inércia para escolher k."""
    X = signatures[CLUSTER_FEATURES].values
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=20, random_state=config.RANDOM_STATE)
        labels = km.fit_predict(Xs)
        sil = silhouette_score(Xs, labels) if k > 1 else np.nan
        rows.append({
            "k": k,
            "inertia": km.inertia_,
            "silhouette": sil,
            "calinski_harabasz": calinski_harabasz_score(Xs, labels),
        })
    return pd.DataFrame(rows)


def fit_kmeans(
    signatures: pd.DataFrame, k: int = config.KMEANS_DEFAULT_K,
) -> ClusterResult:
    """Ajusta KMeans com k fixo e devolve resultado consolidado."""
    X = signatures[CLUSTER_FEATURES].values
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    km = KMeans(n_clusters=k, n_init=30, random_state=config.RANDOM_STATE)
    labels = km.fit_predict(Xs)
    sil = silhouette_score(Xs, labels)
    cal = calinski_harabasz_score(Xs, labels)

    summary = signatures.assign(cluster=labels).copy()
    logger.info("KMeans k=%d | silhueta=%.3f | calinski=%.1f", k, sil, cal)

    return ClusterResult(
        k=k, model=km, scaler=scaler, labels=labels,
        silhouette=sil, calinski=cal, summary=summary,
    )


def cluster_profiles(result: ClusterResult) -> pd.DataFrame:
    """Perfil médio por cluster (centro interpretável)."""
    return (
        result.summary
              .groupby("cluster")[CLUSTER_FEATURES]
              .mean()
              .round(2)
    )


# ---------------------------------------------------------------------------
# Visualizações
# ---------------------------------------------------------------------------

def plot_k_diagnostics(scores: pd.DataFrame) -> Path:
    """Curva-cotovelo + silhueta lado a lado."""
    plt.style.use(config.PLT_STYLE)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(scores["k"], scores["inertia"], marker="o", color="#1f77b4")
    axes[0].set_title("Curva-cotovelo (inércia)")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("Inércia")

    axes[1].plot(scores["k"], scores["silhouette"], marker="o", color="#d62728")
    axes[1].set_title("Coeficiente de silhueta")
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("Silhouette score")

    fig.tight_layout()
    out = config.FIGURES_DIR / "09_kmeans_diagnostics.png"
    fig.savefig(out, dpi=config.FIG_DPI)
    plt.close(fig)
    return out


def plot_pca_projection(result: ClusterResult) -> Path:
    """Projeção 2D via PCA dos clusters, anotando o nome das cidades."""
    plt.style.use(config.PLT_STYLE)
    X = result.summary[CLUSTER_FEATURES].values
    Xs = result.scaler.transform(X)
    coords = PCA(n_components=2, random_state=config.RANDOM_STATE).fit_transform(Xs)

    fig, ax = plt.subplots(figsize=(10, 6))
    palette = sns.color_palette("Set2", n_colors=result.k)
    for cluster_id in range(result.k):
        mask = result.labels == cluster_id
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            s=180, color=palette[cluster_id], edgecolor="black",
            label=f"Cluster {cluster_id}",
        )
    for (x, y), city in zip(coords, result.summary["city"]):
        ax.annotate(city, (x, y), xytext=(6, 5), textcoords="offset points",
                    fontsize=9)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"Projeção PCA dos perfis climáticos (k={result.k})")
    ax.legend(loc="best", framealpha=0.9)
    fig.tight_layout()
    out = config.FIGURES_DIR / "10_cluster_pca.png"
    fig.savefig(out, dpi=config.FIG_DPI)
    plt.close(fig)
    return out


def plot_cluster_heatmap(result: ClusterResult) -> Path:
    """Heatmap padronizado de centróides — perfil interpretável de cada grupo."""
    plt.style.use(config.PLT_STYLE)
    centers = result.model.cluster_centers_
    df = pd.DataFrame(centers, columns=CLUSTER_FEATURES,
                      index=[f"Cluster {i}" for i in range(result.k)])
    fig, ax = plt.subplots(figsize=(10, 4 + 0.3 * result.k))
    sns.heatmap(df, cmap="vlag", center=0, annot=True, fmt=".2f", ax=ax,
                cbar_kws={"label": "z-score do centróide"})
    ax.set_title("Perfil padronizado de cada cluster (z-scores)")
    fig.tight_layout()
    out = config.FIGURES_DIR / "11_cluster_profiles.png"
    fig.savefig(out, dpi=config.FIG_DPI)
    plt.close(fig)
    return out
