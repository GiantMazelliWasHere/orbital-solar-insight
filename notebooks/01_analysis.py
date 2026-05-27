"""Versão script da análise do notebook ``01_analysis.ipynb``.

Permite reproduzir toda a análise sem Jupyter::

    python notebooks/01_analysis.py

O script imprime as métricas principais no stdout e regenera figuras em
``outputs/figures``. Idêntico em conteúdo ao notebook — útil para CI,
ambientes sem Jupyter ou debug rápido.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import (  # noqa: E402
    clustering,
    config,
    data_acquisition,
    data_preprocessing,
    eda,
    modeling,
)


def banner(text: str) -> None:
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


def main() -> None:
    # 1. Aquisição (idempotente)
    banner("1. Aquisição de dados")
    csv_path = data_acquisition.download_all()
    df = pd.read_csv(csv_path, parse_dates=["date"])
    print(f"linhas: {len(df)} | cidades: {df['city'].nunique()}")
    print(f"período: {df['date'].min().date()} -> {df['date'].max().date()}")

    # 2. EDA
    banner("2. Análise exploratória — figuras em outputs/figures/")
    figures = eda.run_all(df)
    for name, path in figures.items():
        print(f"  - {name}: {path.name}")

    # 3. Q1 — supervisionado
    banner("3. Q1 — Regressão (XGBoost / RandomForest / Linear)")
    X, y, meta = data_preprocessing.build_feature_matrix(df)
    X_tr, X_te, y_tr, y_te, meta_tr, meta_te = (
        data_preprocessing.temporal_train_test_split(X, y, meta)
    )
    print(f"treino: {len(X_tr):>6} | teste: {len(X_te):>6} | features: {X.shape[1]}")

    results = modeling.train_and_compare(X_tr, X_te, y_tr, y_te)
    metrics = modeling.metrics_table(results)
    print("\n", metrics.to_string())

    best_name = metrics["RMSE teste"].idxmin()
    best = results[best_name]
    print(f"\nmelhor modelo: {best_name}")
    modeling.plot_predictions(best, X_te, y_te, meta_te)
    modeling.plot_feature_importance(best)
    modeling.persist_best(results)

    # 4. Q4 — não-supervisionado
    banner("4. Q4 — Clustering de cidades por perfil climático")
    signatures = data_preprocessing.aggregate_city_climatology(df)
    scores = clustering.evaluate_k_range(signatures)
    print("\nDiagnóstico de k:\n", scores.to_string(index=False))

    result = clustering.fit_kmeans(signatures)
    print(f"\nk={result.k} | silhueta={result.silhouette:.3f} "
          f"| calinski={result.calinski:.1f}")
    clustering.plot_k_diagnostics(scores)
    clustering.plot_pca_projection(result)
    clustering.plot_cluster_heatmap(result)

    print("\nCidades por cluster:")
    print(result.summary[["city", "region", "biome", "cluster"]]
                       .sort_values("cluster")
                       .to_string(index=False))


if __name__ == "__main__":
    main()
