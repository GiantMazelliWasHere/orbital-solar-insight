"""Pipeline orquestrador — Orbital Solar Insight.

Uso (a partir da raiz do projeto)::

    python -m src.main           # roda pipeline completo
    python -m src.main --no-download   # usa CSVs já presentes
    python -m src.main --force-download  # re-baixa tudo

Saída esperada:
* CSVs em ``data/raw/`` e ``data/processed/``
* PNGs em ``outputs/figures/``
* Modelo serializado em ``outputs/models/``
* Tabelas-resumo impressas no stdout
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from . import (
    clustering,
    config,
    data_acquisition,
    data_preprocessing,
    eda,
    modeling,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pipeline NASA POWER + ML")
    p.add_argument("--no-download", action="store_true",
                   help="pula a etapa de download (usa cache).")
    p.add_argument("--force-download", action="store_true",
                   help="re-baixa mesmo se houver cache.")
    return p.parse_args()


def step_acquire(args: argparse.Namespace) -> Path:
    if args.no_download:
        path = config.PROCESSED_DIR / "nasa_power_consolidated.csv"
        if not path.exists():
            raise FileNotFoundError(
                "Não há dataset consolidado em cache; rode sem --no-download."
            )
        logger.info("usando cache: %s", path)
        return path
    return data_acquisition.download_all(force=args.force_download)


def step_eda(df: pd.DataFrame) -> dict:
    figures = eda.run_all(df)
    desc = eda.descriptive_stats(df)
    desc.to_csv(config.PROCESSED_DIR / "descriptive_stats.csv")
    return {"figures": {k: str(v) for k, v in figures.items()},
            "descriptive_stats": desc.to_dict()}


def step_supervised(df: pd.DataFrame) -> dict:
    X, y, meta = data_preprocessing.build_feature_matrix(df)
    X_tr, X_te, y_tr, y_te, meta_tr, meta_te = (
        data_preprocessing.temporal_train_test_split(X, y, meta)
    )
    results = modeling.train_and_compare(X_tr, X_te, y_tr, y_te)

    metrics = modeling.metrics_table(results)
    metrics.to_csv(config.PROCESSED_DIR / "model_metrics.csv")
    logger.info("\n%s", metrics.to_string())

    best_name = metrics["RMSE teste"].idxmin()
    best_result = results[best_name]
    modeling.persist_best(results)

    pred_path = modeling.plot_predictions(best_result, X_te, y_te, meta_te)
    fi_path = modeling.plot_feature_importance(best_result)

    return {
        "metrics": metrics.to_dict(),
        "best_model": best_name,
        "pred_figure": str(pred_path),
        "feature_importance_figure": str(fi_path) if fi_path else None,
    }


def step_unsupervised(df: pd.DataFrame) -> dict:
    signatures = data_preprocessing.aggregate_city_climatology(df)
    signatures.to_csv(config.PROCESSED_DIR / "city_signatures.csv", index=False)

    scores = clustering.evaluate_k_range(signatures)
    scores.to_csv(config.PROCESSED_DIR / "kmeans_scores.csv", index=False)

    result = clustering.fit_kmeans(signatures)
    profiles = clustering.cluster_profiles(result)
    profiles.to_csv(config.PROCESSED_DIR / "cluster_profiles.csv")
    result.summary.to_csv(config.PROCESSED_DIR / "city_clusters.csv", index=False)

    return {
        "diagnostics_figure": str(clustering.plot_k_diagnostics(scores)),
        "pca_figure": str(clustering.plot_pca_projection(result)),
        "profiles_figure": str(clustering.plot_cluster_heatmap(result)),
        "silhouette": float(result.silhouette),
        "k": int(result.k),
        "profiles": profiles.to_dict(),
    }


def main() -> None:
    args = parse_args()

    csv_path = step_acquire(args)
    df = pd.read_csv(csv_path, parse_dates=["date"])

    eda_out = step_eda(df)
    sup_out = step_supervised(df)
    unsup_out = step_unsupervised(df)

    summary = {"eda": eda_out, "supervised": sup_out, "unsupervised": unsup_out}
    out_path = config.PROCESSED_DIR / "pipeline_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("resumo do pipeline -> %s", out_path)


if __name__ == "__main__":
    main()
