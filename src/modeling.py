"""Modelagem supervisionada — Q1: previsão diária de irradiância solar.

Estratégia
----------
Três modelos com viés/variância contrastantes são comparados:

* **Regressão Linear** — baseline interpretável.
* **Random Forest** — captura não-linearidades e interações sem tuning intenso.
* **Gradient Boosted Trees (XGBoost)** — desempenho de referência em
  tabular, costuma vencer em RMSE com baixa engenharia adicional.

Validação cruzada usa :class:`TimeSeriesSplit` para evitar vazamento. A
divisão final separa o último ~20% das datas como hold-out (operacional).

A função :func:`train_and_compare` devolve um dicionário com modelos
treinados, métricas e DataFrame ``y_true × y_pred`` para diagnósticos.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config

logger = logging.getLogger(__name__)

try:
    from xgboost import XGBRegressor
    _HAS_XGB = True
except ImportError:  # ambiente sem xgboost
    _HAS_XGB = False
    logger.warning("xgboost ausente — modelo será omitido")


@dataclass
class ModelResult:
    name: str
    model: Any
    metrics: dict[str, float]
    cv_rmse: float
    cv_rmse_std: float
    feature_importance: pd.Series | None = field(default=None)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "RMSE": _rmse(y_true, y_pred),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
        "MAPE_%": float(np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1e-3, None))) * 100),
    }


def _cross_validate(
    estimator: Any, X: pd.DataFrame, y: pd.Series,
    n_splits: int = config.CV_FOLDS,
) -> tuple[float, float]:
    """RMSE médio em TimeSeriesSplit."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores: list[float] = []
    for fold, (tr, vl) in enumerate(tscv.split(X), start=1):
        estimator.fit(X.iloc[tr], y.iloc[tr])
        pred = estimator.predict(X.iloc[vl])
        scores.append(_rmse(y.iloc[vl].values, pred))
    return float(np.mean(scores)), float(np.std(scores))


def _build_estimators() -> dict[str, Any]:
    estimators: dict[str, Any] = {
        "LinearRegression": Pipeline([
            ("scaler", StandardScaler(with_mean=False)),
            ("reg", LinearRegression()),
        ]),
        "RandomForest": RandomForestRegressor(
            n_estimators=300,
            max_depth=18,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=config.RANDOM_STATE,
        ),
    }
    if _HAS_XGB:
        estimators["XGBoost"] = XGBRegressor(
            n_estimators=600,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=config.RANDOM_STATE,
            tree_method="hist",
            n_jobs=-1,
        )
    return estimators


def train_and_compare(
    X_train: pd.DataFrame, X_test: pd.DataFrame,
    y_train: pd.Series, y_test: pd.Series,
) -> dict[str, ModelResult]:
    """Treina os candidatos, avalia em hold-out e em CV temporal."""
    results: dict[str, ModelResult] = {}

    for name, estimator in _build_estimators().items():
        logger.info("==> treinando %s", name)
        cv_mean, cv_std = _cross_validate(estimator, X_train, y_train)
        estimator.fit(X_train, y_train)
        pred = estimator.predict(X_test)
        metrics = _evaluate(y_test.values, pred)
        logger.info("    CV RMSE %.3f ± %.3f | Hold-out RMSE %.3f | R² %.3f",
                    cv_mean, cv_std, metrics["RMSE"], metrics["R2"])

        importance: pd.Series | None = None
        if hasattr(estimator, "feature_importances_"):
            importance = pd.Series(
                estimator.feature_importances_, index=X_train.columns
            ).sort_values(ascending=False)
        elif isinstance(estimator, Pipeline) and hasattr(estimator[-1], "coef_"):
            importance = pd.Series(
                np.abs(estimator[-1].coef_), index=X_train.columns
            ).sort_values(ascending=False)

        results[name] = ModelResult(
            name=name, model=estimator, metrics=metrics,
            cv_rmse=cv_mean, cv_rmse_std=cv_std,
            feature_importance=importance,
        )

    return results


def persist_best(results: dict[str, ModelResult]) -> Path:
    """Persiste o melhor modelo (por RMSE de teste) em ``outputs/models``."""
    best_name, best = min(results.items(), key=lambda kv: kv[1].metrics["RMSE"])
    path = config.MODELS_DIR / f"best_{best_name}.joblib"
    joblib.dump(best.model, path)
    logger.info("modelo vencedor: %s -> %s", best_name, path)
    return path


def metrics_table(results: dict[str, ModelResult]) -> pd.DataFrame:
    """Tabela comparativa de métricas dos modelos."""
    rows = []
    for r in results.values():
        rows.append({
            "Modelo": r.name,
            "CV RMSE": round(r.cv_rmse, 3),
            "CV RMSE std": round(r.cv_rmse_std, 3),
            "RMSE teste": round(r.metrics["RMSE"], 3),
            "MAE teste": round(r.metrics["MAE"], 3),
            "R² teste": round(r.metrics["R2"], 3),
            "MAPE % teste": round(r.metrics["MAPE_%"], 2),
        })
    return pd.DataFrame(rows).set_index("Modelo")


def plot_predictions(
    best_result: ModelResult,
    X_test: pd.DataFrame, y_test: pd.Series, meta_test: pd.DataFrame,
) -> Path:
    """Scatter previsto × real + série temporal de uma cidade."""
    import matplotlib.pyplot as plt
    plt.style.use(config.PLT_STYLE)

    pred = best_result.model.predict(X_test)
    df = meta_test.copy()
    df["y_true"] = y_test.values
    df["y_pred"] = pred

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].scatter(df["y_true"], df["y_pred"], alpha=0.15, s=10, c="#1f77b4")
    lo, hi = df["y_true"].min(), df["y_true"].max()
    axes[0].plot([lo, hi], [lo, hi], "r--", lw=1)
    axes[0].set_xlabel("Irradiância observada (kWh/m²/dia)")
    axes[0].set_ylabel("Irradiância prevista")
    axes[0].set_title(f"Previsto × Observado — {best_result.name}")

    sample_city = df["city"].mode().iloc[0]
    ts = (df[df["city"] == sample_city]
          .sort_values("date")
          .tail(180))
    axes[1].plot(ts["date"], ts["y_true"], label="observado", lw=1.4)
    axes[1].plot(ts["date"], ts["y_pred"], label="previsto", lw=1.4, alpha=0.85)
    axes[1].set_title(f"Últimos 180 dias — {sample_city}")
    axes[1].set_ylabel("kWh/m²/dia")
    axes[1].legend()
    fig.autofmt_xdate()
    fig.tight_layout()

    out = config.FIGURES_DIR / "07_predictions.png"
    fig.savefig(out, dpi=config.FIG_DPI)
    plt.close(fig)
    return out


def plot_feature_importance(best_result: ModelResult, top: int = 15) -> Path | None:
    """Top features do melhor modelo (se aplicável)."""
    if best_result.feature_importance is None:
        return None
    import matplotlib.pyplot as plt
    plt.style.use(config.PLT_STYLE)

    top_feats = best_result.feature_importance.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top_feats.index, top_feats.values, color="#2ca02c")
    ax.set_title(f"Top {top} features — {best_result.name}")
    ax.set_xlabel("Importância")
    fig.tight_layout()
    out = config.FIGURES_DIR / "08_feature_importance.png"
    fig.savefig(out, dpi=config.FIG_DPI)
    plt.close(fig)
    return out
