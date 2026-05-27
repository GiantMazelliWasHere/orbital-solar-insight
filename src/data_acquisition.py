"""Aquisição de dados via API NASA POWER.

A NASA POWER (Prediction Of Worldwide Energy Resources) disponibiliza dados
diários derivados de satélite e modelos de reanálise, sem necessidade de chave
de acesso. Este módulo consulta o endpoint *temporal/daily/point* para cada
capital configurada e materializa um CSV por cidade em ``data/raw/``.

Funções principais
------------------
* :func:`build_request_params` — monta o dicionário de query string.
* :func:`fetch_city` — faz uma requisição com retries e devolve um ``DataFrame``.
* :func:`download_all` — itera sobre :data:`config.CITIES`, salva CSVs e
  retorna o caminho do dataset consolidado.

A função :func:`download_all` é idempotente: arquivos já presentes em
``data/raw/`` não são re-baixados, a menos que ``force=True``.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

from . import config

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


class NasaPowerError(RuntimeError):
    """Erro genérico ao consultar a API NASA POWER."""


def build_request_params(
    city: config.City,
    start: str = config.START_DATE,
    end: str = config.END_DATE,
    parameters: Iterable[str] = config.NASA_PARAMETERS,
) -> dict:
    """Monta o conjunto de parâmetros HTTP para uma cidade."""
    return {
        "parameters": ",".join(parameters),
        "community": "RE",          # Renewable Energy
        "longitude": city.longitude,
        "latitude": city.latitude,
        "start": start,
        "end": end,
        "format": "JSON",
    }


def _parse_response(payload: dict, city: config.City) -> pd.DataFrame:
    """Converte a resposta JSON da NASA POWER em DataFrame longo."""
    try:
        params_block = payload["properties"]["parameter"]
    except KeyError as exc:
        raise NasaPowerError(
            f"Resposta da API sem bloco esperado para {city.name}"
        ) from exc

    frames: list[pd.DataFrame] = []
    for parameter, series in params_block.items():
        df = (
            pd.Series(series, name=parameter)
            .rename_axis("date")
            .reset_index()
        )
        frames.append(df)

    merged = frames[0]
    for df in frames[1:]:
        merged = merged.merge(df, on="date", how="outer")

    merged["date"] = pd.to_datetime(merged["date"], format="%Y%m%d")
    merged["city"] = city.name
    merged["uf"] = city.uf
    merged["region"] = city.region
    merged["biome"] = city.biome
    merged["latitude"] = city.latitude
    merged["longitude"] = city.longitude

    merged = merged.replace(config.NASA_MISSING_SENTINEL, pd.NA)
    return merged.sort_values("date").reset_index(drop=True)


def fetch_city(
    city: config.City,
    *,
    max_retries: int = 3,
    backoff_seconds: float = 5.0,
    timeout: int = 60,
) -> pd.DataFrame:
    """Consulta a NASA POWER para uma cidade com retries exponenciais."""
    params = build_request_params(city)
    last_exc: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(
                "Consultando NASA POWER para %s (tentativa %d/%d)",
                city.name, attempt, max_retries,
            )
            response = requests.get(
                config.NASA_POWER_ENDPOINT,
                params=params,
                timeout=timeout,
            )
            response.raise_for_status()
            return _parse_response(response.json(), city)
        except (requests.RequestException, ValueError, NasaPowerError) as exc:
            last_exc = exc
            wait = backoff_seconds * attempt
            logger.warning("Falha em %s: %s — aguardando %.1fs", city.name, exc, wait)
            time.sleep(wait)

    raise NasaPowerError(
        f"Não foi possível obter dados para {city.name} após {max_retries} tentativas"
    ) from last_exc


def download_all(*, force: bool = False) -> Path:
    """Baixa dados para todas as cidades e devolve o CSV consolidado."""
    consolidated_rows: list[pd.DataFrame] = []

    for city in config.CITIES:
        slug = city.name.lower().replace(" ", "_")
        csv_path = config.RAW_DIR / f"{slug}.csv"

        if csv_path.exists() and not force:
            logger.info("[cache] já existe %s", csv_path.name)
            df = pd.read_csv(csv_path, parse_dates=["date"])
        else:
            df = fetch_city(city)
            df.to_csv(csv_path, index=False)
            logger.info("salvo %s (%d linhas)", csv_path.name, len(df))

        consolidated_rows.append(df)

    consolidated = pd.concat(consolidated_rows, ignore_index=True)
    consolidated_path = config.PROCESSED_DIR / "nasa_power_consolidated.csv"
    consolidated.to_csv(consolidated_path, index=False)
    logger.info(
        "dataset consolidado salvo em %s (%d linhas, %d cidades)",
        consolidated_path, len(consolidated), consolidated["city"].nunique(),
    )
    return consolidated_path


if __name__ == "__main__":
    download_all()
