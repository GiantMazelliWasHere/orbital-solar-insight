"""Configurações globais do projeto Orbital Solar Insight.

Centraliza caminhos, lista de cidades, parâmetros NASA POWER e hiperparâmetros
para que todo o pipeline seja reprodutível com uma única fonte de verdade.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Caminhos do projeto
# ---------------------------------------------------------------------------
ROOT_DIR: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = ROOT_DIR / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
OUTPUTS_DIR: Path = ROOT_DIR / "outputs"
FIGURES_DIR: Path = OUTPUTS_DIR / "figures"
MODELS_DIR: Path = OUTPUTS_DIR / "models"
REPORTS_DIR: Path = ROOT_DIR / "reports"

for _dir in (RAW_DIR, PROCESSED_DIR, FIGURES_DIR, MODELS_DIR, REPORTS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Recorte temporal e geográfico
# ---------------------------------------------------------------------------
START_DATE: str = "20150101"   # 10 anos é suficiente para sazonalidade e ENSO
END_DATE: str = "20241231"


@dataclass(frozen=True)
class City:
    """Representa uma capital brasileira com coordenadas e biome dominante."""

    name: str
    uf: str
    latitude: float
    longitude: float
    region: str
    biome: str


CITIES: list[City] = [
    City("Manaus", "AM", -3.1190, -60.0217, "Norte", "Amazonia"),
    City("Belem", "PA", -1.4558, -48.5039, "Norte", "Amazonia"),
    City("Fortaleza", "CE", -3.7172, -38.5433, "Nordeste", "Caatinga"),
    City("Recife", "PE", -8.0476, -34.8770, "Nordeste", "Mata Atlantica"),
    City("Salvador", "BA", -12.9714, -38.5014, "Nordeste", "Mata Atlantica"),
    City("Brasilia", "DF", -15.7942, -47.8825, "Centro-Oeste", "Cerrado"),
    City("Goiania", "GO", -16.6864, -49.2643, "Centro-Oeste", "Cerrado"),
    City("Belo Horizonte", "MG", -19.9167, -43.9345, "Sudeste", "Cerrado"),
    City("Sao Paulo", "SP", -23.5505, -46.6333, "Sudeste", "Mata Atlantica"),
    City("Rio de Janeiro", "RJ", -22.9068, -43.1729, "Sudeste", "Mata Atlantica"),
    City("Curitiba", "PR", -25.4284, -49.2733, "Sul", "Mata Atlantica"),
    City("Porto Alegre", "RS", -30.0346, -51.2177, "Sul", "Pampa"),
]


# ---------------------------------------------------------------------------
# NASA POWER — parâmetros e endpoint
# ---------------------------------------------------------------------------
NASA_POWER_ENDPOINT: str = (
    "https://power.larc.nasa.gov/api/temporal/daily/point"
)

# Subconjunto de parâmetros relevantes para análise solar/clima.
# Documentação: https://power.larc.nasa.gov/docs/services/api/temporal/daily/
NASA_PARAMETERS: list[str] = [
    "ALLSKY_SFC_SW_DWN",   # Irradiância solar global na superfície (kWh/m2/dia)
    "CLRSKY_SFC_SW_DWN",   # Irradiância em céu limpo (proxy do potencial maximo)
    "T2M",                 # Temperatura média a 2 m (C)
    "T2M_MAX",             # Temperatura máxima a 2 m (C)
    "T2M_MIN",             # Temperatura mínima a 2 m (C)
    "RH2M",                # Umidade relativa a 2 m (%)
    "PRECTOTCORR",         # Precipitação corrigida (mm/dia)
    "WS2M",                # Velocidade do vento a 2 m (m/s)
    "PS",                  # Pressão atmosférica na superfície (kPa)
    "ALLSKY_KT",           # Índice de claridade (transmissividade atmosférica)
]

# A NASA POWER usa o valor sentinela -999 para missing data.
NASA_MISSING_SENTINEL: float = -999.0


# ---------------------------------------------------------------------------
# Hiperparâmetros e seeds
# ---------------------------------------------------------------------------
RANDOM_STATE: int = 42
TEST_SIZE: float = 0.2          # divisão temporal: últimos ~20% das datas
CV_FOLDS: int = 5               # validação cruzada temporal

# Clustering (Q4)
KMEANS_RANGE: range = range(2, 9)
KMEANS_DEFAULT_K: int = 4       # ajustado após análise de silhueta


# ---------------------------------------------------------------------------
# Estilo visual
# ---------------------------------------------------------------------------
PLT_STYLE: str = "seaborn-v0_8-whitegrid"
PALETTE: str = "viridis"
FIG_DPI: int = 130
