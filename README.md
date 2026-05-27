# Orbital Solar Insight

## Equipe

- Carolina Cavalli Machado | RM: 552925
- Eduardo Mazelli | RM:553236
- Joseh Gabriel Trimboli Agra | RM:553094
- Lucas Masaki Nagahama  | RM:553084
- Pedro Henrique de Assumção Lima | RM:552746

> **FIAP — Global Solution 2026 · IA e Machine Learning com dados de temática espacial**
> Investigação analítica do potencial solar e dos perfis climáticos urbanos brasileiros usando dados de satélite da **NASA POWER**.

## Sumário executivo

Este projeto combina **regressão supervisionada** e **agrupamento não-supervisionado** sobre 10 anos (2015–2024) de dados diários derivados de satélite para 12 capitais brasileiras, com o objetivo de:

1. **(Q1)** prever a **irradiância solar global diária** em centros urbanos — insumo para dimensionamento de infraestrutura fotovoltaica;
2. **(Q4)** agrupar cidades por **perfil climático** — apoio a políticas públicas regionalizadas.

### Principais resultados

| Modelo (Q1) | CV RMSE | RMSE teste | MAE teste | R² teste | MAPE |
|---|---|---|---|---|---|
| Regressão Linear | 0,259 | 0,272 | 0,184 | 0,969 | 6,64 % |
| Random Forest | 0,098 | 0,107 | 0,069 | 0,995 | 1,67 % |
| **XGBoost (vencedor)** | **0,077** | **0,069** | **0,052** | **0,998** | **1,26 %** |

> Erro médio de 0,069 kWh/m²/dia em irradiância global — equivalente a ~1,3% sobre uma média diária típica de 5 kWh/m²/dia.

| Clustering (Q4) | k | Silhueta | Calinski-Harabasz |
|---|---|---|---|
| KMeans (assinaturas anuais) | **4** | **0,404** | 10,3 |

Os quatro clusters resultantes refletem regimes climáticos brasileiros bem reconhecíveis: **Amazônia úmida**, **Sul/Sudeste úmido**, **Litoral Nordeste seco e ventoso**, e **Cerrado de interior**.

## ODS endereçados

- **ODS 9 — Indústria, Inovação e Infraestrutura:** previsão de geração solar para planejamento de redes.
- **ODS 11 — Cidades e Comunidades Sustentáveis:** identificação de cidades vulneráveis a regimes climáticos extremos.
- **ODS 13 — Ação Climática:** uso de dados de observação da Terra para apoio à tomada de decisão climática.

## Estrutura do repositório

```
orbital-solar-insight/
├── data/
│   ├── raw/              # CSVs por cidade vindos da API NASA POWER
│   └── processed/        # Dataset consolidado, métricas e clusters
├── notebooks/
│   └── 01_analysis.ipynb # Notebook exploratório
├── src/                  # Pacote Python modular
│   ├── config.py             — caminhos, cidades, parâmetros
│   ├── data_acquisition.py   — chamada à API NASA POWER
│   ├── data_preprocessing.py — limpeza, feature engineering, splits
│   ├── eda.py                — visualizações exploratórias
│   ├── modeling.py           — regressão (Q1)
│   ├── clustering.py         — KMeans (Q4)
│   └── main.py               — orquestrador
├── outputs/
│   ├── figures/          # 11 PNGs gerados pelo pipeline
│   └── models/           # Modelo serializado (joblib)
├── reports/
│   ├── relatorio.md      # Relatório acadêmico
│   ├── relatorio.pdf     # Versão PDF
│   └── apresentacao.md   # Bullets para slide deck
├── requirements.txt
└── README.md
```

## Reprodutibilidade

### Pré-requisitos

- Python 3.10+ (testado em 3.14)
- ~100 MB livres em disco para dados brutos
- Conexão com internet apenas para o primeiro download

### Instalação

```bash
git clone https://github.com/<seu-usuario>/orbital-solar-insight.git
cd orbital-solar-insight
python -m venv .venv
.\.venv\Scripts\activate          # Windows PowerShell
# source .venv/bin/activate       # macOS / Linux
pip install -r requirements.txt
```

### Como obter os dados — duas opções

#### **Opção A · Download programático via API (recomendado, totalmente reprodutível)**

Basta executar o script de aquisição. Nenhuma chave de API é necessária:

```bash
python -m src.data_acquisition
```

O script:

1. Itera sobre as **12 capitais** em [`src/config.py`](src/config.py) (latitude/longitude pré-definidas).
2. Consulta o endpoint **`https://power.larc.nasa.gov/api/temporal/daily/point`** com os parâmetros:

   | Parâmetro HTTP | Valor |
   |---|---|
   | `parameters` | `ALLSKY_SFC_SW_DWN,CLRSKY_SFC_SW_DWN,T2M,T2M_MAX,T2M_MIN,RH2M,PRECTOTCORR,WS2M,PS,ALLSKY_KT` |
   | `community` | `RE` (Renewable Energy) |
   | `longitude` | longitude da cidade |
   | `latitude` | latitude da cidade |
   | `start` | `20150101` |
   | `end` | `20241231` |
   | `format` | `JSON` |

3. Salva um CSV por cidade em `data/raw/<cidade>.csv` (idempotente: re-execuções usam cache).
4. Gera o dataset consolidado em `data/processed/nasa_power_consolidated.csv`.

> **Dica:** rode o script novamente passando `force=True` em `download_all(force=True)` se quiser refazer o download.

#### **Opção B · Download manual pelo portal**

1. Acesse o portal interativo: <https://power.larc.nasa.gov/data-access-viewer/>
2. Em **User Community**, selecione `Renewable Energy`.
3. Em **Temporal Average**, selecione `Daily`.
4. Em **Location**, digite latitude/longitude (ou clique no mapa) para cada cidade da tabela abaixo.
5. Em **Time Extent**, defina **2015-01-01 → 2024-12-31**.
6. Em **Parameters**, marque exatamente:
   - ALLSKY_SFC_SW_DWN
   - CLRSKY_SFC_SW_DWN
   - T2M, T2M_MAX, T2M_MIN
   - RH2M
   - PRECTOTCORR
   - WS2M
   - PS
   - ALLSKY_KT
7. Em **Output**, escolha **CSV** e baixe.
8. Renomeie cada arquivo para `<cidade>.csv` (em minúsculas, sem acento) e mova para `data/raw/`.

##### Coordenadas usadas

| Cidade | UF | Latitude | Longitude | Região | Bioma |
|---|---|---|---|---|---|
| Manaus | AM | -3.1190 | -60.0217 | Norte | Amazônia |
| Belém | PA | -1.4558 | -48.5039 | Norte | Amazônia |
| Fortaleza | CE | -3.7172 | -38.5433 | Nordeste | Caatinga |
| Recife | PE | -8.0476 | -34.8770 | Nordeste | Mata Atlântica |
| Salvador | BA | -12.9714 | -38.5014 | Nordeste | Mata Atlântica |
| Brasília | DF | -15.7942 | -47.8825 | Centro-Oeste | Cerrado |
| Goiânia | GO | -16.6864 | -49.2643 | Centro-Oeste | Cerrado |
| Belo Horizonte | MG | -19.9167 | -43.9345 | Sudeste | Cerrado |
| São Paulo | SP | -23.5505 | -46.6333 | Sudeste | Mata Atlântica |
| Rio de Janeiro | RJ | -22.9068 | -43.1729 | Sudeste | Mata Atlântica |
| Curitiba | PR | -25.4284 | -49.2733 | Sul | Mata Atlântica |
| Porto Alegre | RS | -30.0346 | -51.2177 | Sul | Pampa |

### Executando o pipeline

```bash
# Pipeline completo (baixa se necessário, treina, avalia, plota, persiste)
python -m src.main

# Apenas usar cache (mais rápido)
python -m src.main --no-download

# Forçar re-download
python -m src.main --force-download
```

Ao final, inspecione:

- `outputs/figures/` — 11 PNGs (EDA, predições, clusters).
- `outputs/models/best_XGBoost.joblib` — modelo persistido.
- `data/processed/pipeline_summary.json` — sumário consolidado.

### Notebook exploratório

```bash
jupyter notebook notebooks/01_analysis.ipynb
```

O notebook usa exatamente os mesmos módulos de `src/`, garantindo paridade com o pipeline em script.

## Variáveis NASA POWER — significado técnico

| Variável | Descrição | Unidade |
|---|---|---|
| `ALLSKY_SFC_SW_DWN` | Irradiância solar global na superfície (todas as condições de céu) — **alvo da regressão** | kWh/m²/dia |
| `CLRSKY_SFC_SW_DWN` | Irradiância sob céu limpo — proxy do teto máximo de geração | kWh/m²/dia |
| `ALLSKY_KT` | Índice de claridade = ALLSKY/CLRSKY — adimensional, mede transmissividade | — |
| `T2M`, `T2M_MAX`, `T2M_MIN` | Temperatura média / máxima / mínima a 2 m | °C |
| `RH2M` | Umidade relativa a 2 m | % |
| `PRECTOTCORR` | Precipitação corrigida por bias | mm/dia |
| `WS2M` | Velocidade do vento a 2 m | m/s |
| `PS` | Pressão atmosférica na superfície | kPa |

> **Nota técnica:** a NASA POWER fornece um *raster* global (~0,5°). Em centros urbanos densos, micro-clima e ilhas de calor não são resolvidos — uso operacional requer calibração com estações de superfície (INMET).

## Limitações conhecidas

- **Granularidade espacial:** ~50 km/píxel — agregação adequada para política regional, não para um telhado específico.
- **Sazonalidade ENSO:** 10 anos cobrem ~2 ciclos completos — extender para 30 anos reduz viés de período.
- **Cluster com n=12 cidades:** silhueta de 0,40 indica separação moderada — mais cidades reforçariam a estabilidade.

## Licença

Dataset: NASA POWER — domínio público (NASA Langley Research Center).  
Código: MIT.
