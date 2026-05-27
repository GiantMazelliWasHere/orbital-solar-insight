# Orbital Solar Insight — Apresentação

> Roteiro em **bullets**, organizado por slide. Cada bloco corresponde a um slide.

---

## Slide 1 · Capa

- **Orbital Solar Insight** — FIAP Global Solution 2026
- IA e Machine Learning aplicados a dados de **observação da Terra** (NASA POWER)
- Foco: **energia solar urbana** + **perfis climáticos regionais**
- ODS: **9 · 11 · 13**

---

## Slide 2 · O problema

- Brasil tem **clima fortemente heterogêneo** entre regiões — Amazônia úmida, Nordeste seco, Sul temperado.
- Planejamento energético e adaptação climática urbana exigem **dados confiáveis e espacialmente distribuídos**.
- Estações de superfície (INMET) concentram-se no **Sul/Sudeste** — vazios no Norte/Centro-Oeste.
- **Satélite resolve o vazio:** cobertura global e uniforme.

---

## Slide 3 · Perguntas de pesquisa

- **Q1 — Supervisionado.** Prever a **irradiância solar diária** em capitais brasileiras com erro adequado para dimensionar fotovoltaicos.
- **Q4 — Não-supervisionado.** Agrupar capitais por **perfil climático** para apoiar políticas públicas regionalizadas.
- **Por que as duas juntas?** Q1 sustenta decisões locais; Q4 sustenta decisões estruturais.

---

## Slide 4 · Dataset NASA POWER

- API pública: `power.larc.nasa.gov/api/temporal/daily/point` — **sem chave**.
- **10 variáveis** diárias (irradiância, T, RH, vento, chuva, pressão, Kt).
- **12 capitais brasileiras** cobrindo 5 regiões e 5 biomas.
- **2015–2024** → **43.836 registros** diários.
- **Zero missing** após filtros — produto maduro pós-2015.

---

## Slide 5 · Pipeline

- `data_acquisition.py` → API → CSVs por cidade.
- `data_preprocessing.py` → limpeza + lags + médias móveis + componentes cíclicos.
- `eda.py` → 6 visualizações (distribuição, sazonalidade, correlação, missing…).
- `modeling.py` → Linear / RF / XGBoost com `TimeSeriesSplit`.
- `clustering.py` → KMeans + PCA + silhueta.
- `main.py` → orquestrador, idempotente, com cache.

---

## Slide 6 · Decisões metodológicas

- **Split temporal** (últimos 20 %) — evita vazamento, simula uso operacional.
- **`TimeSeriesSplit(5)`** na validação cruzada.
- **Features cíclicas** (`sin/cos` do DOY) — modelos de árvore capturam sazonalidade direta.
- **Lags 1 e 7 dias** — memória atmosférica curta.
- **One-hot da região** — diferencia regimes sem precisar treinar um modelo por cidade.

---

## Slide 7 · Q1 — Resultados de regressão

| Modelo | RMSE teste | R² teste | MAPE |
|---|---|---|---|
| Linear | 0,272 | 0,969 | 6,6 % |
| Random Forest | 0,107 | 0,995 | 1,7 % |
| **XGBoost** | **0,069** | **0,998** | **1,26 %** |

- Erro de **0,069 kWh/m²/dia** ≈ **1,3 % relativo** sobre média típica.
- **Mais preciso que a próxima fonte de incerteza** (performance ratio FV).

---

## Slide 8 · Features que mais explicam (XGBoost)

1. **ALLSKY_KT** — índice de claridade
2. **CLRSKY_SFC_SW_DWN** — teto físico
3. **Lags 1d / médias 7d** — persistência
4. **`doy_sin`, `doy_cos`** — sazonalidade
5. **RH2M** — proxy de nuvens

> Interpretação: **modelo aprende a física** sem ser ensinado.

---

## Slide 9 · Q4 — Quantos clusters?

- Curva-cotovelo + silhueta + Calinski-Harabasz **convergem em k = 4**.
- Silhueta máxima: **0,404** (moderada).
- Solução estável (n_init = 30).

---

## Slide 10 · Q4 — Quatro Brasis climáticos

| Cluster | Cidades | Caracterização |
|---|---|---|
| 0 — Amazônia úmida | Manaus, Belém | T estável, RH 83 %, chuva alta |
| 1 — Sul/Sudeste úmido | Porto Alegre, Curitiba, São Paulo, Rio | T 20°C, amplitude alta |
| 2 — Litoral Nordeste | **Fortaleza, Recife** | **Solar + vento alto = híbrido ideal** |
| 3 — Cerrado interior | Brasília, Goiânia, BH, Salvador | RH 69 %, céu limpo, alta Kt |

- Clusterização **descobre região e bioma** sem usá-los como input.

---

## Slide 11 · Interpretação para os ODS

- **ODS 9.** Q1 alimenta planejamento de redes — geração estimada por CEP com erro < 2 %.
- **ODS 11.** Q4 mostra que políticas de adaptação podem ser **compartilhadas entre cidades do mesmo cluster** — racionaliza esforço técnico.
- **ODS 13.** A série permite, em trabalhos futuros, detectar **tendências e anomalias** regionais (extensão para Q10).

---

## Slide 12 · Limitações honestas

- Resolução espacial **~50 km/píxel** — política regional, não rooftop.
- `n = 12` cidades — silhueta moderada; com 50 a separação seria nítida.
- Modelo Q1 é **estático** (mesmo dia) — extensão prospectiva é direta com lags.
- Sem grid search — ganho marginal esperado.

---

## Slide 13 · Reprodutibilidade

- `pip install -r requirements.txt`
- `python -m src.main` → pipeline completo.
- `python -m src.main --no-download` → usa cache.
- Notebook `notebooks/01_analysis.ipynb` espelha os módulos.
- Saídas: **11 PNGs**, **modelo joblib**, **8 CSVs de resultados**, JSON-resumo.

---

## Slide 14 · Take-aways

- Dados **públicos** e **gratuitos** + **ML clássico** = previsão solar com erro ~1,3 %.
- O **satélite carrega a assinatura climática regional** — clustering recupera biomas sem usá-los.
- Pipeline é **modular, testado, idempotente e reprodutível**.
- Próximo passo: **previsão prospectiva (h+1, h+7)** e cruzamento socioeconômico.

---

## Slide 15 · Obrigado

- Repositório: `github.com/<equipe>/orbital-solar-insight`
- Dataset: NASA POWER — `power.larc.nasa.gov`
- Equipe: **FIAP — Global Solution 2026**
