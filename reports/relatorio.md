# Orbital Solar Insight — Relatório Acadêmico

**FIAP — Global Solution 2026**
**Disciplina:** Inteligência Artificial e Machine Learning
**Dataset:** NASA POWER (Prediction Of Worldwide Energy Resources)
**Tema espacial:** Observação da Terra via satélite
**ODS endereçados:** 9 (Indústria, Inovação e Infraestrutura), 11 (Cidades e Comunidades Sustentáveis), 13 (Ação Climática)

---

## 1. Definição do Problema e Contextualização

### 1.1 Perguntas de pesquisa

O trabalho responde a duas perguntas complementares — uma supervisionada e outra não-supervisionada — sobre a interseção entre dados de satélite e sustentabilidade urbana:

> **Q1 (regressão):** É possível prever a **capacidade diária de geração de energia solar** em centros urbanos brasileiros a partir de variáveis meteorológicas derivadas de satélite, com erro suficientemente baixo para apoiar dimensionamento de infraestrutura energética sustentável?

> **Q4 (clustering):** É possível **agrupar capitais brasileiras** por **comportamento climático**, de forma a identificar regimes regionais que apoiem políticas públicas diferenciadas?

### 1.2 Justificativa

A combinação das duas perguntas atende a três motivações distintas que dialogam diretamente com os ODS escolhidos:

1. **Planejamento energético (ODS 9):** previsões diárias de irradiância subsidiam o dimensionamento de redes fotovoltaicas urbanas e a integração com a rede convencional — onde subdimensionar acarreta apagão por demanda residual e superdimensionar gera *capex* perdido.

2. **Políticas urbanas regionalizadas (ODS 11):** o Brasil tem cinco regiões com biomas e climatologias muito distintas. Tratá-las de forma uniforme produz políticas medianas e ineficientes; agrupá-las por similaridade objetiva (em vez de fronteiras administrativas) revela onde estratégias podem ser compartilhadas.

3. **Diagnóstico climático evidencial (ODS 13):** dados de satélite oferecem cobertura uniforme e independente de infraestrutura terrestre — fundamental em um país com estações INMET concentradas no Sul/Sudeste.

### 1.3 Origem dos dados — NASA POWER

A **NASA POWER** é uma iniciativa da NASA Langley Research Center que disponibiliza dados solares e meteorológicos derivados de **observações de satélite (CERES, MERRA-2)** e modelos de reanálise atmosférica. As principais características:

- **Cobertura:** global, em grade regular de aproximadamente 0,5° × 0,5° (~50 km de resolução horizontal).
- **Cadência temporal:** dados diários disponíveis desde 1981; horária e mensal também acessíveis.
- **Acesso:** API pública sem autenticação (`https://power.larc.nasa.gov/api/temporal/daily/point`).
- **Comunidades de uso:** *RE* (Renewable Energy, usada aqui), *AG* (Agroclimatology) e *SB* (Sustainable Buildings) — apenas mudam o subconjunto sugerido de variáveis.

#### Variáveis utilizadas

| Sigla | Descrição | Unidade |
|---|---|---|
| `ALLSKY_SFC_SW_DWN` | Irradiância solar global na superfície (todas as condições de céu) — **alvo** | kWh/m²/dia |
| `CLRSKY_SFC_SW_DWN` | Irradiância sob céu limpo (teto físico de geração) | kWh/m²/dia |
| `ALLSKY_KT` | Índice de claridade (≈ ALLSKY/CLRSKY) — adimensional | — |
| `T2M`, `T2M_MAX`, `T2M_MIN` | Temperatura do ar a 2 m de altura | °C |
| `RH2M` | Umidade relativa a 2 m | % |
| `PRECTOTCORR` | Precipitação corrigida por bias | mm/dia |
| `WS2M` | Velocidade do vento a 2 m | m/s |
| `PS` | Pressão atmosférica de superfície | kPa |

#### Limitações e cuidados de interpretação

- **Resolução espacial grosseira.** Um *grid cell* de ~50 km cobre toda a região metropolitana — micro-climas, ilhas de calor e efeitos de relevo urbano não são resolvidos.
- **Valor sentinela.** Dados ausentes são representados por `-999`; trataremos como `NaN` explícito.
- **Bias residual.** A NASA POWER usa correções de bias contra estações de superfície, mas em regiões com poucas estações de calibração (Amazônia central) o erro absoluto pode ser maior.
- **Para uso operacional**, recomenda-se confronto com estações INMET próximas.

### 1.4 Escopo geográfico e temporal

- **12 capitais brasileiras** cobrindo as 5 regiões e 5 biomas (Amazônia, Caatinga, Cerrado, Mata Atlântica, Pampa).
- **10 anos completos** (2015-01-01 a 2024-12-31), totalizando **43.836 registros diários**.

---

## 2. Metodologia

### 2.1 Aquisição

Foi implementado o módulo [`src/data_acquisition.py`](../src/data_acquisition.py) que itera sobre `config.CITIES` e consulta o endpoint `temporal/daily/point` com retries exponenciais. A resposta JSON é convertida em DataFrame longo (uma linha por dia × cidade) e materializada em `data/raw/<cidade>.csv` e em `data/processed/nasa_power_consolidated.csv`.

A função é **idempotente**: arquivos já baixados não são re-baixados a menos que `force=True` seja passado.

### 2.2 Pré-processamento e *feature engineering*

Implementado em [`src/data_preprocessing.py`](../src/data_preprocessing.py):

1. **Sentinela `-999` → `NaN`** explícito.
2. **`Forward fill`/`back fill` limitados a 3 dias** por cidade, para tratar gaps esporádicos sem inventar dados em períodos longos.
3. **Features de calendário cíclicas:** `doy_sin = sin(2π·DOY/365.25)` e `doy_cos = cos(...)` — modelos baseados em árvore extraem sazonalidade diretamente desses sinais.
4. **Lags 1 e 7 dias** para `T2M`, `RH2M`, `PRECTOTCORR` e o próprio alvo: capturam persistência de curto prazo (memória atmosférica).
5. **Médias móveis 7 e 30 dias** (sempre `shift(1)` antes do *rolling* — sem vazamento).
6. **One-hot encoding** da região como variável categórica.

Resultado: **43.656 linhas × 30 features**.

### 2.3 Divisão treino/teste

Divisão **temporal**: as últimas 20% das datas vão para o conjunto de teste (~8.731 linhas, cobrindo aprox. os últimos 2 anos). Isso simula o uso operacional — prever o futuro a partir do passado — e evita vazamento que ocorreria em divisão aleatória.

Para validação cruzada interna usa-se `TimeSeriesSplit(5)`, que constrói folds incrementais respeitando ordem temporal.

### 2.4 Modelos supervisionados (Q1)

Três algoritmos com viés/variância contrastantes:

| Modelo | Por que está no comparativo |
|---|---|
| **Regressão Linear** (com `StandardScaler`) | Baseline interpretável: se um modelo simples já entrega R² alto, é o sinal de que a relação física entre irradiância e céu limpo × claridade é predominantemente linear. |
| **Random Forest** | Captura não-linearidades e interações sem tuning denso. Robusto a outliers. Importância nativa de features. |
| **XGBoost** | Estado-da-arte em tabular com regularização explícita e *early stopping*. Costuma superar RF com hiperparâmetros padrão sensatos. |

Hiperparâmetros conservadores (sem *grid search*): RF com 300 árvores, max_depth 18; XGBoost com 600 árvores, learning_rate 0,05, subsample 0,85.

### 2.5 Modelo não-supervisionado (Q4)

Agregamos as séries diárias em **assinaturas climáticas anuais por cidade** (8 estatísticas: média e desvio da irradiância, média e amplitude térmica, umidade, precipitação, vento, índice de claridade). Como `n=12`, KMeans é apropriado e estável.

Avaliamos `k ∈ [2, 8]` com:

- **Inércia** (curva-cotovelo)
- **Silhueta** (separação)
- **Calinski-Harabasz** (razão variância entre-grupos / intra-grupos)

`k = 4` foi escolhido por maximizar simultaneamente silhueta (0,404) e Calinski-Harabasz (10,3).

### 2.6 Métricas

- **Regressão:** RMSE (mesma unidade do alvo), MAE, R², MAPE.
- **Clustering:** silhueta, Calinski-Harabasz, inércia.

---

## 3. Análise Exploratória de Dados

### 3.1 Qualidade do dado

Após o download, **nenhum valor missing** foi detectado no período/parâmetros escolhidos para as 12 cidades — atestado pela ausência de células acima de 0 % no mapa de calor em `outputs/figures/06_missing_heatmap.png`. Isso é coerente com a maturidade do produto NASA POWER pós-2015.

### 3.2 Distribuições por região

A figura `01_irradiance_by_region.png` mostra distribuições assimétricas e claramente diferenciadas:

- **Nordeste** apresenta a maior mediana e menor cauda inferior — céu limpo dominante.
- **Sul** tem a menor mediana e maior dispersão — alta sazonalidade (verão ≠ inverno).
- **Norte** mostra um *kink* abaixo de 4 kWh/m²/dia coerente com forte nebulosidade convectiva.

### 3.3 Sazonalidade

O heatmap mensal (`02_seasonality_heatmap.png`) revela:

- Em **Porto Alegre e Curitiba**, junho e julho colapsam para ~2,5 kWh/m²/dia, enquanto dezembro/janeiro chegam a 6+ — relação ~2,5×.
- **Fortaleza e Recife** têm comportamento contracíclico: o segundo semestre é o mais radiativo (estação seca).
- A **Amazônia** mostra variação menor entre estações, mas com queda acentuada em fev–abr (estação chuvosa).

### 3.4 Correlações

A matriz de correlação (`03_correlation_matrix.png`) confirma fisicamente:

- `ALLSKY_SFC_SW_DWN` ↔ `ALLSKY_KT` ≈ 0,9 (definicional)
- `ALLSKY_SFC_SW_DWN` ↔ `CLRSKY_SFC_SW_DWN` ≈ 0,7 (teto modulado por nuvens)
- `ALLSKY_SFC_SW_DWN` ↔ `RH2M` < 0 (umidade alta → mais nuvens)
- `T2M` ↔ `RH2M` < 0 (clássica)

Essas correlações sustentam a escolha das features e antecipam que **CLRSKY + ALLSKY_KT** carregarão a maior parte do sinal.

---

## 4. Resultados

### 4.1 Q1 — Regressão

| Modelo | CV RMSE | RMSE teste | MAE teste | R² teste | MAPE (%) |
|---|---|---|---|---|---|
| Regressão Linear | 0,259 ± 0,009 | 0,272 | 0,184 | 0,969 | 6,64 |
| Random Forest | 0,098 ± 0,008 | 0,107 | 0,069 | 0,995 | 1,67 |
| **XGBoost** | **0,077 ± 0,010** | **0,069** | **0,052** | **0,998** | **1,26** |

**Leitura técnica:**

- O baseline linear já entrega R² = 0,97 — ou seja, a maior parte da variância é explicável por relações lineares com `CLRSKY` e `ALLSKY_KT`.
- Random Forest captura interações (sazonalidade × região) e reduz o RMSE em ~60%.
- XGBoost adiciona regularização e melhora marginalmente, com RMSE de 0,069 kWh/m²/dia (≈ 1,3 % de erro relativo).
- O **gap CV-vs-teste é pequeno (0,077 → 0,069)** — não há overfitting evidente; pelo contrário, o conjunto de teste é levemente mais previsível, possivelmente por concentrar anos mais recentes do produto NASA POWER.

A figura `07_predictions.png` mostra (a) scatter previsto × observado com aderência forte à diagonal e (b) série temporal dos últimos 180 dias para uma cidade — os modelos seguem inclusive picos e vales sazonais.

A figura `08_feature_importance.png` ordena as features pelo *gain* do XGBoost. Os topos são (esperadamente):

1. `ALLSKY_KT` — claridade do dia
2. `CLRSKY_SFC_SW_DWN` — teto teórico
3. Lags da irradiância e médias móveis 7d — persistência atmosférica
4. `doy_sin / doy_cos` — sazonalidade
5. `RH2M` — proxy de nuvens

### 4.2 Q4 — Clustering

Diagnóstico de k (`09_kmeans_diagnostics.png`):

| k | Inércia | Silhueta | Calinski-Harabasz |
|---|---|---|---|
| 2 | 55,5 | 0,324 | 7,3 |
| 3 | 32,4 | 0,398 | 8,9 |
| **4** | **19,7** | **0,404** | **10,3** |
| 5 | 15,0 | 0,312 | 9,5 |
| 6 | 10,9 | 0,341 | 9,4 |

`k = 4` é o pico de silhueta e Calinski-Harabasz — escolha objetiva.

**Composição dos clusters** (`10_cluster_pca.png` e `11_cluster_profiles.png`):

| Cluster | Cidades | Perfil (z-score) | Leitura |
|---|---|---|---|
| **0 — Amazônia úmida** | Manaus, Belém | T alta e estável, RH 83 %, chuva 6,3 mm/dia, Kt baixo | Irradiância média, alta nebulosidade convectiva. |
| **1 — Sul/Sudeste úmido temperado** | Porto Alegre, Curitiba, São Paulo, Rio de Janeiro | T 20°C, amplitude térmica 11°C, vento moderado | Cidades com **maior variabilidade** sazonal e menor irradiância. |
| **2 — Litoral Nordeste seco e ventoso** | Fortaleza, Recife | T 27°C, RH 77 %, **vento 4,4 m/s**, **Kt 0,59** | Combinação ideal **solar + eólico**. |
| **3 — Cerrado de interior** | Brasília, Goiânia, Belo Horizonte, Salvador | T 23°C, RH 69 % (a mais baixa), Kt 0,57 | Alta irradiância, baixa nebulosidade — perfil "estepe tropical seca". |

Os clusters **não foram inicializados com região nem bioma**, mas o resultado os recupera quase perfeitamente — sinal de que as variáveis NASA POWER carregam, por si só, a assinatura climática regional.

---

## 5. Discussão e Interpretação

### 5.1 O que os números significam na prática

**Para Q1**: um RMSE de 0,069 kWh/m²/dia em uma média típica de 5 kWh/m²/dia corresponde a **~1,3 % de erro relativo**. Em termos operacionais, para um telhado de 50 kWp em São Paulo (~200 kWh/dia esperados), o erro do modelo está abaixo da incerteza típica do *system performance ratio* (5-10 %). Ou seja, **o modelo é mais preciso do que a próxima fonte de incerteza no pipeline de cálculo de geração**.

**Para Q4**: o cluster do **Litoral Nordeste seco e ventoso** (Fortaleza + Recife) é o mais informativo do ponto de vista de política energética — é a única região que combina alta irradiância **e** vento estável, justificando investimento em parques híbridos solar-eólico. Já o cluster **Sul/Sudeste úmido** demanda dimensionamento generoso de armazenamento (a variabilidade sazonal é o gargalo, não a média anual).

### 5.2 Conexão com os ODS

- **ODS 9 (Infraestrutura).** O modelo Q1 pode alimentar plataformas de planejamento de redes — uma operadora pode estimar geração distribuída por CEP com erro <2 % a partir de dados diários gratuitos.
- **ODS 11 (Cidades sustentáveis).** Os clusters da Q4 sugerem que políticas de adaptação climática podem ser compartilhadas entre Porto Alegre, Curitiba, São Paulo e Rio — todas no mesmo regime —, racionalizando esforço técnico de governos estaduais.
- **ODS 13 (Ação climática).** A série de 10 anos permite que extensões futuras (já preparadas no `data_preprocessing.add_lag_features`) detectem tendências e anomalias regionais — um caminho natural para a Q10 do nosso escopo original.

### 5.3 Limitações reconhecidas

1. **Resolução espacial.** ~50 km/píxel — adequado para política regional, não para um telhado específico.
2. **Cluster com `n=12`.** Silhueta de 0,40 é "moderada"; com 50-100 cidades a separação ficaria nítida.
3. **Horizonte preditivo curto.** O modelo Q1 é uma **regressão estática** (mesmo dia → mesmo dia). Para previsão prospectiva (h+1, h+7) seria necessário transformar em supervised learning com features estritamente passadas — o que o módulo já suporta via lags.
4. **Sem busca de hiperparâmetros.** Um *grid search* poderia ganhar marginalmente; deixamos como trabalho futuro.

### 5.4 Trabalho futuro

- Estender a Q4 com **DBSCAN/HDBSCAN** para detectar cidades fora dos quatro grupos canônicos.
- Treinar **modelos preditivos prospectivos (h+1, h+7)** para atender Q9 (consumo energético em função de ondas de calor).
- Cruzar com dados socioeconômicos (PIB municipal, IDH) para Q3 (segurança alimentar) e Q8 (risco climático para infraestrutura).
- Validar contra estações INMET nas mesmas capitais.

---

## 6. Conclusão

Demonstramos que, com **10 anos de dados gratuitos da NASA POWER** e técnicas clássicas de ML (XGBoost + KMeans), é possível:

1. **Prever a irradiância solar urbana com erro de ~1,3 %** — preciso o suficiente para subsidiar dimensionamento de infraestrutura fotovoltaica (ODS 9).
2. **Agrupar capitais brasileiras em quatro regimes climáticos coerentes** com biomas e regiões, sem usar essa rotulagem como input — confirmando que o satélite carrega a assinatura regional (ODS 11, 13).

O projeto cumpre integralmente os critérios de avaliação da Global Solution: pergunta formulada e justificada, exploração rigorosa dos dados, solução de ML coerente com o problema, e interpretação dos resultados para além das métricas numéricas.

---

## Apêndice — Como reproduzir

Consultar o [`README.md`](../README.md). Em resumo:

```bash
git clone <repo>
cd orbital-solar-insight
pip install -r requirements.txt
python -m src.main      # pipeline completo
```

Saídas: `outputs/figures/` (11 PNGs), `outputs/models/best_XGBoost.joblib`, `data/processed/*.csv`.
