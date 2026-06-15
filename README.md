# LLM Demand Simulation Code

This repository contains the code for the paper:

Huang, Chengpiao and Wang, Kaizheng. (2026). LLM-powered virtual population for demand simulation and pricing.

## Structure

```text
demand_sim/
  config.py              Shared paths and constants
  data.py                Loading/parsing utilities
  preprocessing.py       Product/persona/query preprocessing
  embeddings.py          SigLIP product/persona embedding generation
  llm.py                 OpenAI simulation runner
  metrics.py             Distributional fit metrics
  evaluation.py          Distributional evaluation pipeline
  models/
    llm_mix.py           LLM persona mixture
    llm_mix_cal.py       Calibrated LLM persona mixture
    emb.py               Persona/product embedding mixture
    gaussian.py          Rounded Gaussian demand model
scripts/
  build_products.py
  build_personas.py
  build_queries.py
  build_product_embeddings.py
  build_persona_embeddings.py
  run_llm_simulations.py
  evaluate_demand_prediction.py
  evaluate_pricing.py
  make_illustrative_use_case.py
data/                   Local Kaggle CSV files, not committed
images/                 Local Kaggle product images, not committed
outputs/
  products/
  personas/
  query_plan/
  responses/
  embeddings/
  demand_prediction/
  pricing/
  illustrative use case/
```

## Data

The project uses the H&M Personalized Fashion Recommendations dataset from Kaggle:

https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations

The downloaded Kaggle files are not included in this repository. After accepting Kaggle's terms and downloading the competition data, place the required files under the repository root as follows:

```text
data/
  articles.csv
  customers.csv
  transactions_train.csv
images/
  <article_id>.jpg
```

The source Kaggle image archive is organized in nested folders by article-id prefix. This code expects a flat `images/` directory where each image is named by article id, for example `images/554450004.jpg`. The current workflow only needs images for the selected product set, which defaults to the top 100 online trousers built by `scripts/build_products.py`.

## Typical Workflow

From the repository root:

```bash
python3 scripts/build_products.py
python3 scripts/build_personas.py
python3 scripts/build_queries.py
python3 scripts/build_product_embeddings.py --allow-download
python3 scripts/build_persona_embeddings.py --allow-download
OPENAI_API_KEY=... python3 scripts/run_llm_simulations.py
python3 scripts/evaluate_demand_prediction.py
python3 scripts/evaluate_pricing.py
python3 scripts/make_illustrative_use_case.py --split 6 --article-id 554450004
```

The embedding scripts use `google/siglip2-base-patch16-224` by default. Omit `--allow-download` after the model is already cached locally.

## Models

The cleaned model set is:

- `llm-mix`: binomial mixture over LLM persona purchase probabilities.
- `llm-mix-cal`: monotone logit calibration followed by the same persona mixture.
- `emb`: binomial mixture whose persona-level probabilities are logistic functions of product embeddings, persona embeddings, and price.
- `gaussian`: rounded Gaussian distributional regression for aggregate demand, fit by closed-form ridge regression.

Distributional evaluation reports zero-truncated metrics against real positive-demand observations: NLL, CRPS, randomized PIT KS, interval scores, MAE, and RMSE. For the three binomial models, MAE and RMSE use the zero-truncated conditional mean. For `gaussian`, they use the fitted Gaussian mean, while distributional scores use the rounded integer PMF conditioned on positive demand.

For repeated product-level train/test evaluation, run:

```bash
python3 scripts/evaluate_demand_prediction.py --n-splits 10
```

This splits products 60:40 into train/test for each split, refits the requested models on the training products, evaluates both in-sample and out-of-sample distributional fit. If the output folder already contains completed splits, `--n-splits` adds that many new splits and then rebuilds the aggregate summaries over all completed splits. For the three binomial models, it searches over `--exposure-n-values 100 150 200 250` by default and records the selected value in `model_fit_summary_all_splits.csv`. It also writes `summary_mean_metrics_train.csv` and `summary_mean_metrics_test.csv`, with models as rows and mean metrics as columns.

To recompute metrics from saved fitted models without refitting, run:

```bash
python3 scripts/evaluate_demand_prediction.py --reevaluate-existing
```

## Pricing Sample Efficiency

The pricing experiment is implemented in `scripts/evaluate_pricing.py`. It uses the `llm-mix-cal` model to test how many synthetic training observations are needed to learn a good pricing policy.

For each split:

1. Products are split into `J1:J2:J3 = 60:25:15`.
2. `J1` fits a zero-truncated `llm-mix-cal` ground truth model `Q*`.
3. `Q*` generates untruncated synthetic demand on `J2` and `J3`.
4. Fractions of `J2` fit an estimated model `Qhat` using the untruncated objective.
5. `Qhat` is evaluated on `J3` by expected revenue and `CVaR_0.25`.

Run or refresh:

```bash
python3 scripts/evaluate_pricing.py
python3 scripts/evaluate_pricing.py --refresh-existing
```

Main outputs are written under `outputs/pricing/`:

- `pricing_mean_performance_ratio_by_fraction.csv`
- `pricing_mean_regret_by_fraction.csv`
- `sample_efficiency_expected_revenue.pdf`
- `sample_efficiency_cvar_0p25.pdf`


## Illustrative Use Case Figures

`scripts/make_illustrative_use_case.py` loads a saved pricing ground-truth model and generates paper-style demand-distribution and pricing-curve figures. The current illustrative product is `554450004` in split `006`.

Example:

```bash
python3 scripts/make_illustrative_use_case.py \
  --split 6 \
  --article-id 554450004 \
  --demand-prices 16.99
```

Outputs are written under `outputs/illustrative use case/`. Demand-distribution figures are one PDF per price, with light-blue PMF bars and a blue interpolated curve. Pricing figures are one PDF per objective, with the optimal price highlighted in red.
