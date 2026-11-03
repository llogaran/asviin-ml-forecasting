# SPEC — Authoritative build contract for `asviin-ml-forecasting`

> **Audience:** all build agents (and the coordinator). Read this before writing code.
> **Rule:** if this document and your intuition disagree, this document wins. If this
> document is ambiguous, choose the simplest defensible option, document it in the
> module docstring, and report the deviation in your final message.

---

## 0. Context and guardrails

- The project is an **AI/MLOps extension of Asviin**, a real ERP (Vue 3 frontend, Firebase
  Cloud Functions backend, Firestore NoSQL). Asviin v2 lives at
  `/home/llogaran/proyectos/soberanaia-nubes/asviin` and
  `/home/llogaran/proyectos/soberanaia-nubes/functions/asviinV2`. You may **read** it for
  schema fidelity, never modify it.
- **Data is 100% synthetic.** Never copy, reference, export or embed real client data,
  backups, credentials or `.env` files. The synthetic generator is itself a deliverable.
- Business context: multi-branch ERP for SME plastic recyclers/transformers in **Mexico**,
  currency **MXN**, timezone **America/Mexico_City**.
- The ML layer must stay **decoupled** from the ERP: CSV data contract + API. No imports
  from Asviin, no Firestore access in this repo.
- Code, docstrings, comments, docs and commit-ready files are written in **English**.
  The user-facing chat summary is Spanish; that is the coordinator's job, not yours.
- Repository root: `/home/llogaran/proyectos-dsh/asviin/asviin-ml-forecasting`.
  All paths below are relative to it.

## 1. Repository map and ownership

```
data/raw/               products.csv, branches.csv, recipes.csv,
                        economic_events.csv, stock_movements.csv, pending_sales.csv, seed.json
data/processed/         demand_daily.csv, data_quality.json
notebooks/              01_eda.ipynb, 02_modeling.ipynb, 03_explainability.ipynb
src/config.py           DONE — do not modify
src/data_contracts/     DONE — do not modify
src/pipelines/          synthetic_generator.py, etl.py, features.py, train.py,
                        forecast.py, impact.py
src/api/                __init__.py, app.py, models.py, service.py
src/monitoring/         __init__.py, drift.py
src/agent/              __init__.py, agent.py, cli.py
tests/                  one test module per area
artifacts/models/       xgboost_global.joblib, registry.json
artifacts/reports/      backtest.json, backtest.md, impact.json
artifacts/monitoring/   drift_report.json (generated example)
Dockerfile, .dockerignore, .github/workflows/ci.yml
README.md, docs/ARCHITECTURE.md, docs/DATA_CONTRACT.md, docs/DECISIONS.md
```

**Ownership:** each agent writes only its assigned files (see prompts). Shared files
(`pyproject.toml`, `Makefile`, `src/config.py`, `src/data_contracts/*`, `docs/SPEC.md`)
are **owned by the coordinator**. Never edit them; report needed changes instead.

## 2. Environment and commands

- Python **3.12** virtualenv already provisioned at `.venv/`. Always use
  `.venv/bin/python` and `.venv/bin/ruff`. Never create another venv; never install
  packages from the internet (offline build).
- Run commands from the repository root.
- Tests: `.venv/bin/python -m pytest tests/test_<area>.py -q -p no:cacheprovider`.
- Make targets (already defined, do not change):
  - `make data` / `make data-quick`
  - `make train` / `make train-quick`
  - `make test`, `make lint`, `make serve`, `make monitor`, `make agent Q="..."`
- CLI invocations required by the Makefile:
  - `python -m src.pipelines.synthetic_generator --out DIR --days N --products N --branches N --seed N`
  - `python -m src.pipelines.etl --raw DIR --out PATH`
  - `python -m src.pipelines.train --data DIR --artifacts DIR [--quick]`
  - `python -m src.monitoring.drift --data PATH --out PATH [--inject-drift]`
  - `python -m src.agent.cli "QUESTION"`

## 3. Canonical data contracts (CSV)

All files: UTF-8, comma-separated, header row, ISO-8601 dates, floats with up to 2
decimals, no index column. The authoritative column order and validation live in
`src/data_contracts/schemas.py` and must not change.

### 3.1 `data/raw/products.csv`
| column | type | rules |
|---|---|---|
| product_id | str | unique, `P-001` style |
| name | str | realistic Spanish product name |
| category | str | e.g. Materia Prima, Producto Terminado, Insumos, Consumibles |
| unit | str | pza, kg, l, ml |
| price_mxn | float | > 0 |
| cost_mxn | float | >= 0, 55–80% of price |
| is_ingredient | bool 0/1 | ingredients are mostly inputs, not sold directly |
| has_recipe | bool 0/1 | at least 5 sellable products must have a recipe |
| safety_stock | float | >= 0 |
| lead_time_days | int | 3–21 |
| demand_pattern | str | smooth, intermittent, lumpy |
| abc_class | str | A, B, C (roughly 20/30/50 of sellable volume) |

### 3.2 `data/raw/branches.csv`
`branch_id`, `business_id`, `name`, `region`. Exactly 3 branches for the full run:
`B-01` Matriz Centro, `B-02` Sucursal Norte, `B-03` Bodega Sur, with volume multipliers
≈ 1.0 / 0.62 / 0.38.

### 3.3 `data/raw/recipes.csv`
`recipe_id`, `business_id`, `output_product_id`, `input_product_id`,
`quantity_per_unit` (> 0). BOM lines for the `has_recipe` products; inputs must be
`is_ingredient` products.

### 3.4 `data/raw/economic_events.csv`
One row per event **line**. `venta` rows are item-level (`event_id` repeated across the
lines of the same ticket); `gasto` rows may have empty product columns.

Columns: `event_id`, `business_id`, `branch_id`, `event_type` (venta|compra|gasto|merma|
produccion|transferencia), `event_ts` (ISO-8601 with offset), `amount_mxn` (line amount
for venta; event amount otherwise), `payment_method` (efectivo|transferencia|tarjeta|
credito, empty for non-sales), `customer_id` (empty unless credit), `product_id`,
`product_name`, `quantity` (sale line units), `unit_price_mxn`, `unit_cost_mxn`,
`is_cancelled` (0/1, only meaningful for venta), `note`.

### 3.5 `data/raw/stock_movements.csv`
`movement_id`, `business_id`, `branch_id`, `product_id`, `movement_ts`, `direction`
(entrada|salida), `reason` (venta_manual|compra|ajuste_manual|merma|produccion|
transferencia_entrante|transferencia_saliente), `quantity` (> 0), `unit_cost_mxn`,
`stock_after` (>= 0, closing stock for that product/branch after the movement), `note`.

### 3.6 `data/raw/pending_sales.csv`
`pending_sale_id`, `business_id`, `branch_id`, `product_id`, `created_ts`,
`promised_ts` (nullable), `quantity` (> 0), `unit_price_mxn`, `status`
(pendiente|convertida|cancelada), `converted_event_id` (nullable; set when convertida).

### 3.7 `data/processed/demand_daily.csv`
One row per `(date, business_id, branch_id, product_id)` for every calendar day in the
raw date range, zero-filled when there is no sale. Columns exactly as in
`schemas.CSV_COLUMNS["demand_daily"]`:

`date`, `business_id`, `branch_id`, `product_id`, `category`, `unit`, `units_sold`,
`revenue_mxn`, `cost_mxn`, `cancelled_units`, `pending_units`, `stock_on_hand`,
`stockout_flag` (0/1), `promo` (0/1), `unit_price_mxn`, `unit_cost_mxn`,
`lead_time_days`, `demand_pattern`, `abc_class`.

Semantics:
- `units_sold`, `revenue_mxn`, `cost_mxn`: only non-cancelled sale lines.
- `cancelled_units`: sum of `quantity` on cancelled sale lines (excluded from sales).
- `pending_units`: quantity of pending sales **created that day** (status != cancelada),
  i.e. committed demand not yet invoiced.
- `stock_on_hand`: closing stock for the product/branch that day (last `stock_after`
  of the day, forward-filled; 0 before first movement).
- `stockout_flag`: 1 when closing stock <= 0.
- `promo`: 1 when any sale line that day had `unit_price_mxn < 0.85 * products.price_mxn`.
- `unit_price_mxn`, `unit_cost_mxn`: weighted averages of sale lines that day; fall back
  to the product master when there are no sales.

`data/processed/data_quality.json` keys (at minimum): `rows`, `series`, `date_min`,
`date_max`, `zero_sale_share`, `negative_stock_rows`, `missing_product_refs`,
`duplicate_keys`, `cancelled_units_total`, `pending_units_total`.

## 4. Module interfaces (exact public API)

### 4.1 `src/pipelines/synthetic_generator.py`
```python
def generate_dataset(out_dir: Path | str, *, days: int = 1096, n_products: int = 20,
                     n_branches: int = 3, seed: int = 42,
                     start: str = "2023-01-01") -> dict[str, Path]
```
Writes `raw/` files (3.1–3.6) + `raw/seed.json` (deterministic params only), returns path
map. Requirements: deterministic per seed (byte-identical CSVs across runs); realistic
seasonality (weekly Thu–Sat uplift, December peak ≈1.4×, May dip ≈0.85×, March–April
plastic-recycling peak), trend ≈ +3%/quarter, ~8% promo days with 10–25% discounts,
2% cancelled sale lines, pending sales ≈10% of demand, intermitency by pattern,
zero stockouts handled by capping sales at available stock. Production/BOM simulation is
optional; stock integrity (`stock_after` never negative, FK valid) is mandatory.
`seed.json` must not contain timestamps or run-specific values.

### 4.2 `src/pipelines/etl.py`
```python
def build_daily_demand(raw_dir: Path | str) -> pd.DataFrame
def run_etl(raw_dir: Path | str, out_path: Path | str) -> pd.DataFrame  # writes CSV + quality JSON
```
Uses `src.data_contracts` validators. Aggregates events into the daily grain, zero-fills,
computes stock/promo/pending per 3.7, writes `data_quality.json` next to `out_path`.
Must reject invalid inputs with `SchemaValidationError` / clear errors.

### 4.3 `src/pipelines/features.py`
```python
FEATURES: list[str]
def build_supervised(daily: pd.DataFrame, holdout_days: int = 28
                     ) -> tuple[pd.DataFrame, pd.DataFrame, dict]
def features_for_next_day(history: pd.DataFrame, meta: dict) -> pd.Series
```
- Sort by `(business_id, branch_id, product_id, date)`; target `y = log1p(units_sold)`.
- Holdout = last `holdout_days` calendar days per series; training rows exclude the first
  28 days of each series and the holdout.
- `meta` must contain `features`, `encoders` (dict of category → int), `target_transform`.
- `FEATURES` (exact names, all numeric after encoding):
  `lag_1, lag_7, lag_14, lag_28, roll_mean_7, roll_mean_14, roll_mean_28, roll_std_7,
  days_since_last_sale, dow, month, is_weekend, sin_doy, cos_doy, promo_lag1,
  stockout_flag_lag1, pending_units, unit_price_mxn, unit_cost_mxn, margin_pct,
  lead_time_days, stock_to_demand, product_code, branch_code, category_code, unit_code,
  demand_pattern_code`
- No target leakage: every lag/rolling feature is shifted by at least 1 day.
- `features_for_next_day` produces one feature row from the latest history (used by the
  recursive forecaster). It must tolerate zero-history series (fill with zeros/first values).

### 4.4 `src/pipelines/train.py` and `src/pipelines/impact.py`
```python
def train(data_dir: Path | str, artifacts_dir: Path | str, quick: bool = False) -> dict
```
- Loads `data_dir/processed/demand_daily.csv`, builds supervised data (holdout 28 days;
  quick: 14 days), trains three XGBoost quantile regressors (`p10`, `p50`, `p90`,
  objective `reg:quantileerror`, `quantile_alpha` = 0.1/0.5/0.9, `tree_method="hist"`,
  `random_state=42`, small `n_estimators` when quick).
- Baselines on the holdout: `naive_seasonal` (same weekday previous week) and
  `croston_sba` (implementation in train.py; deterministic, documented formula).
- Metrics on holdout, units scale: `mae`, `wape`, `mase` (vs naive_seasonal),
  `coverage_90` (share of actual inside [p10, p90]), `n_series`, `holdout_days`.
- Writes `artifacts/models/xgboost_global.joblib` = `{"models": {"p10":..., "p50":...,
  "p90":...}, "meta": meta}` and `artifacts/models/registry.json` (see §5), plus
  `artifacts/reports/backtest.json`, `backtest.md`, `impact.json`.
- `impact.py` simulates a simple periodic-review policy on the holdout: forecast-driven
  (planning signal = 0.7·p50 + 0.3·p90, median-bias corrected) vs naive-driven
  replenishment; report average on-hand units, stockout days, service level, and relative
  improvement. Keep the simulation deterministic and documented.
- Returns the registry dict.

### 4.5 `src/pipelines/forecast.py`
```python
class Forecaster:
    @classmethod
    def load(cls, data_dir: Path | str, artifacts_dir: Path | str) -> "Forecaster": ...
    def forecast(self, product_id: str, branch_id: str, horizon_days: int = 28) -> dict
    def recommend(self, product_id: str, branch_id: str, horizon_days: int = 30,
                  service_level: float = 0.95) -> dict
    def products(self) -> list[dict]
    def branches(self) -> list[dict]
    def model_info(self) -> dict
```
- Recursive one-step forecasting using `features_for_next_day`; p10/p50/p90 per day.
  The recursive state advances with the planning signal (`0.7·p50 + 0.3·p90`) and
  keeps stock flat: simulating stock-out days inside the forecast would censor
  future demand and collapse long horizons.
- `forecast` returns `{"product_id", "branch_id", "model", "model_version",
  "generated_at", "horizon_days", "points": [{"date", "p50", "p10", "p90"}]}` with
  non-negative rounded values.
- `recommend` uses the horizon forecast. Planning demand is
  `0.7 * p50 + 0.3 * p90` per day (median-bias correction for zero-inflated
  series): daily mean μ is the average of that signal, uncertainty
  σ ≈ mean((p90 − p10) / 2.562) over horizon, lead time L from the product, z from
  the normal quantile of `service_level`, `safety_stock = z * σ * sqrt(L)`,
  `reorder_point = μ_daily * L + safety_stock`, `current_stock` from the last
  `stock_on_hand`, `reorder_quantity = max(0, reorder_point - current_stock)`.
  Returns those numbers plus `days_of_cover`, `rationale`, `model_version`.
- Unknown product/branch raises `KeyError`. Invalid horizon (< 1 or > 90) raises `ValueError`.

### 4.6 `src/monitoring/drift.py`
```python
def detect_drift(reference, current, features: list[str] | None = None) -> dict
```
- Numeric features default: `units_sold, revenue_mxn, pending_units, stock_on_hand,
  unit_price_mxn, promo, stockout_flag`.
- Per feature: PSI (10 quantile bins from reference), two-sample KS (`scipy.stats`),
  status: `drift` if PSI >= 0.2 or KS p < 0.01; `warning` if PSI >= 0.1 or p < 0.05;
  else `ok`.
- Overall `status` = `drift` if ≥30% features drift, else `warning` if any warning/drift,
  else `ok`; include `drift_share`, thresholds, row counts, timestamp.
- CLI: `--data data/processed/demand_daily.csv --out artifacts/monitoring/drift_report.json
  [--mode seasonal|chronological] [--window-days 28] [--lag-days 364] [--inject-drift]`.
  The default is a **seasonal split**: compare the last 28 days with the same calendar
  window 364 days earlier (preserves weekday alignment), so normal annual seasonality is
  not mistaken for drift. If the file is shorter than a year, the CLI falls back to the
  chronological 70/30 split and records `window_mode` in the report. `--inject-drift`
  multiplies the current `units_sold` by 1.6 to demonstrate detection. Prints a one-line
  summary.

### 4.7 `src/api/`
Files: `__init__.py`, `models.py` (Pydantic request/response models), `service.py`
(`ForecastService` with `list_products`, `forecast`, `recommend`, `model_info`, lazy
loading, injectable data/artifacts dirs), `app.py` (`create_app(service=None) -> FastAPI`,
module-level `app`).
Endpoints: `GET /health`, `GET /ready`, `GET /model`, `GET /products?q=`,
`POST /forecast`, `POST /recommendation`, `GET /metrics`.
- Auth: if `settings.api_key` is set, every endpoint except `/health` requires
  `X-API-Key` (401 otherwise). CORS from `settings.cors_origins`.
- Errors: unknown product/branch → 404 with clear Spanish-friendly English message;
  validation → 422; missing model → 503.
- `/metrics`: JSON counters (`requests_total`, `forecast_requests`,
  `recommendation_requests`, `errors_total`, `uptime_seconds`).
- Request/response examples must match README and Pydantic models.

### 4.8 `src/agent/`
Files: `__init__.py`, `agent.py` (`def answer(question: str, forecaster=None) -> dict`),
`cli.py` (`python -m src.agent.cli "..."`).
- Offline deterministic mode (default, no network): normalize accents, detect intent
  (recommend vs forecast), product by name (difflib fuzzy ≥ 0.55), branch by id/name
  (default to highest-volume branch when unambiguous), horizon by regex
  (`días/semanas`), call `Forecaster.recommend/forecast`, compose a concise Spanish
  answer quoting exact numbers and the model version. Missing product → clarification
  listing candidates. Return `{"answer", "mode": "offline", "tool_calls": [], "data"}`.
- LLM mode when `OPENAI_API_KEY` is set: OpenAI-compatible chat completion with tool
  definitions (`list_products`, `forecast`, `recommendation`) executed locally against
  the forecaster; max 3 tool rounds; on any failure fall back to offline and set
  `mode="offline_fallback"`. No new dependencies beyond `httpx` (already installed).
- Tests use offline mode only and must not hit the network.

## 5. Artifacts

`artifacts/models/registry.json`:
```json
{
  "schema_version": "1.0.0",
  "generated_at": "<UTC ISO>",
  "champion": "xgboost_global",
  "dataset": {"path": "data/processed/demand_daily.csv", "sha256": "...", "rows": 0,
               "series": 0, "date_min": "...", "date_max": "..."},
  "models": {
    "xgboost_global": {
      "artifact": "xgboost_global.joblib",
      "trained_at": "<UTC ISO>",
      "holdout_days": 28,
      "metrics": {"mae": 0.0, "wape": 0.0, "mase": 0.0, "coverage_90": 0.0,
                   "n_series": 0},
      "baselines": {"naive_seasonal": {"mae": 0.0, "wape": 0.0},
                     "croston_sba": {"mae": 0.0, "wape": 0.0}}
    }
  }
}
```
`backtest.json` mirrors the metrics plus a small per-series sample.

## 6. Coding standards

- Python 3.11+ syntax, type hints on every public function, short docstrings explaining
  *why* when the logic is non-obvious.
- `logging.getLogger(__name__)`; never `print` in library modules (CLIs may print).
- No new third-party dependencies. Available: fastapi, uvicorn, pydantic, pandas, numpy,
  scikit-learn, xgboost, joblib, pyyaml, pytest, httpx, ruff. Optional imports (`shap`,
  `statsmodels`, `matplotlib`) must be guarded with try/except and degrade gracefully.
- No global mutable state except module-level `app` required by uvicorn.
- Determinism: fixed seeds (`MLFC_RANDOM_SEED`, default 42); no `datetime.now()` inside
  data generation; timestamps allowed only in artifacts/reports.
- No secrets, tokens or client data anywhere.
- Ruff clean (`ruff check src tests`) with the project config.

## 7. Testing requirements

- One test file per area, **no network**, **no reliance on committed artifacts** where a
  temp fixture is cheap (API/agent tests may load a tiny model trained in a fixture).
- Contract tests: valid CSVs pass, malformed ones fail with useful messages, FK checks.
- Generator tests: determinism (two runs byte-identical), schema validity, date range,
  zero share bounds, seasonality smoke, no negative stock.
- ETL tests: grain uniqueness, zero-fill, cancellation exclusion, stock carry-forward,
  quality report keys.
- Features tests: no leakage (correlation/shift check), holdout disjoint, no NaNs.
- Training tests: quick mode runs on a tiny dataset, registry/ artifacts written, metrics
  present.
- API tests: health, 401 with key, 404 unknown product, 422 invalid horizon, forecast
  shape, recommendation math.
- Drift tests: stable split → not `drift`; injected shift → drift detected.
- Agent tests: recommendation intent, unknown product clarification, no network.
- Keep the whole suite under ~90 seconds on a laptop and under ~2 minutes in CI.

## 8. Definition of done (per agent)

1. Assigned files exist and match the public API above.
2. `ruff check` on your files is clean.
3. Your tests pass with `.venv/bin/python -m pytest <your tests> -q -p no:cacheprovider`.
4. No edits outside your ownership; no new dependencies; no secrets.
5. Final message: files, commands + results, deviations/blockers (≤ 15 lines).

## 9. Integration notes

- The ETL is the only writer of `data/processed/`; everything downstream reads
  `demand_daily.csv` through `src.data_contracts.load_csv`.
- The ML artifacts are the only interface between training and serving.
- The API and the agent must work with either the committed full dataset/artifacts or a
  quick fixture; they must never trigger training implicitly.
- If a file is missing at runtime, fail with a clear, actionable error (e.g. "run
  `make data` / `make train` first").
