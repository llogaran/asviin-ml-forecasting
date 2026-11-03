# Architecture Decision Records

This document records the architectural decisions behind `asviin-ml-forecasting`. Each ADR
states the context, the decision, its consequences and the alternatives that were
considered. The decisions are binding for the current implementation unless a later ADR
supersedes them.

Conventions:

- ADRs are numbered sequentially and are immutable once accepted. A changed decision
  requires a new ADR that supersedes the old one; the old record stays for history.
- Status is one of `Proposed`, `Accepted`, `Superseded` or `Deprecated`.
- Cross-references use repository-relative paths and specification sections, never
  external links.

| ADR | Title | Status |
|---|---|---|
| ADR-001 | Synthetic dataset as the primary data source | Accepted |
| ADR-002 | Strict isolation of the AI layer via a data contract and an API | Accepted |
| ADR-003 | Global XGBoost quantile models with seasonal and Croston/SBA baselines | Accepted |
| ADR-004 | Canonical CSV contracts validated at the ETL boundary | Accepted |
| ADR-005 | PSI and KS drift detection with explicit thresholds | Accepted |
| ADR-006 | FastAPI serving with Pydantic schemas, optional API key and 503 without a model | Accepted |
| ADR-007 | LLM agent offline-first with deterministic fallback and tool calling | Accepted |
| ADR-008 | Public repository with zero real client data | Accepted |

## ADR-001: Synthetic dataset as the primary data source

**Status:** Accepted

**Context.** The project needs several years of daily demand per branch and product, with
inventory movements, cancellations, promotions, pending orders and intermittent demand.
The only real source is Asviin's production ERP, whose data belongs to clients and cannot
be copied into a public portfolio repository. Real client volume is also uneven: a small
install may not contain enough history or enough intermittent series to train and evaluate
a global model, and its data changes over time, which would make results irreproducible.
The specification makes the generator itself a deliverable, not a test fixture.

**Decision.** The primary data source is a deterministic synthetic generator
(`src/pipelines/synthetic_generator.py`). It produces the raw CSV contract plus
`data/raw/seed.json` with parameters only. Defaults are 1096 days, 20 products, 3 branches,
seed 42 and start date `2023-01-01`. The committed dataset and every artifact derived from
it are synthetic. The generator encodes the business assumptions explicitly (weekly and
seasonal patterns, promotions, cancellations, pending-sale share, intermittency and stock
caps) so they can be reviewed and changed. Real data may enter later only through the same
CSV contract, never through a separate code path or credential.

**Consequences.** The repository is publishable, reproducible and free of client-data
exposure; rare events and failure cases can be injected deliberately; tests can run
offline. The cost is that accuracy figures describe synthetic behavior, not production
performance. The generator's assumptions become the project's assumptions and need
maintenance; a real deployment will face distribution shift and must revalidate the models.

**Alternatives considered.** Anonymized real exports were rejected because anonymization
does not remove re-identification and consent risk, and because the data would still not
be redistributable. Aggregated real statistics were rejected because they cannot support
the daily per-product grain. Public forecasting benchmarks were rejected because they do
not share Asviin's schema, units or business drivers and would not exercise the data
contract.

## ADR-002: Strict isolation of the AI layer via a data contract and an API

**Status:** Accepted

**Context.** Asviin runs on Vue 3 and Firebase Cloud Functions with Firestore, in
JavaScript. This repository is Python and must remain reviewable and deployable on its own.
Importing ERP handlers or reading Firestore directly would couple the ML layer to
credentials, security rules, internal collection names and a release cycle it does not
control. It would also make the public repository depend on private infrastructure.

**Decision.** The ML layer integrates through two boundaries only: the canonical CSV data
contract for batch data, and a FastAPI service for online predictions. The repository never
imports Asviin code, never connects to Firestore and never writes to the ERP. The mapping
from Asviin documents to CSV is documented in `docs/DATA_CONTRACT.md`; the synthetic
generator emulates that projection. Any future exporter is a separate adapter whose only
output is the contract.

**Consequences.** The ML code can be tested, versioned and deployed independently, and the
public repository has no production credentials. The trade-off is batch latency and a
duplicated projection that must be kept in sync with the ERP; schema changes are handled by
an explicit contract change instead of an implicit import. There is no real-time demand
signal and no write-back of recommendations into the ERP.

**Alternatives considered.** A direct Firestore client was rejected for credential and
privacy risk. A shared internal package would still require private distribution and a
common release cadence. An ERP plugin or cloud function written in JavaScript was rejected
because it would move the ML stack into a runtime without the required libraries and would
recreate the coupling. A managed data warehouse export was rejected because it adds
infrastructure that this portfolio build does not need.

## ADR-003: Global XGBoost quantile models with seasonal and Croston/SBA baselines

**Status:** Accepted

**Context.** The dataset has many intermittent or lumpy series (product x branch), strong
weekly and yearly seasonality, promotion effects, stockout censoring, a small catalog and
only about three years of history. Inventory recommendations need an uncertainty range, not
only a point forecast. The environment already provides XGBoost, scikit-learn and pandas,
has no GPU and must run deterministically on a laptop and in CI.

**Decision.** Train one global XGBoost model over all series with three quantile
regressors: p10, p50 and p90 (`objective="reg:quantileerror"`, `quantile_alpha` 0.1/0.5/0.9,
`tree_method="hist"`, `random_state=42`). The target is `log1p(units_sold)`; features come
from `src/pipelines/features.py` and are lagged by at least one day. Evaluation uses a
28-day holdout (14 days in quick mode) and reports MAE, WAPE, MASE against the seasonal
naive forecast, and 90% interval coverage. Two deterministic baselines are mandatory:
naive seasonal (same weekday of the previous week) and Croston/SBA. Serving uses recursive
one-step forecasting through `features_for_next_day`.

**Consequences.** Pooling shares statistical strength across sparse series and keeps one
artifact instead of dozens. Quantiles provide the interval needed by the recommendation
formula. The global model can underfit series-specific behavior, recursive forecasting
accumulates error, and independently trained quantiles can cross; coverage is therefore
monitored on the backtest. Baselines make any claimed benefit falsifiable, and the model
must be retrained when the catalog or the demand regime changes.

**Alternatives considered.** One ARIMA/ETS model per series was rejected because many
series are too short and intermittent for stable estimation, and one artifact per series
(about 60 in a full run) complicates operations. Prophet adds a dependency and is a poor
fit for item-level intermittency. A point-only XGBoost model with residual-based intervals
was rejected because it makes the uncertainty an assumption rather than a learned
quantity. Deep learning was rejected for data volume, determinism and dependency cost.
Using Croston/SBA as the champion was rejected because it cannot use price, promotion,
stockout or calendar covariates.

## ADR-004: Canonical CSV contracts validated at the ETL boundary

**Status:** Accepted

**Context.** Several components produce and consume data: the generator, the ETL, feature
building, training, monitoring, the API and the agent. Without a fixed shape, a renamed
column or an unexpected unit would propagate silently into features and metrics. The
project also needs the dataset to be inspectable and diffable in reviews, and it must not
add heavy data-quality dependencies.

**Decision.** Define one canonical CSV contract in `src/data_contracts/schemas.py`: ordered
columns, Pydantic models, enums, `SCHEMA_VERSION` and `CSV_COLUMNS`. Validate at the ETL
boundary with `src/data_contracts/validation.py`: exact column set, required values,
ISO-8601 dates, enum domains, non-negative quantities, boolean flags, uniqueness of
product and branch keys, and foreign keys between raw files. Invalid input raises
`SchemaValidationError` with a readable list of violations. Downstream readers use
`load_csv`, never a bare `pandas.read_csv`. The ETL is the only writer of
`data/processed/`. Any contract change updates code, validators, this documentation and
fixtures together, and follows the versioning policy in `docs/DATA_CONTRACT.md`.

**Consequences.** Failures are loud and early, tests are stable, and the dataset remains
reviewable. The strict header check means adding a column is a MAJOR change; this is
intentional. The team pays a maintenance cost whenever the contract evolves, and
exploratory analysis must go through an explicit mapping rather than editing a file in
place.

**Alternatives considered.** `pandera` or Great Expectations would add dependencies and
configuration without removing the need to define the same rules. A database schema was
rejected because it adds operational infrastructure and hides the data from review. Ad hoc
assertions inside each pipeline were rejected because they drift. Parquet or Arrow was
rejected because the specification requires reviewable CSV and the dataset is small.

## ADR-005: PSI and KS drift detection with explicit thresholds

**Status:** Accepted

**Context.** The model is trained on a historical window and served later. Input
distributions can move because of seasonality, promotions, catalog changes or a genuine
regime change. Label-based performance monitoring is not practical in this build: the
outcome is available only after the sale is realized, and the portfolio deployment has no
ground-truth feed. Monitoring still needs a deterministic, explainable early warning.

**Decision.** Compare a reference window with a current window per numeric feature using
two univariate statistics: Population Stability Index (PSI) with ten quantile bins derived
from the reference, and a two-sample Kolmogorov-Smirnov test. A feature is `drift` when
PSI >= 0.2 or KS p-value < 0.01; it is `warning` when PSI >= 0.1 or p-value < 0.05;
otherwise it is `ok`. The overall status is `drift` when at least 30% of features drift,
`warning` when any feature warns or drifts, and `ok` otherwise. The CLI splits the data
chronologically, first 70% as reference and last 30% as current, and `--inject-drift`
multiplies current `units_sold` by 1.6 to demonstrate that detection works. Reports are
JSON with row counts, thresholds, per-feature values and a timestamp.

**Consequences.** The check is simple, dependency-light and explainable to operators.
It is univariate, so it misses joint shifts and causal relationships. The KS test becomes
very sensitive with large samples, while PSI depends on the binning; both will produce
false positives at some rate. The thresholds are operational heuristics and must be
recalibrated on real data. Drift is a warning to investigate, not an automatic retraining
trigger.

**Alternatives considered.** Evidently or a full monitoring platform was rejected as an
extra dependency and service. PSI-only was rejected because it does not detect shape
changes that preserve the binned proportions. KS-only was rejected because its decision
depends heavily on sample size and it provides no bin-level diagnostic. Performance
monitoring was rejected because labels are delayed and unavailable in this build. A domain
classifier was rejected because it needs labels, adds complexity and is harder to explain
per feature.

## ADR-006: FastAPI serving with Pydantic schemas, optional API key and 503 without a model

**Status:** Accepted

**Context.** Forecasts and replenishment recommendations must be consumable by a UI or
another service. The team is Python-based and wants a small, documented HTTP surface that
runs locally and in a container, without a model-serving platform. The service can start
before training has produced artifacts, and some deployments need a minimal access gate
without building an identity provider.

**Decision.** Serve with FastAPI and Pydantic v2. The app exposes `GET /health`,
`GET /ready`, `GET /model`, `GET /products`, `POST /forecast`, `POST /recommendation` and
`GET /metrics`. Request and response bodies are Pydantic models. If `MLFC_API_KEY` is set,
every endpoint except `/health` requires the `X-API-Key` header and returns 401 otherwise.
Unknown products or branches return 404 with a clear message, validation errors return 422,
and a missing model returns 503 with an actionable message. CORS origins come from
`MLFC_CORS_ORIGINS`. Model loading is lazy and injectable for tests; the API never trains
implicitly. `/metrics` exposes per-process counters. Errors are written in English with a
message suitable for a Spanish-speaking operator to relay.

**Consequences.** The service is easy to run, document through OpenAPI and test with an
injected service. The static key provides coarse access control only: no identity, roles,
rotation or rate limiting. Counters are per process and reset on restart, and a new model
requires a restart or a new instance. The 503 behavior makes the missing-artifact state
explicit instead of serving stale or fabricated predictions.

**Alternatives considered.** Flask was rejected because validation, serialization and
OpenAPI would be rebuilt by hand. BentoML or MLflow serving was rejected as
infrastructure-heavy for one global model. A managed cloud endpoint was rejected because it
adds cost, credentials and vendor coupling to a portfolio build. gRPC was rejected because
the expected consumers are HTTP clients and browsers. Full OAuth/RBAC was rejected as
disproportionate; a real deployment should place the service behind an authenticating
gateway.

## ADR-007: LLM agent offline-first with deterministic fallback and tool calling

**Status:** Accepted

**Context.** Non-technical users should be able to ask for a forecast or a replenishment
recommendation in natural language. CI and local development must run without network
access, the test suite must be deterministic, and no new dependency should be added.
An LLM can improve language coverage, but it can also invent numbers or fail when the
provider is unavailable.

**Decision.** The default agent mode is fully offline and deterministic. It normalizes
accents, detects the intent (recommendation or forecast), resolves the product by name
with `difflib` at a 0.55 similarity floor, resolves the branch by ID or name (defaulting to
the highest-volume branch when the request is unambiguous), extracts the horizon from
`días`/`semanas` expressions, calls the existing `Forecaster`, and composes a concise
Spanish answer that quotes the exact numbers and the model version. If a product is not
found, it asks for clarification and lists candidates. When `OPENAI_API_KEY` is set, an
OpenAI-compatible chat completion may be used with tool definitions
(`list_products`, `forecast`, `recommendation`) executed locally against the forecaster,
for at most three tool rounds. Any failure falls back to the offline path and reports
`mode="offline_fallback"`. Tests use offline mode only.

**Consequences.** The agent works without a network, produces reproducible answers and can
be tested exactly. Its language coverage is narrower than an LLM's, fuzzy matching can
pick the wrong product near the threshold, and the branch default is a convenience rather
than a clarification. The optional LLM adds latency, cost and non-determinism, but it never
computes numbers itself: tool outputs remain the source of truth and are quoted back.
There is no conversation memory across invocations.

**Alternatives considered.** An LLM-only agent was rejected because it requires network
access, is non-deterministic, can hallucinate inventory numbers and would make offline CI
impossible. An agent framework such as LangChain was rejected as a new dependency for a
small tool surface. A local model was rejected because of hardware and download
requirements. Omitting the agent was rejected because a natural-language interface is part
of the product value.

## ADR-008: Public repository with zero real client data

**Status:** Accepted

**Context.** This repository is a public portfolio piece. Asviin V1 operated with real
clients and production records; Asviin V2 is used with test data. Client records can
contain names, addresses, credit balances, supplier data and commercially sensitive
volumes. Publishing any of it would create privacy, contractual and legal exposure, even
after superficial anonymization. The repository must still demonstrate realistic ERP
semantics and ML engineering.

**Decision.** The public repository contains only synthetic data generated from documented
seeds. It must never contain real client data, production backups, `.env` files,
credentials, API keys, customer or supplier identifiers, or screenshots of real systems.
The Asviin source tree is read-only reference material for schema fidelity; it is not
vendored and no private code is copied. Documentation states that all data is synthetic and
that reported metrics describe synthetic behavior. Contributions that add client data are
rejected.

**Consequences.** The project is safe to share and review, and the privacy guarantee is
structural rather than policy-only. The cost is that the repository cannot demonstrate
accuracy on real demand, and every demonstration must be regenerated from the seed. A real
deployment needs a private data path, access controls and a separate evaluation with the
client's consent. It also means the public numbers must never be presented as production
results.

**Alternatives considered.** Keeping the repository private was rejected because it defeats
the portfolio and review purpose. Publishing anonymized real data was rejected because
re-identification and consent risk remain, and deletion is not verifiable. Publishing only
aggregated real statistics was rejected because aggregates still belong to clients and are
insufficient for item-level modeling. Using a public benchmark dataset was rejected because
it cannot exercise Asviin's schema, Mexican MXN context or the ERP projection documented in
`docs/DATA_CONTRACT.md`.
