# Data Contract

Canonical dataset contract for `asviin-ml-forecasting`. This document describes the CSV
files under `data/`, the derived quality report, and the projection from Asviin v2
(economic events, products, inventory movements and pending sales) into those files.

The executable source of truth is:

- `src/data_contracts/schemas.py` — Pydantic models, enums, `SCHEMA_VERSION`, `CSV_COLUMNS`.
- `src/data_contracts/validation.py` — header, null, date, enum, sign and foreign-key checks.

If this document and those modules disagree, the modules win and this document must be
corrected. Section 4 defines how such corrections are versioned.

## 1. Purpose and scope

The contract has three jobs:

1. Fix the shape of the synthetic raw dataset and of the daily modeling table so that the
   generator, the ETL, training, monitoring, the API and the agent can be developed
   independently.
2. State the business meaning of every column (units, currency, time zone, cancellation
   and inventory semantics) so that a value is not reinterpreted downstream.
3. Record how the canonical fields correspond to Asviin v2 documents, so a future export
   adapter can be written without re-deriving the mapping.

In scope: `data/raw/*.csv`, `data/processed/demand_daily.csv`,
`data/processed/data_quality.json`, and the Asviin v2 projection. Out of scope: the HTTP
API contract (FastAPI/OpenAPI), the model artifact layout (`artifacts/models/`), and the
drift report schema; those are documented by their own modules and the specification.

## 2. Provenance

- The dataset is **100% synthetic**. It contains no real client data, no production
  backups, no credentials and no personal information. See ADR-001 and ADR-008 in
  `docs/DECISIONS.md`.
- It is produced by `src/pipelines/synthetic_generator.py`. Default parameters: 1096
  calendar days, 20 products, 3 branches, seed 42, start date `2023-01-01`. The full run
  uses exactly three branches (`B-01` Matriz Centro, `B-02` Sucursal Norte, `B-03` Bodega
  Sur) with approximate demand multipliers 1.0 / 0.62 / 0.38.
- Determinism: with the same seed and parameters the generator emits byte-identical CSVs.
  `data/raw/seed.json` stores the deterministic generator parameters only; it must not
  contain timestamps or run-specific values.
- Realism rules baked into the generator: weekly Thursday-Saturday uplift, December peak
  around 1.4x, May dip around 0.85x, a March-April recycling-season peak, a trend of
  roughly +3% per quarter, about 8% promo days with 10-25% discounts, about 2% cancelled
  sale lines, pending sales around 10% of demand, intermittent/lumpy demand by product
  pattern, and sales capped at available stock.
- Business context: multi-branch ERP for small and medium plastic recyclers and
  transformers in Mexico. Currency is **MXN**; the business time zone is
  **America/Mexico_City**.
- The Asviin v2 source tree is read as a schema reference only. This repository never
  imports Asviin code and never connects to Firestore. The mapping in Section 7 defines a
  projection; it is not an integration.

## 3. Format conventions

All CSV files share these rules:

| Rule | Convention |
|---|---|
| Encoding | UTF-8, no BOM |
| Delimiter | comma |
| Header | one header row, canonical column names and canonical order |
| Index | no index column |
| Dates | `YYYY-MM-DD` (the `date` column of `demand_daily`) |
| Timestamps | ISO-8601 with UTC offset, for example `2024-03-15T14:32:00-06:00` |
| Floats | decimal point, up to 2 decimals in the committed dataset |
| Booleans | integer `0` or `1` |
| Empty values | an empty field means "not applicable" or "unknown"; it is allowed only in columns marked optional |
| Currency | nominal MXN in every `*_mxn` column; no FX conversion, no inflation adjustment, no tax decomposition |
| Time zone | timestamps carry an offset; daily aggregation uses the America/Mexico_City calendar date |
| Quantities | expressed in each product's base `unit`; fractional values are valid for `kg`, `l` and `ml` |
| Ordering | row order is not semantically significant except for the documented stock tie-break |

Column order is canonical for producers and for byte-level review. Readers address
columns by name. The validator rejects missing and unexpected columns, but it does not
verify column order.

## 4. Schema versioning policy

`SCHEMA_VERSION` in `src/data_contracts/schemas.py` versions the whole canonical contract
(all CSV files plus the `data_quality.json` key set). The current value is **`1.0.0`**.

The version follows semantic versioning adapted to a data interface:

| Change | Bump |
|---|---|
| Add, remove, rename or reorder a column; add or remove a file | MAJOR |
| Change a column type, unit, currency, time-zone convention or grain | MAJOR |
| Change the meaning of an existing column (for example promotion or stockout semantics) | MAJOR |
| Add or remove an enum value; make an optional column required or a required column optional | MAJOR |
| Change the requiredness of a `data_quality.json` key, remove a key, or change a key's meaning | MAJOR |
| Add a new key to `data_quality.json` (readers must ignore unknown keys) | MINOR |
| Relax a validation without changing the file shape, add a new derived file that existing readers do not load | MINOR |
| Documentation, error-message or validator fixes that restore the documented behavior | PATCH |

Adding a CSV column is a MAJOR change because `validate_csv` rejects unexpected columns.
That strictness is intentional: a schema mismatch must fail loudly instead of being
silently dropped.

Rules for changing the contract:

1. Update `schemas.py` and `validation.py` together; never change only one.
2. Update this document and the affected fixtures/tests in the same change.
3. Bump `SCHEMA_VERSION` exactly once per contract change.
4. Regenerate the committed dataset when the change affects the generated files.
5. Announce the change in the pull request description; downstream consumers must pin a
   repository revision, because CSV headers carry no version field.

The version is declared by code, not embedded in the CSV files. The `schema_version`
field of `artifacts/models/registry.json` versions the model artifact manifest and is
independent of `SCHEMA_VERSION`; both currently start at `1.0.0` but they evolve
separately.

Generator parameter changes (days, product count, seed) are not contract changes and do
not bump `SCHEMA_VERSION`.

## 5. File inventory and lineage

| File | Grain | Producer | Primary consumers |
|---|---|---|---|
| `data/raw/products.csv` | one row per product | synthetic generator | ETL, features, API metadata |
| `data/raw/branches.csv` | one row per branch | synthetic generator | ETL, API metadata |
| `data/raw/recipes.csv` | one row per BOM line | synthetic generator | ETL validation, product metadata |
| `data/raw/economic_events.csv` | one row per event line (sales are item-level) | synthetic generator | ETL |
| `data/raw/stock_movements.csv` | one row per inventory movement | synthetic generator | ETL |
| `data/raw/pending_sales.csv` | one row per pending-sale product line | synthetic generator | ETL |
| `data/processed/demand_daily.csv` | one row per date, business, branch and product | ETL only | features, training, drift, API |
| `data/processed/data_quality.json` | one object per ETL run | ETL only | operators, CI, reports |

`data/raw/seed.json` is a deterministic parameter manifest, not a data table. Its keys are
defined by the generator and are not part of this CSV contract.

Lineage:

```
synthetic_generator -> data/raw/*.csv + seed.json
                    -> etl.build_daily_demand (validate + aggregate + zero-fill)
                    -> data/processed/demand_daily.csv + data_quality.json
                    -> features/train -> artifacts/models + artifacts/reports
                    -> drift / API / agent (read processed data and artifacts)
```

`data/processed/` is written only by the ETL. Downstream code reads
`demand_daily.csv` through `src.data_contracts.load_csv`, so the validation boundary is
not bypassed.

## 6. Data dictionary

Type notation: `string`, `integer`, `float`, `date`, `datetime`, `enum`, `0/1` (integer
boolean flag), `object`. "Required" refers to the column: `yes` means every row must have
a non-empty value; `no` means empty values are allowed; `conditional` means the rule in
the semantics column applies.

### 6.1 `data/raw/products.csv`

One row per product in the catalog. Product identifiers are stable across every file.

| Column | Type | Required | Rules | Semantics |
|---|---|---|---|---|
| `product_id` | string | yes | unique; non-empty; `P-###` style in the committed dataset | Product primary key. |
| `name` | string | yes | non-empty; realistic Spanish name | Display name; not a key and may change over time. |
| `category` | string | yes | non-empty; committed values include `Materia Prima`, `Producto Terminado`, `Insumos`, `Consumibles` | Reporting group. |
| `unit` | enum | yes | `pza`, `kg`, `l`, `ml` | Base unit for every quantity of this product. |
| `price_mxn` | float | yes | `> 0` | List selling price per base unit, MXN. Also the reference for promo detection in the ETL. |
| `cost_mxn` | float | yes | `>= 0`; committed data keeps it at 55-80% of `price_mxn` | Standard/average cost per base unit, MXN. |
| `is_ingredient` | 0/1 | yes | `0` or `1` | `1` means the product is an input consumed by recipes and is normally not sold directly. |
| `has_recipe` | 0/1 | yes | `0` or `1`; at least five sellable products are `1` in a full run | `1` means at least one BOM line exists in `recipes.csv`. |
| `safety_stock` | float | yes | `>= 0` | Buffer quantity in base units. It is not a reorder point. |
| `lead_time_days` | integer | yes | `>= 0` by schema; 3-21 in the committed dataset | Calendar days between replenishment order and availability. |
| `demand_pattern` | enum | yes | `smooth`, `intermittent`, `lumpy` | Synthetic demand class used for stratification and model analysis. |
| `abc_class` | enum | yes | `A`, `B`, `C`; roughly 20/30/50 of sellable volume | Synthetic ABC class, assigned by the generator, not computed by the ETL. |

Notes:

- Asviin scopes products to a business; `products.csv` has no `business_id`. The committed
  dataset is a single-business projection. Multi-tenant exports require one dataset per
  business or a contract change (MAJOR).
- `has_recipe` and `is_ingredient` are independent flags. A product can be an ingredient
  without being the output of a recipe.

### 6.2 `data/raw/branches.csv`

One row per branch (sucursal).

| Column | Type | Required | Rules | Semantics |
|---|---|---|---|---|
| `branch_id` | string | yes | unique; `B-01`, `B-02`, `B-03` in a full run | Branch primary key and foreign key in every fact table. |
| `business_id` | string | yes | non-empty; same tenant in all rows of a dataset | Owning business (tenant isolation key). |
| `name` | string | yes | non-empty | Branch display name (`Matriz Centro`, `Sucursal Norte`, `Bodega Sur` in a full run). |
| `region` | string | yes | non-empty | Analytical region label. Not a guaranteed native Asviin field; see Section 7. |

The full run has exactly three branches with demand multipliers approximately 1.0 / 0.62 /
0.38. `region` is descriptive only and is not used as a model key.

### 6.3 `data/raw/recipes.csv`

One row per bill-of-materials line: how much of an input is consumed per unit of output.

| Column | Type | Required | Rules | Semantics |
|---|---|---|---|---|
| `recipe_id` | string | yes | unique per recipe body in the committed dataset | Recipe identifier; several rows may share it if the recipe has several inputs. |
| `business_id` | string | yes | must match the dataset tenant | Owning business. |
| `output_product_id` | string | yes | foreign key to `products.product_id` | Product produced or assembled. |
| `input_product_id` | string | yes | foreign key to `products.product_id`; that product should have `is_ingredient = 1` | Product consumed. |
| `quantity_per_unit` | float | yes | `> 0` | Quantity of the input, in the input's base unit, consumed per one unit of output. |

Producer invariants: every product with `has_recipe = 1` has at least one row; at least
five sellable products have a recipe in a full run; recipe inputs are ingredient products.
The validator checks the foreign keys, not these business invariants.

### 6.4 `data/raw/economic_events.csv`

One row per event line. A `venta` ticket with three items produces three rows that share
the same `event_id`; the ticket total is not repeated. Non-sale events may have a single
row and may leave the product columns empty.

| Column | Type | Required | Rules | Semantics |
|---|---|---|---|---|
| `event_id` | string | yes | non-empty; repeated across the item lines of one sale | Source event identifier. Not unique in this file. |
| `business_id` | string | yes | must match the dataset tenant | Owning business. |
| `branch_id` | string | yes | foreign key to `branches.branch_id` | Branch where the event occurred. |
| `event_type` | enum | yes | `venta`, `compra`, `gasto`, `merma`, `produccion`, `transferencia` | Economic event type. |
| `event_ts` | datetime | yes | ISO-8601 with offset; business-local attribution | Event timestamp. It determines the local calendar day used by the ETL. |
| `amount_mxn` | float | yes | `>= 0` | For `venta`, the line amount (quantity times unit price, after line discounts as recorded). For every other type, the event amount. |
| `payment_method` | enum | no | `efectivo`, `transferencia`, `tarjeta`, `credito`; empty for non-sales | Payment method for sales. `credito` is what licenses a non-empty `customer_id`. |
| `customer_id` | string | no | empty unless the sale was on credit | Opaque customer reference. Never contains personal data in the synthetic dataset. |
| `product_id` | string | conditional | required for sale lines; foreign key when present; may be empty for `gasto` | Product moved or sold. |
| `product_name` | string | conditional | expected when `product_id` is present | Product name at the time of the event (denormalized snapshot). |
| `quantity` | float | conditional | `>= 0`; required for sale lines | Units sold or moved, in the product's base unit. Not a package count. |
| `unit_price_mxn` | float | conditional | `>= 0`; required for sale lines | Price per base unit at the time of the line. |
| `unit_cost_mxn` | float | conditional | `>= 0`; required for sale lines | Cost per base unit at the time of the line. |
| `is_cancelled` | 0/1 | yes | `0` or `1`; only meaningful for `venta`; non-sale rows use `0` | `1` marks a cancelled sale line, which is excluded from sales and counted in `cancelled_units`. |
| `note` | string | no | free text; no personal data | Operational annotation. |

Notes:

- Cancellations are represented by `is_cancelled = 1`, not by negative amounts. Amounts
  and quantities are non-negative.
- A ticket-level discount may exist in the source without being allocated to line
  amounts. `amount_mxn` is the line amount as recorded; it is not recomputed from
  `quantity * unit_price_mxn`.
- `event_id` plus `product_id` is not a guaranteed unique key: one ticket may contain the
  same product on two lines. The ETL aggregates additively and does not rely on line
  identity.
- For non-sale events, product columns may be empty; the ETL does not use them to build
  daily demand.

### 6.5 `data/raw/stock_movements.csv`

One row per inventory movement. This is the authoritative ledger for daily on-hand stock.

| Column | Type | Required | Rules | Semantics |
|---|---|---|---|---|
| `movement_id` | string | yes | unique in the committed dataset | Movement primary key. |
| `business_id` | string | yes | must match the dataset tenant | Owning business. |
| `branch_id` | string | yes | foreign key to `branches.branch_id` | Branch whose stock changed. |
| `product_id` | string | yes | foreign key to `products.product_id` | Product whose stock changed. |
| `movement_ts` | datetime | yes | ISO-8601 with offset | Effective timestamp; defines the local day for the stock series. |
| `direction` | enum | yes | `entrada`, `salida` | Sign of the movement. |
| `reason` | enum | yes | `venta_manual`, `compra`, `ajuste_manual`, `merma`, `produccion`, `transferencia_entrante`, `transferencia_saliente` | Business cause of the movement. |
| `quantity` | float | yes | `> 0`; magnitude only, direction carries the sign | Units moved, in base units. |
| `unit_cost_mxn` | float | yes | `>= 0` | Unit cost applied to the movement, MXN. |
| `stock_after` | float | yes | `>= 0` | Closing stock for this product and branch after the movement. |
| `note` | string | no | free text; no personal data | Operational annotation. |

Producer invariants: movements are ordered by `movement_ts`; within the same timestamp,
file order is the tie-break. For one product and branch, each `stock_after` equals the
previous `stock_after` plus `quantity` for `entrada` or minus `quantity` for `salida`.
`stock_after` never goes below zero. Both continuity and non-negativity are mandatory
producer invariants, not optional conventions.

### 6.6 `data/raw/pending_sales.csv`

One row per pending-sale product line. A pending order with several products produces
several rows that share `pending_sale_id`.

| Column | Type | Required | Rules | Semantics |
|---|---|---|---|---|
| `pending_sale_id` | string | yes | non-empty; may repeat across the lines of one order | Pending order identifier. |
| `business_id` | string | yes | must match the dataset tenant | Owning business. |
| `branch_id` | string | yes | foreign key to `branches.branch_id` | Branch that committed the order. |
| `product_id` | string | yes | foreign key to `products.product_id` | Requested product. |
| `created_ts` | datetime | yes | ISO-8601 with offset | When the commitment was created; defines the local day used by `pending_units`. |
| `promised_ts` | datetime | no | ISO-8601 with offset when present; must not precede `created_ts` | Promised delivery timestamp. Optional and not guaranteed by the source. |
| `quantity` | float | yes | `> 0` in the committed dataset | Committed quantity in base units. |
| `unit_price_mxn` | float | yes | `>= 0` | Agreed price per base unit, MXN. |
| `status` | enum | yes | `pendiente`, `convertida`, `cancelada` | Lifecycle state. `pendiente` is open; `convertida` became a sale; `cancelada` is void. |
| `converted_event_id` | string | no | must be set when `status = convertida` and point to the resulting sale `event_id`; empty otherwise | Link from committed demand to the invoice event. |

Lifecycle rules: a pending sale starts as `pendiente` and ends as `convertida` or
`cancelada`; terminal states are not reopened. Converted and cancelled rows are kept as
historical demand commitments. Only rows whose `status != cancelada` contribute to
`pending_units`, and only on their `created_ts` day. The conditional rules for
`promised_ts` and `converted_event_id` are producer invariants; the current validator does
not enforce them.

### 6.7 `data/processed/demand_daily.csv`

The modeling table. Grain: one row per `(date, business_id, branch_id, product_id)` for
every calendar day in the raw event range and every branch/product combination. There are
no gaps and no duplicate keys.

| Column | Type | Required | Rules | Semantics |
|---|---|---|---|---|
| `date` | date | yes | `YYYY-MM-DD`; America/Mexico_City local day | Calendar day of the row. |
| `business_id` | string | yes | matches the masters | Tenant key. |
| `branch_id` | string | yes | matches the masters | Branch key. |
| `product_id` | string | yes | matches the masters | Product key. |
| `category` | string | yes | copied from `products.category` | Product category snapshot at ETL time. |
| `unit` | enum | yes | copied from `products.unit` | Base unit of every quantity in the row. |
| `units_sold` | float | yes | `>= 0` | Sum of non-cancelled sale-line quantities for the day. |
| `revenue_mxn` | float | yes | `>= 0` | Sum of non-cancelled sale-line amounts, MXN. |
| `cost_mxn` | float | yes | `>= 0` | Sum of non-cancelled sale-line costs, MXN. |
| `cancelled_units` | float | yes | `>= 0` | Sum of quantities on cancelled sale lines attributed to the sale date. Excluded from `units_sold`. |
| `pending_units` | float | yes | `>= 0` | Quantity of pending sales created that day with `status != cancelada`. Not invoiced demand. |
| `stock_on_hand` | float | yes | `>= 0` | Closing stock for the product/branch on that day: last `stock_after` of the day, forward-filled; `0` before the first movement. |
| `stockout_flag` | 0/1 | yes | `0` or `1` | `1` when closing stock is `<= 0`; signal that sales may be censored. |
| `promo` | 0/1 | yes | `0` or `1` | `1` when any sale line that day had `unit_price_mxn < 0.85 * products.price_mxn`. |
| `unit_price_mxn` | float | yes | `>= 0` | Quantity-weighted average sale price for the day; falls back to `products.price_mxn` when there are no sales. |
| `unit_cost_mxn` | float | yes | `>= 0` | Quantity-weighted average sale cost for the day; falls back to `products.cost_mxn` when there are no sales. |
| `lead_time_days` | integer | yes | `>= 0` | Lead time copied from the product master. |
| `demand_pattern` | enum | yes | `smooth`, `intermittent`, `lumpy` | Product demand class copied from the master. |
| `abc_class` | enum | yes | `A`, `B`, `C` | Product ABC class copied from the master. |

Semantics that downstream code relies on:

- **Grain and completeness.** The ETL emits the full cartesian product of calendar dates,
  branches and products from the raw range. A missing sale is a zero row, not a missing
  row. `series` in the quality report is the number of distinct
  `(business_id, branch_id, product_id)` combinations.
- **Cancellations.** Cancelled lines contribute only to `cancelled_units`. They are
  attributed to the calendar day of the original sale line, not to the later cancellation
  timestamp. Their restocking movements do affect `stock_on_hand` on the day they occur.
- **Pending sales.** `pending_units` measures commitments created that day. A converted
  pending sale can therefore appear both as `pending_units` on its creation day and as
  `units_sold` on its conversion day. The columns are different demand concepts and must
  not be added together.
- **Stock and stockouts.** `stock_on_hand` is a physical-ledger quantity. Reservations for
  pending sales do not reduce it. `stockout_flag = 1` marks a closing stock of zero (the
  contract forbids negative stock); sales on such a day are censored because the
  generator caps sales at available stock. A zero `units_sold` on a stockout day is not
  evidence of zero demand.
- **Promotion.** The flag is derived from recorded prices, using the current product
  master list price as the reference, not from a generator flag. A strong negotiated
  discount can therefore be classified as a promotion.
- **Price and cost.** Daily averages are weighted by sale quantity. In the absence of
  sales, the master values are used. They are not inflation-adjusted and are not
  comparable across product units.
- **Slowly changing attributes.** `category`, `unit`, `lead_time_days`,
  `demand_pattern` and `abc_class` are snapshots of the master at ETL time. Historical
  master changes are not reconstructed.
- **Business calendar.** Every calendar day is present, including weekends and holidays.
  Weeks follow the Monday-Sunday convention; weekend features are derived from `date`.

### 6.8 `data/processed/data_quality.json`

A single UTF-8 JSON object written next to `demand_daily.csv` by the ETL. It describes the
table that was actually written. It is a report, not an input to training.

| Key | Type | Required | Rules | Semantics |
|---|---|---|---|---|
| `rows` | integer | yes | `>= 0` | Number of data rows in `demand_daily.csv`, excluding the header. |
| `series` | integer | yes | `>= 0` | Number of distinct `(business_id, branch_id, product_id)` series. |
| `date_min` | string | yes | `YYYY-MM-DD` | First `date` present. |
| `date_max` | string | yes | `YYYY-MM-DD` | Last `date` present. |
| `zero_sale_share` | float | yes | `0.0 <= x <= 1.0` | Share of rows with `units_sold == 0`. |
| `negative_stock_rows` | integer | yes | `>= 0`; expected `0` | Rows with `stock_on_hand < 0`. A non-zero value indicates a contract violation upstream, because negative stock is rejected at the boundary. |
| `missing_product_refs` | integer | yes | `>= 0`; expected `0` | Rows whose product or branch cannot be resolved against the masters. |
| `duplicate_keys` | integer | yes | `>= 0`; expected `0` | Duplicate `(date, business_id, branch_id, product_id)` groups. |
| `cancelled_units_total` | float | yes | `>= 0` | Sum of `cancelled_units` over all rows. |
| `pending_units_total` | float | yes | `>= 0` | Sum of `pending_units` over all rows. |

The key set is a minimum, not closed: extra keys may be added in a MINOR release, and
readers must ignore keys they do not know. A missing required key is a contract error. No
schema-version key is required; if a producer adds one, it is informational. The producing
repository revision identifies the contract.

## 7. Asviin v2 to CSV mapping

This section records the correspondence with Asviin v2 as implemented in
`functions/asviinV2` (Firestore collections, `handlers/`, `schemas/`). It is a design and
provenance reference for the synthetic generator and for a future read-only export
adapter. The repository does not execute this mapping and has no Firestore dependency.

### 7.1 Source collections

| Asviin v2 path | Document | Canonical target |
|---|---|---|
| `businesses/{businessId}/products/{productId}` | product master, with fields under `data.*` and per-branch stock in `stock` | `products.csv`, stock context |
| `businesses/{businessId}/branches/{branchId}` | branch master | `branches.csv` |
| `businesses/{businessId}/recipes/{recipeId}` | active or inactive recipe with an `insumos` array | `recipes.csv` |
| `businesses/{businessId}/economicEvents/{eventId}` | economic event, sale items nested in `metadata.items` | `economic_events.csv` |
| `businesses/{businessId}/products/{productId}/movimientos/{movementId}` | inventory movement document | `stock_movements.csv` |
| `businesses/{businessId}/pendingSales/{pendingSaleId}` | pending order with an `items` array | `pending_sales.csv` |
| `businesses/{businessId}/business_metrics/{businessId}_{period}` | monthly pre-computed metrics | not used (Section 7.7) |

### 7.2 Master data: products, branches and recipes

| Canonical column | Asviin v2 source | Projection rule |
|---|---|---|
| `products.product_id` | document ID | Copy as-is. |
| `products.name` | `data.nombre` | Trim; reject empty. |
| `products.category` | `data.categoria` | Trim; keep the free-text category. |
| `products.unit` | `data.unidadMedida` | Normalize to `pza`, `kg`, `l` or `ml`; reject unmappable units instead of guessing. |
| `products.price_mxn` | `data.precioVenta` | Numeric copy; must be `> 0`. |
| `products.cost_mxn` | `data.costoPromedio` | Numeric copy; must be `>= 0`. |
| `products.is_ingredient` | `data.isInsumo` | Coerce to `0/1`. |
| `products.has_recipe` | existence of an active recipe with `productId` | `1` if at least one active recipe references the product. |
| `products.safety_stock` | `data.stockMinimo` | Numeric copy; it is a minimum-stock setting, used as the safety-stock analogue. |
| `products.lead_time_days` | no native field | Analyst-supplied or synthetic attribute; not exported from the ERP. |
| `products.demand_pattern` | no native field | Synthetic/analytical classification. |
| `products.abc_class` | no native field | Synthetic/analytical classification. |
| `branches.branch_id` | branch document ID | Copy as-is. |
| `branches.business_id` | parent business path | Copy as-is. |
| `branches.name` | `nombre` | Trim. |
| `branches.region` | no reliable native field | Derive from `direccion`/`metadata`, or synthesize; label it as analytical. |

Recipe projection:

| Canonical column | Asviin v2 source (`recipes` document) | Projection rule |
|---|---|---|
| `recipes.recipe_id` | recipe document ID | Copy. |
| `recipes.business_id` | parent business path | Copy. |
| `recipes.output_product_id` | `productId` | Copy; this is the product the recipe produces. |
| `recipes.input_product_id` | `insumos[].productId` | Flatten each input into one row. |
| `recipes.quantity_per_unit` | `insumos[].cantidad` divided by `rendimiento` | `rendimiento` is the output quantity of one recipe batch (default 1); the canonical column is the input consumed per one output unit, in the input's base unit. |

Only active recipes (`activa: true`) are projected; an inactive recipe does not set
`products.has_recipe`. The source already converts `insumos[].cantidad` to the input
product's unit before storing it.

Other product `data.*` fields (`folio`, `descripcion`, `margenUtilidad`, `requiresStock`,
`costoProduccion`, `imageUrl`) are not part of the canonical CSV and are not projected.

The product master is tenant-scoped in Asviin, while `products.csv` has no `business_id`.
The export must therefore be single-tenant, or the contract must be extended (MAJOR
change). The `stock[branchId]` map on the product document is a current snapshot only; it
is not a stock history and cannot be used to build daily stock (Section 7.5).

### 7.3 Sale events: `metadata.items[]`

A sale event stores one nested item array. The live sales handler writes `metadata.items`
with these per-line fields. The schema declaration in `schemas/eventSchemas.js` lists a
`metadata.productos` array for `venta`, but the runtime handler is the authoritative shape
for the export.

| Canonical column | Asviin v2 source (`economicEvents.metadata.items[]`) | Projection rule |
|---|---|---|
| `economic_events.product_id` | `productId` | Copy. |
| `economic_events.product_name` | `nombre` | Copy. |
| `economic_events.quantity` | `cantidad` | Copy as a float in base units. |
| `economic_events.unit_price_mxn` | `precioUnitario` | Copy. |
| `economic_events.unit_cost_mxn` | `costoUnitario` | Copy. |
| `economic_events.amount_mxn` | `subtotal` (the line importe) | Copy the recorded line amount. Do not recompute it from quantity and price. |

Event-level fields repeated on each item row:

| Canonical column | Asviin v2 source | Projection rule |
|---|---|---|
| `event_id` | event document ID | Repeat for every item line of the ticket. |
| `business_id` | parent business path | Copy. |
| `branch_id` | `branchId` | Copy; a sale without a branch is invalid for the canonical contract. |
| `event_type` | `type` | `venta`. |
| `event_ts` | `timestamp` | Convert the Firestore timestamp to ISO-8601 with offset. |
| `payment_method` | `metadata.formaPago` | Normalize accents/case to `efectivo`, `transferencia`, `tarjeta` or `credito`; empty for non-sales. |
| `customer_id` | `metadata.customerId` (written only for credit sales) | Copy only for credit sales; the customer display name in `metadata.cliente` is not an ID and must not be used. |
| `is_cancelled` | `estado == "cancelado"` | `1` only for cancelled sale lines; `0` otherwise. |

The source also writes a mirror event of type `cancelacion` when a sale is cancelled. The
projection does not emit mirror events as separate rows; it marks the original sale lines
as cancelled. `intercambio_b2b` events are outside the demand scope and are not projected;
they are not part of the six canonical event types.

### 7.4 Non-sale events

For `compra`, `gasto`, `merma`, `produccion` and `transferencia`, the event document is
projected with `event_id` from the document ID, `amount_mxn = amount`, `event_type = type`,
and the event-level branch and timestamp. Product columns come from the nested
`metadata.items` or `metadata.productos` array when the handler recorded one; `gasto` may
legitimately have no product, quantity or unit price. `is_cancelled` is `0`. The canonical
contract needs the event amount for financial context, not item-level detail.

### 7.5 Inventory movements

| Canonical column | Asviin v2 source (`movimientos` document) | Projection rule |
|---|---|---|
| `movement_id` | document ID | Copy. |
| `business_id` | parent business path | Copy. |
| `branch_id` | `branchId` | Copy when present; legacy movements may lack it, in which case the branch must be recovered from the caller context or the row rejected. |
| `product_id` | parent product path or `productId` | Copy. |
| `movement_ts` | `fecha` (server timestamp) or `createdAt` | Convert to ISO-8601 with offset. |
| `direction` | `tipo` (`entrada`, `venta`, `salida`) | Map `entrada` to `entrada`; `venta` and `salida` to `salida`. |
| `reason` | `razon` for exits; `fuente`/caller context for entries | Normalize to the seven canonical reasons; reject when the cause cannot be established. |
| `quantity` | `cantidad` | Take the absolute value. Source exits are stored as negative quantities; direction carries the sign in the canonical file. |
| `unit_cost_mxn` | `costoUnitario` | Copy; `>= 0`. |
| `stock_after` | `stockNuevo` | Copy; this is the authoritative closing balance used by the ETL. |

The product-level `stock[branchId]` map is a current quantity, not a dated series.
Likewise, `stockReserved[branchId]` records pending-sale reservations and does not reduce
physical stock. Neither field can replace the movement ledger for daily
`stock_on_hand`. Legacy products may use `data.cantidadInicial` instead of the per-branch
map; the canonical stock series still starts at the first movement and is `0` before it.

### 7.6 Pending sales

| Canonical column | Asviin v2 source (`pendingSales` document) | Projection rule |
|---|---|---|
| `pending_sale_id` | document ID | Copy for each item line. |
| `business_id` | parent business path | Copy. |
| `branch_id` | `branchId` | Copy. |
| `product_id` | `items[].productId` | Flatten each item into one row. |
| `created_ts` | `createdAt` | Convert to ISO-8601 with offset. |
| `promised_ts` | no native field | Optional; may be empty or supplied by planning. |
| `quantity` | `items[].cantidad` | Copy. |
| `unit_price_mxn` | `items[].precioUnitario` | Copy. |
| `status` | `status` | Normalize `draft` to `pendiente`; `convertida` and `cancelada` are lifecycle projections, not raw stored statuses. |
| `converted_event_id` | sale ID returned when the order is closed | Set when the conversion is traceable; empty otherwise. |

Asviin currently deletes a pending order when it is closed or deleted, and stores only
`draft` while it is open. A live export therefore cannot reconstruct the full
`convertida`/`cancelada` history from the current documents; the synthetic generator
materializes it so the canonical lifecycle can be exercised. When an export adapter is
built, it must state which history source it uses and fail rather than invent a status.

### 7.7 Monthly business metrics limitation

`business_metrics` documents are written under
`businesses/{businessId}/business_metrics/` with an ID of the form
`{businessId}_{period}`, where `period` is `YYYY_MM`, and `periodType` is `monthly`. They
contain business-level aggregates: sales total and count, average ticket, expenses, profit,
cash flow and B2B counters. They do not contain:

- a branch dimension;
- a product dimension;
- daily buckets or per-product quantities;
- the line-level detail needed to reconstruct demand.

They are also derived snapshots that can be recalculated, not an immutable ledger. For
these reasons `business_metrics` cannot feed `demand_daily`, whose grain is
date x branch x product, and is not used as an ML feature or target in this repository.
At most it can serve as a coarse monthly business-level reconciliation of sale totals; any
such check must account for the same local-time calendar and cancellation rules. The
daily dataset is built from event lines, movements and pending sales instead.

### 7.8 Projection rules and known gaps

1. **Single tenant per dataset.** `products.csv` omits `business_id`; every other file
   carries it. Do not mix tenants in one dataset.
2. **Local calendar.** Convert source timestamps to America/Mexico_City before assigning
   a day. Storing UTC days would shift evening sales to the next day.
3. **Line flattening.** Sale and pending-order item arrays become one CSV row per item;
   event-level fields repeat. Event IDs are not unique in those files.
4. **Sign normalization.** Source exit movements use negative quantities; the canonical
   file uses positive magnitude plus `direction`.
5. **Status normalization.** `draft` becomes `pendiente`; closed and deleted pending
   orders become `convertida` or `cancelada` only when a lifecycle source exists.
6. **Synthetic-only attributes.** `region`, `lead_time_days`, `demand_pattern`,
   `abc_class`, `promised_ts` and `converted_event_id` have no guaranteed native source
   and must be labelled as analytical or synthetic when exported.
7. **No personal data.** Names, addresses, phone numbers, credit balances and customer
   documents are excluded. Only an opaque `customer_id` may appear, and only for credit
   sales.
8. **No financial reinterpretation.** Amounts are copied as recorded in MXN. The export
   does not recompute taxes, allocate ticket discounts, convert currencies or adjust for
   inflation.
9. **Strict failure.** If a required mapping is ambiguous (unknown unit, missing branch,
   unknown movement reason), the adapter must reject the row and report it. Silent
   guessing would corrupt the contract.

## 8. Validation at the ETL boundary

The ETL validates raw inputs before aggregation. `validate_csv(path, schema_name)` returns
a list of human-readable violations and `assert_valid_csv` raises
`SchemaValidationError` when the list is non-empty. `load_csv` validates and then loads,
returning `datetime`/`date` objects for the declared date columns.

Checks performed by `src/data_contracts/validation.py`:

1. The file exists and has exactly the canonical columns: no missing and no unexpected
   column.
2. Required columns contain no null values. Optional columns may be empty but must be
   present.
3. Date and timestamp columns parse as ISO-8601; `demand_daily.date` is a plain date.
4. Enum columns contain only their allowed values.
5. Numeric columns listed as non-negative contain no negative value.
6. Boolean flags contain only `0` or `1`.
7. `products.product_id` and `branches.branch_id` are unique; `products.price_mxn > 0`.
8. `validate_referential_integrity(raw_dir)` checks that product and branch references in
   events, movements and pending sales exist, and that recipe inputs and outputs exist.

The ETL must aggregate only after these checks pass. It must reject invalid input with
`SchemaValidationError` or an equally explicit error; silent coercion, dropping bad rows or
repairing values is not allowed. `data_quality.json` reports the post-ETL invariants
(`negative_stock_rows`, `missing_product_refs`, `duplicate_keys`), which are expected to be
zero.

## 9. Consumer checklist

Before consuming a dataset, a component should confirm:

1. The producer revision's `SCHEMA_VERSION` is the one it was built against.
2. `demand_daily.csv` loads through `src.data_contracts.load_csv`.
3. The grain is unique and complete for the requested date range.
4. Monetary values are treated as MXN and quantities in the product's base unit.
5. Cancelled units are excluded from sales and pending units are not added to sales.
6. `stockout_flag` is respected when interpreting zero or low sales.
7. Unknown keys in `data_quality.json` are ignored; missing required keys are an error.

Any change that violates these expectations requires a contract change under Section 4,
not an ad hoc reader exception.
