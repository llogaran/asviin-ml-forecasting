# Backtest report — xgboost_global

Generated: 2026-11-03T19:42:11Z

- Dataset: `data/processed/demand_daily.csv`
- SHA-256: `1666735c681989446edc88903376fbbb8a8bd91b2312f6c272793c99d789b5fd`
- Rows: 65760 | Series: 60 | Dates: 2023-01-01 → 2025-12-31
- Holdout: 2025-12-04 → 2025-12-31 (60 series, 1680 rows, holdout_days=28)

MASE is scaled by the naive-seasonal MAE on the same holdout.

| metric | xgboost_global | naive_seasonal | croston_sba |
| --- | ---: | ---: | ---: |
| MAE | 1.9205 | 2.4713 | 2.2163 |
| WAPE | 0.5273 | 0.6785 | 0.6085 |
| MASE | 0.7771 | 1.0000 | 0.8968 |
| coverage_90 | 0.8375 | — | — |

## Per-series sample

| business | branch | product | rows | mae | naive_mae | croston_mae | coverage_90 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| BIZ-001 | B-01 | P-001 | 28 | 1.3329 | 1.8879 | 1.3548 | 0.6786 |
| BIZ-001 | B-01 | P-002 | 28 | 1.2901 | 1.4543 | 1.4490 | 0.7500 |
| BIZ-001 | B-01 | P-003 | 28 | 0.9286 | 1.4643 | 1.5182 | 0.9286 |
| BIZ-001 | B-01 | P-004 | 28 | 0.2500 | 0.5000 | 0.8530 | 0.9286 |
| BIZ-001 | B-01 | P-005 | 28 | 0.8791 | 1.7054 | 1.1658 | 0.9286 |
| BIZ-001 | B-01 | P-006 | 28 | 0.2811 | 0.5621 | 0.5845 | 0.9286 |
| BIZ-001 | B-01 | P-007 | 28 | 6.4352 | 6.6071 | 9.0895 | 0.6786 |
| BIZ-001 | B-01 | P-008 | 28 | 4.2924 | 4.7500 | 5.9398 | 0.7143 |
| BIZ-001 | B-01 | P-009 | 28 | 1.5925 | 1.8929 | 1.8285 | 0.7143 |
| BIZ-001 | B-01 | P-010 | 28 | 3.2857 | 4.8571 | 3.8644 | 0.8214 |

Only the first 10 of 60 series are shown; the full panel is scored in `backtest.json`.
