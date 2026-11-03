SHELL := /bin/bash
PY := .venv/bin/python
PIP := .venv/bin/pip
QUICK_DATA := .ci/data
QUICK_ARTIFACTS := .ci/artifacts

.PHONY: venv install data data-quick train train-quick serve test lint fmt monitor agent smoke docker-build clean

venv:
	python3 -m venv .venv
	$(PIP) install -U pip

install: venv
	$(PIP) install -e ".[dev]"

data:
	$(PY) -m src.pipelines.synthetic_generator --out data --days 1096 --products 20 --branches 3 --seed 42
	$(PY) -m src.pipelines.etl --raw data/raw --out data/processed/demand_daily.csv

data-quick:
	$(PY) -m src.pipelines.synthetic_generator --out $(QUICK_DATA) --days 120 --products 6 --branches 2 --seed 7
	$(PY) -m src.pipelines.etl --raw $(QUICK_DATA)/raw --out $(QUICK_DATA)/processed/demand_daily.csv

train:
	$(PY) -m src.pipelines.train --data data --artifacts artifacts

train-quick:
	$(PY) -m src.pipelines.train --data $(QUICK_DATA) --artifacts $(QUICK_ARTIFACTS) --quick

serve:
	$(PY) -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check src tests

fmt:
	$(PY) -m ruff check --fix src tests
	$(PY) -m ruff format src tests

monitor:
	$(PY) -m src.monitoring.drift --data data/processed/demand_daily.csv --out artifacts/monitoring/drift_report.json

agent:
	$(PY) -m src.agent.cli "$(Q)"

smoke: data-quick train-quick
	$(PY) -c "from src.pipelines.forecast import Forecaster; f = Forecaster.load('$(QUICK_DATA)', '$(QUICK_ARTIFACTS)'); print('forecaster ready')"

docker-build:
	docker build -t asviin-ml-forecasting:local .

clean:
	rm -rf .pytest_cache .ruff_cache .ci
	find src tests -type d -name __pycache__ -exec rm -rf {} +
