"""Offline tests for the natural-language agent (SPEC 4.8).

The suite never touches the network: ``answer`` is forced into offline mode
and the LLM path is exercised with ``httpx.MockTransport``. A deterministic
forecaster double keeps the tests fast while exposing exactly the public API
used by the agent (``products``, ``branches``, ``forecast`` and ``recommend``).
"""

from __future__ import annotations

import json
from datetime import date, timedelta

import httpx
import pytest
from src.agent import agent as agent_module
from src.agent import cli as cli_module
from src.agent.agent import answer
from src.config import Settings


class StubForecaster:
    """Tiny deterministic forecaster with the same public API as ``Forecaster``."""

    model_version = "stub-1.0.0"

    def __init__(self) -> None:
        self._products = [
            {
                "product_id": "P-001",
                "name": "Hojuela de PET reciclado transparente",
                "category": "Materia Prima",
                "unit": "kg",
                "lead_time_days": 6,
            },
            {
                "product_id": "P-002",
                "name": "Bolsa de basura PEAD 60 x 90 cm",
                "category": "Producto Terminado",
                "unit": "pza",
                "lead_time_days": 5,
            },
            {
                "product_id": "P-003",
                "name": "Tarima de plástico reciclado",
                "category": "Producto Terminado",
                "unit": "pza",
                "lead_time_days": 10,
            },
            {
                "product_id": "P-004",
                "name": "Caja apilable de plástico reciclado",
                "category": "Producto Terminado",
                "unit": "pza",
                "lead_time_days": 8,
            },
            {
                "product_id": "P-005",
                "name": "Botella PET post-consumo compactada",
                "category": "Materia Prima",
                "unit": "kg",
                "lead_time_days": 8,
            },
        ]
        self._branches = [
            {
                "branch_id": "B-01",
                "business_id": "BIZ-001",
                "name": "Matriz Centro",
                "region": "Centro",
                "units_sold": 500.0,
            },
            {
                "branch_id": "B-02",
                "business_id": "BIZ-001",
                "name": "Sucursal Norte",
                "region": "Norte",
                "units_sold": 900.0,
            },
            {
                "branch_id": "B-03",
                "business_id": "BIZ-001",
                "name": "Bodega Sur",
                "region": "Sur",
                "units_sold": 100.0,
            },
        ]

    def products(self) -> list[dict]:
        return [dict(product) for product in self._products]

    def branches(self) -> list[dict]:
        return [dict(branch) for branch in self._branches]

    def model_info(self) -> dict:
        return {"model": "stub_global", "model_version": self.model_version}

    def _require(self, product_id: str, branch_id: str) -> None:
        if product_id not in {product["product_id"] for product in self._products}:
            raise KeyError(f"Unknown product_id '{product_id}'")
        if branch_id not in {branch["branch_id"] for branch in self._branches}:
            raise KeyError(f"Unknown branch_id '{branch_id}'")

    def forecast(self, product_id: str, branch_id: str, horizon_days: int = 28) -> dict:
        self._require(product_id, branch_id)
        if not 1 <= horizon_days <= 90:
            raise ValueError("horizon_days must be between 1 and 90")
        start = date(2026, 1, 1)
        points = [
            {
                "date": (start + timedelta(days=offset)).isoformat(),
                "p50": round(1.0 + offset * 0.1, 2),
                "p10": 0.5,
                "p90": 2.0,
            }
            for offset in range(horizon_days)
        ]
        return {
            "product_id": product_id,
            "branch_id": branch_id,
            "model": "stub_global",
            "model_version": self.model_version,
            "generated_at": "2026-01-01T00:00:00Z",
            "horizon_days": horizon_days,
            "points": points,
        }

    def recommend(
        self,
        product_id: str,
        branch_id: str,
        horizon_days: int = 30,
        service_level: float = 0.95,
    ) -> dict:
        self._require(product_id, branch_id)
        if not 1 <= horizon_days <= 90:
            raise ValueError("horizon_days must be between 1 and 90")
        return {
            "product_id": product_id,
            "branch_id": branch_id,
            "model": "stub_global",
            "model_version": self.model_version,
            "generated_at": "2026-01-01T00:00:00Z",
            "horizon_days": horizon_days,
            "service_level": service_level,
            "lead_time_days": 7,
            "daily_mean": 0.42,
            "sigma": 0.21,
            "daily_mean_units": 0.42,
            "sigma_units": 0.21,
            "z_score": 1.6449,
            "safety_stock": 1.11,
            "reorder_point": 45.67,
            "current_stock": 33.33,
            "reorder_quantity": 12.34,
            "days_of_cover": 79.36,
            "rationale": "stub rationale",
        }


@pytest.fixture()
def stub() -> StubForecaster:
    return StubForecaster()


@pytest.fixture(autouse=True)
def _force_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Guarantee no test can reach an LLM endpoint by accident."""
    monkeypatch.setattr(agent_module, "get_settings", lambda: Settings(openai_api_key=None))


def test_recommendation_intent_returns_reorder_quantity(stub: StubForecaster) -> None:
    result = answer("¿Cuánto debo pedir de la hojuela de PET reciclado transparente?", stub)

    assert result["mode"] == "offline"
    assert result["tool_calls"] == []
    assert result["data"]["reorder_quantity"] == 12.34
    assert result["data"]["horizon_days"] == 30
    assert result["data"]["branch_id"] == "B-02"  # highest-volume default branch
    assert "12.34" in result["answer"]
    assert "stub-1.0.0" in result["answer"]
    assert "probabilístico" in result["answer"]
    assert "No indicaste sucursal" in result["answer"]
    json.dumps(result, ensure_ascii=False)


def test_forecast_intent_returns_points(stub: StubForecaster) -> None:
    result = answer(
        "Pronóstico de demanda para 2 semanas de la caja apilable de plástico reciclado",
        stub,
    )

    assert result["mode"] == "offline"
    assert result["data"]["horizon_days"] == 14
    assert len(result["data"]["points"]) == 14
    assert "14 días" in result["answer"]
    assert "stub-1.0.0" in result["answer"]
    assert "probabilístico" in result["answer"]


def test_unknown_product_asks_for_clarification(stub: StubForecaster) -> None:
    result = answer("¿Cuánto debo pedir de tornillos inoxidables?", stub)

    assert result["mode"] == "offline"
    assert result["data"] is None
    assert result["tool_calls"] == []
    assert "refieres" in result["answer"].lower()
    assert result["answer"].rstrip().endswith(".")
    listed = [product["name"] for product in stub.products() if product["name"] in result["answer"]]
    assert 1 <= len(listed) <= 5


def test_accent_and_case_normalization(stub: StubForecaster) -> None:
    result = answer(
        "CUÁNTO DEBO PEDIR DEL PRODUCTO CAJA APILABLE DE PLASTICO RECICLADO",
        stub,
    )

    assert result["mode"] == "offline"
    assert result["data"]["product_id"] == "P-004"
    assert result["data"]["reorder_quantity"] == 12.34


def test_horizon_regex_and_defaults(stub: StubForecaster) -> None:
    ten_days = answer("¿Cuánto debo pedir de la tarima para los próximos 10 días?", stub)
    assert ten_days["data"]["horizon_days"] == 10

    three_weeks = answer("Pronóstico a 3 semanas de la tarima de plástico reciclado", stub)
    assert three_weeks["data"]["horizon_days"] == 21

    recommend_default = answer("¿Cuánto debo pedir de la tarima?", stub)
    assert recommend_default["data"]["horizon_days"] == 30

    forecast_default = answer("Pronóstico de la tarima de plástico reciclado", stub)
    assert forecast_default["data"]["horizon_days"] == 28

    clamped = answer("Pronóstico a 120 días de la tarima de plástico reciclado", stub)
    assert clamped["data"]["horizon_days"] == 90
    assert "máximo" in clamped["answer"]


def test_branch_resolution_by_name_id_and_default(stub: StubForecaster) -> None:
    by_name = answer("¿Cuánto debo pedir de la tarima en Matriz Centro?", stub)
    assert by_name["data"]["branch_id"] == "B-01"

    by_id = answer("¿Cuánto debo pedir de la tarima en B-03?", stub)
    assert by_id["data"]["branch_id"] == "B-03"

    default = answer("¿Cuánto debo pedir de la tarima?", stub)
    assert default["data"]["branch_id"] == "B-02"


def test_llm_mode_executes_local_tools(stub: StubForecaster) -> None:
    requests: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        requests.append(payload)
        tool_messages = [message for message in payload["messages"] if message["role"] == "tool"]
        if not tool_messages:
            tool_call = {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "recommendation",
                    "arguments": json.dumps(
                        {
                            "product": "Caja apilable de plástico reciclado",
                            "branch": "Matriz Centro",
                            "horizon_days": 15,
                        }
                    ),
                },
            }
            message = {"role": "assistant", "content": None, "tool_calls": [tool_call]}
        else:
            message = {"role": "assistant", "content": "Recomendación lista."}
        return httpx.Response(200, json={"choices": [{"message": message}]})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    settings = Settings(openai_api_key="test-key")
    result = agent_module._llm_answer(
        "¿Cuánto debo pedir de la caja apilable?", stub, settings, client=client
    )

    assert result["mode"] == "llm"
    assert result["answer"] == "Recomendación lista."
    assert result["data"]["reorder_quantity"] == 12.34
    assert result["data"]["horizon_days"] == 15
    assert result["tool_calls"][0]["name"] == "recommendation"
    assert len(requests) == 2
    assert "tools" in requests[0]


def test_llm_failure_falls_back_to_offline(
    stub: StubForecaster, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(agent_module, "get_settings", lambda: Settings(openai_api_key="test-key"))

    class _ExplodingClient:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("network disabled in tests")

    monkeypatch.setattr(agent_module.httpx, "Client", _ExplodingClient)

    result = answer("¿Cuánto debo pedir de la tarima?", stub)
    assert result["mode"] == "offline_fallback"
    assert result["data"]["reorder_quantity"] == 12.34
    assert "12.34" in result["answer"]


def test_empty_question_is_rejected(stub: StubForecaster) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        answer("   ", stub)
    with pytest.raises(TypeError, match="string"):
        answer(None, stub)  # type: ignore[arg-type]


def test_cli_success_and_error_exit_codes(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        cli_module,
        "answer",
        lambda question: {"answer": f"Respuesta para: {question}", "mode": "offline"},
    )
    assert cli_module.main(["¿Cuánto", "debo", "pedir?"]) == 0
    captured = capsys.readouterr()
    assert "Respuesta para: ¿Cuánto debo pedir?" in captured.out

    def _boom(question: str) -> dict:
        raise RuntimeError("model missing")

    monkeypatch.setattr(cli_module, "answer", _boom)
    assert cli_module.main(["pregunta"]) == 1
    captured = capsys.readouterr()
    assert "Error: model missing" in captured.err
