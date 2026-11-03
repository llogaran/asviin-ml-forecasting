"""Spanish question answering over the forecasting service (SPEC 4.8).

Two execution modes are supported:

* **offline** (the default, no network): a small deterministic pipeline
  normalizes Spanish accents/case, detects whether the user wants a reorder
  recommendation or a demand forecast, resolves the product and branch
  mentioned in the question, parses the horizon and calls
  :class:`~src.pipelines.forecast.Forecaster`. The Spanish answer is rendered
  from the numbers returned by the forecaster; nothing is invented.
* **LLM**: only when ``OPENAI_API_KEY`` is set. An OpenAI-compatible chat
  completion can call three local tools (``list_products``, ``forecast`` and
  ``recommendation``) that are executed against the same forecaster, with at
  most three tool rounds. Any HTTP/parsing/validation error falls back to the
  offline mode and reports ``mode="offline_fallback"``, so the CLI is always
  usable without connectivity.

Product names are matched with :class:`difflib.SequenceMatcher` (cutoff 0.55).
Because users usually mention only a fragment of a product name, the ratio is
computed against sliding windows of the question instead of the whole sentence.
A word that appears in exactly one product name (for example "tarima") is also
accepted as a strong signal, because one word alone can never reach the cutoff
against a long full name. Unknown products never raise: the answer becomes a
clarification question that lists up to five candidate names and ``data`` is
``None``. Branch resolution is by id or name and falls back to the
highest-volume branch, which is the only branch-level default exposed
(indirectly) by the forecaster.
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from difflib import SequenceMatcher
from typing import TYPE_CHECKING, Any

import httpx

from src.config import Settings, get_settings

if TYPE_CHECKING:
    from src.pipelines.forecast import Forecaster

LOGGER = logging.getLogger(__name__)

PRODUCT_MATCH_CUTOFF = 0.55
AMBIGUITY_MARGIN = 0.05
MAX_CANDIDATES = 5
#: A single distinctive word such as "tarima" identifies a product even though
#: its difflib ratio against the full name is below the cutoff.
UNIQUE_TOKEN_SCORE = 0.75
MIN_NAME_TOKEN_LENGTH = 4
DEFAULT_RECOMMEND_HORIZON = 30
DEFAULT_FORECAST_HORIZON = 28
MIN_HORIZON_DAYS = 1
MAX_HORIZON_DAYS = 90
LLM_MAX_TOOL_ROUNDS = 3
LLM_TIMEOUT_SECONDS = 30.0

_CAVEAT = (
    "Nota: el pronóstico es probabilístico (intervalos p10–p90); "
    "es una estimación, no una garantía."
)

_RECOMMEND_STEMS = (
    "ped",
    "compr",
    "reabastec",
    "repon",
    "reorden",
    "recomend",
    "recomiend",
    "suger",
    "inventari",
    "existenci",
    "stock",
    "surt",
)
_FORECAST_STEMS = (
    "pronostic",
    "predic",
    "predec",
    "forecast",
    "demand",
    "venta",
    "vend",
    "proyect",
    "estim",
)
_GENERIC_BRANCH_TOKENS = {"sucursal", "branch"}
#: Words that must never make a product "unique" on their own: they are common
#: in questions (intent, dates, courtesy) and in product names ("para", "con").
_GENERIC_NAME_TOKENS = frozenset(
    {
        "para",
        "como",
        "cual",
        "cuales",
        "cuanto",
        "cuanta",
        "cuantos",
        "cuantas",
        "donde",
        "cuando",
        "este",
        "esta",
        "estos",
        "estas",
        "ser",
        "son",
        "hay",
        "tiene",
        "tienen",
        "debo",
        "debe",
        "deben",
        "pedir",
        "pedido",
        "comprar",
        "compra",
        "necesito",
        "quiero",
        "puedes",
        "producto",
        "productos",
        "sucursal",
        "inventario",
        "stock",
        "demanda",
        "pronostico",
        "forecast",
        "unidades",
        "unidad",
        "hola",
        "favor",
        "ayuda",
        "gracias",
        "sobre",
        "entre",
        "desde",
        "hasta",
        "dias",
        "semana",
        "semanas",
        "meses",
        "hoy",
        "manana",
    }
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_HORIZON_RE = re.compile(r"\b(\d+)\s*(dias?|semanas?|sem)\b")

_LLM_SYSTEM_PROMPT = (
    "Eres el asistente de inventarios de Asviin. Responde en español, de forma breve. "
    "Usa siempre las herramientas para obtener cifras reales; nunca inventes números. "
    "Si no identificas el producto o la sucursal, pide aclaración. "
    "Incluye la versión del modelo y aclara que el pronóstico es probabilístico."
)

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "list_products",
            "description": "List catalog products, optionally filtered by a name fragment.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Optional product name fragment.",
                    }
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "forecast",
            "description": "Demand forecast (p10/p50/p90 per day) for a product and branch.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product": {
                        "type": "string",
                        "description": "Product name or product_id (e.g. 'P-003').",
                    },
                    "branch": {
                        "type": "string",
                        "description": "Optional branch name or branch_id.",
                    },
                    "horizon_days": {
                        "type": "integer",
                        "minimum": MIN_HORIZON_DAYS,
                        "maximum": MAX_HORIZON_DAYS,
                    },
                },
                "required": ["product"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recommendation",
            "description": "Reorder recommendation for a product and branch.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product": {
                        "type": "string",
                        "description": "Product name or product_id (e.g. 'P-003').",
                    },
                    "branch": {
                        "type": "string",
                        "description": "Optional branch name or branch_id.",
                    },
                    "horizon_days": {
                        "type": "integer",
                        "minimum": MIN_HORIZON_DAYS,
                        "maximum": MAX_HORIZON_DAYS,
                    },
                },
                "required": ["product"],
                "additionalProperties": False,
            },
        },
    },
]


# --------------------------------------------------------------------- helpers
def _normalize(text: str) -> str:
    """Lowercase, strip accents and punctuation, and collapse whitespace."""
    decomposed = unicodedata.normalize("NFKD", str(text).casefold())
    plain = "".join(char for char in decomposed if not unicodedata.combining(char))
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", plain)).strip()


def _detect_intent(question_norm: str) -> str:
    """Return ``"recommend"`` or ``"forecast"`` for a normalized question.

    Stem matches are counted instead of exact words so conjugations such as
    "pedir"/"pidiendo" or "recomendar"/"recomienda" keep working. A question
    with no inventory/forecast keyword defaults to a forecast, which is the
    safest read of a vague demand question; ties favour the actionable
    recommendation.
    """
    tokens = _TOKEN_RE.findall(question_norm)
    recommend_hits = sum(
        1 for token in tokens if any(token.startswith(stem) for stem in _RECOMMEND_STEMS)
    )
    forecast_hits = sum(
        1 for token in tokens if any(token.startswith(stem) for stem in _FORECAST_STEMS)
    )
    if recommend_hits and recommend_hits >= forecast_hits:
        return "recommend"
    return "forecast"


def _parse_horizon(question_norm: str, default: int) -> tuple[int, bool]:
    """Parse "N días"/"N semanas"; return ``(horizon, was_clamped)``."""
    match = _HORIZON_RE.search(question_norm)
    if match is None:
        return default, False
    amount = int(match.group(1))
    unit = match.group(2)
    days = amount * 7 if unit.startswith("sem") else amount
    if days < MIN_HORIZON_DAYS:
        return default, False
    if days > MAX_HORIZON_DAYS:
        return MAX_HORIZON_DAYS, True
    return days, False


def _meaningful_name_tokens(product: dict[str, Any]) -> set[str]:
    """Return the distinctive (non-generic, length-filtered) tokens of a name."""
    name = _normalize(str(product.get("name", "")))
    return {
        token
        for token in name.split()
        if len(token) >= MIN_NAME_TOKEN_LENGTH and token not in _GENERIC_NAME_TOKENS
    }


def _unique_name_tokens(products: list[dict[str, Any]]) -> set[str]:
    """Tokens present in exactly one product name (e.g. "tarima", "hojuela").

    A user who only says "la tarima" is referring unambiguously to the single
    product whose name contains that word; comparing that one word with the
    full name through difflib would never cross the 0.55 cutoff.
    """
    counts: dict[str, int] = {}
    for product in products:
        for token in _meaningful_name_tokens(product):
            counts[token] = counts.get(token, 0) + 1
    return {token for token, count in counts.items() if count == 1}


def _product_score(question_norm: str, product: dict[str, Any], unique_tokens: set[str]) -> float:
    """Fuzzy score in ``[0, 1]`` for one product against a normalized question."""
    padded_question = f" {question_norm} "
    product_id = _normalize(str(product.get("product_id", "")))
    if product_id:
        compact_id = product_id.replace(" ", "")
        compact_question = question_norm.replace(" ", "")
        if f" {product_id} " in padded_question or (compact_id and compact_id in compact_question):
            return 1.0

    name = _normalize(str(product.get("name", "")))
    if not name:
        return 0.0
    if f" {name} " in padded_question:
        return 1.0

    question_tokens = question_norm.split()
    name_tokens = name.split()
    if not question_tokens or not name_tokens:
        return 0.0

    best = SequenceMatcher(None, name, question_norm).ratio()
    widths = range(max(1, len(name_tokens) - 2), len(name_tokens) + 3)
    for width in sorted({width for width in widths if width <= len(question_tokens)}):
        for start in range(len(question_tokens) - width + 1):
            window = " ".join(question_tokens[start : start + width])
            best = max(best, SequenceMatcher(None, name, window).ratio())
    if unique_tokens & set(question_tokens) & _meaningful_name_tokens(product):
        best = max(best, UNIQUE_TOKEN_SCORE)
    return best


def _rank_products(
    question_norm: str, products: list[dict[str, Any]]
) -> list[tuple[dict[str, Any], float]]:
    """Rank products by fuzzy score with a deterministic name/id tie-break."""
    unique_tokens = _unique_name_tokens(products)
    scored = [
        (product, _product_score(question_norm, product, unique_tokens)) for product in products
    ]
    scored.sort(
        key=lambda item: (
            -item[1],
            _normalize(str(item[0].get("name", ""))),
            str(item[0].get("product_id", "")),
        )
    )
    return scored


def _resolve_product(
    question_norm: str, products: list[dict[str, Any]]
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], bool]:
    """Return ``(product, candidates, ambiguous)`` for a normalized question.

    ``product`` is ``None`` when no name reaches the difflib cutoff or when the
    two best candidates are too close to choose safely; in both cases the
    caller must ask for clarification instead of guessing.
    """
    if not products:
        return None, [], False
    ranked = _rank_products(question_norm, products)
    candidates = [product for product, _ in ranked[:MAX_CANDIDATES]]
    best_product, best_score = ranked[0]
    if best_score < PRODUCT_MATCH_CUTOFF:
        return None, candidates, False
    if len(ranked) > 1:
        runner_up = ranked[1][1]
        if runner_up >= PRODUCT_MATCH_CUTOFF and best_score - runner_up < AMBIGUITY_MARGIN:
            return None, candidates, True
    return best_product, candidates, False


def _branch_volumes(forecaster: Any, branches: list[dict[str, Any]]) -> dict[str, float]:
    """Return total units sold per branch.

    The public forecaster API does not expose branch volume, so the internal
    daily panel is used when available. Lightweight test doubles can instead
    put ``units_sold`` in the branch dict; when neither source exists the
    caller falls back to the first branch id.
    """
    volumes: dict[str, float] = {}
    daily = getattr(forecaster, "_daily", None)
    columns = getattr(daily, "columns", ())
    if daily is not None and {"branch_id", "units_sold"} <= set(columns):
        try:
            totals = daily.groupby("branch_id")["units_sold"].sum()
            volumes = {str(key): float(value) for key, value in totals.items()}
        except (KeyError, TypeError, ValueError):
            volumes = {}
    if not volumes:
        for branch in branches:
            branch_id = str(branch.get("branch_id", ""))
            try:
                volumes[branch_id] = float(branch.get("units_sold", 0.0) or 0.0)
            except (TypeError, ValueError):
                volumes[branch_id] = 0.0
    return volumes


def _default_branch(forecaster: Any, branches: list[dict[str, Any]]) -> tuple[dict[str, Any], bool]:
    """Pick the highest-volume branch; return ``(branch, chosen_by_volume)``."""
    ordered = sorted(branches, key=lambda branch: str(branch.get("branch_id", "")))
    volumes = _branch_volumes(forecaster, ordered)
    if volumes:
        ranked = sorted(
            ordered,
            key=lambda branch: (
                -volumes.get(str(branch.get("branch_id", "")), 0.0),
                str(branch.get("branch_id", "")),
            ),
        )
        if volumes.get(str(ranked[0].get("branch_id", "")), 0.0) > 0.0:
            return ranked[0], True
    return ordered[0], False


def _resolve_branch(question_norm: str, forecaster: Any) -> tuple[dict[str, Any], bool]:
    """Resolve a branch by id/name; return ``(branch, used_default)``."""
    branches = forecaster.branches()
    if not branches:
        raise KeyError("The forecaster has no branches loaded; run `make data` first.")
    padded_question = f" {question_norm} "

    for branch in branches:
        branch_id = _normalize(str(branch.get("branch_id", "")))
        if branch_id and f" {branch_id} " in padded_question:
            return branch, False
    for branch in branches:
        name = _normalize(str(branch.get("name", "")))
        if name and f" {name} " in padded_question:
            return branch, False

    question_tokens = set(question_norm.split())
    best_branch: dict[str, Any] | None = None
    best_overlap = 0
    for branch in branches:
        name_tokens = {
            token
            for token in _normalize(str(branch.get("name", ""))).split()
            if len(token) >= 3 and token not in _GENERIC_BRANCH_TOKENS
        }
        overlap = len(name_tokens & question_tokens)
        if overlap > best_overlap:
            best_branch, best_overlap = branch, overlap
    if best_branch is not None:
        return best_branch, False
    return _default_branch(forecaster, branches)


def _number(value: Any) -> str:
    """Format a numeric value with two decimals for the answer text."""
    return f"{float(value):.2f}"


def _clarification(candidates: list[dict[str, Any]], ambiguous: bool) -> str:
    """Ask the user which product they mean, listing up to five candidates."""
    if not candidates:
        return (
            "No hay productos disponibles en el catálogo. "
            "¿Quieres intentarlo de nuevo cuando el catálogo esté cargado?"
        )
    options = "; ".join(
        f"{product.get('name', product.get('product_id', '?'))} ({product.get('product_id', '?')})"
        for product in candidates[:MAX_CANDIDATES]
    )
    lead = (
        "Hay varios productos parecidos en tu pregunta. ¿A cuál de ellos te refieres?"
        if ambiguous
        else "No pude identificar con certeza el producto de tu pregunta. "
        "¿A cuál producto te refieres?"
    )
    return f"{lead} Opciones: {options}."


def _compose_recommendation(
    product: dict[str, Any],
    branch: dict[str, Any],
    result: dict[str, Any],
    used_default_branch: bool,
    horizon_clamped: bool,
) -> str:
    """Render the Spanish recommendation answer from forecaster numbers."""
    horizon = int(result["horizon_days"])
    level = float(result["service_level"])
    sentence = (
        f"Para {product['name']} ({product['product_id']}) en {branch['name']} "
        f"({branch['branch_id']}), con un horizonte de {horizon} días y un nivel de "
        f"servicio del {level:.0%}, la cantidad recomendada a pedir es "
        f"{_number(result['reorder_quantity'])} unidades."
    )
    detail = (
        f"El punto de pedido es {_number(result['reorder_point'])} unidades, el stock "
        f"actual es {_number(result['current_stock'])} unidades y el plazo de entrega "
        f"es de {int(result['lead_time_days'])} días; la demanda pronosticada promedia "
        f"{_number(result['daily_mean'])} unidades/día."
    )
    notes = []
    if used_default_branch:
        notes.append(
            f"No indicaste sucursal, así que usé {branch['name']} ({branch['branch_id']})."
        )
    if horizon_clamped:
        notes.append(f"Ajusté el horizonte al máximo soportado de {MAX_HORIZON_DAYS} días.")
    notes.append(f"Versión del modelo: {result['model_version']}.")
    notes.append(_CAVEAT)
    return " ".join([sentence, detail, *notes])


def _compose_forecast(
    product: dict[str, Any],
    branch: dict[str, Any],
    result: dict[str, Any],
    used_default_branch: bool,
    horizon_clamped: bool,
) -> str:
    """Render the Spanish forecast answer from forecaster numbers."""
    points = result["points"]
    horizon = int(result["horizon_days"])
    count = len(points)
    mean_daily = sum(float(point["p50"]) for point in points) / count
    total_p50 = sum(float(point["p50"]) for point in points)
    total_p10 = sum(float(point["p10"]) for point in points)
    total_p90 = sum(float(point["p90"]) for point in points)
    first_date = points[0]["date"]
    last_date = points[-1]["date"]
    sentence = (
        f"Pronóstico a {horizon} días para {product['name']} ({product['product_id']}) "
        f"en {branch['name']} ({branch['branch_id']}): la demanda media diaria estimada "
        f"es de {_number(mean_daily)} unidades (p50), con un total de "
        f"{_number(total_p50)} unidades y un rango p10–p90 de {_number(total_p10)} a "
        f"{_number(total_p90)} unidades entre {first_date} y {last_date}."
    )
    notes = []
    if used_default_branch:
        notes.append(
            f"No indicaste sucursal, así que usé {branch['name']} ({branch['branch_id']})."
        )
    if horizon_clamped:
        notes.append(f"Ajusté el horizonte al máximo soportado de {MAX_HORIZON_DAYS} días.")
    notes.append(f"Versión del modelo: {result['model_version']}.")
    notes.append(_CAVEAT)
    return " ".join([sentence, *notes])


def _offline_answer(question: str, forecaster: Forecaster) -> dict[str, Any]:
    """Deterministic offline answer: normalize, resolve, call, compose."""
    question_norm = _normalize(question)
    intent = _detect_intent(question_norm)
    product, candidates, ambiguous = _resolve_product(question_norm, forecaster.products())
    if product is None:
        return {
            "answer": _clarification(candidates, ambiguous),
            "mode": "offline",
            "tool_calls": [],
            "data": None,
        }

    branch, used_default_branch = _resolve_branch(question_norm, forecaster)
    default_horizon = (
        DEFAULT_RECOMMEND_HORIZON if intent == "recommend" else DEFAULT_FORECAST_HORIZON
    )
    horizon, horizon_clamped = _parse_horizon(question_norm, default_horizon)
    if intent == "recommend":
        data = forecaster.recommend(
            product["product_id"], branch["branch_id"], horizon_days=horizon
        )
        text = _compose_recommendation(product, branch, data, used_default_branch, horizon_clamped)
    else:
        data = forecaster.forecast(product["product_id"], branch["branch_id"], horizon_days=horizon)
        text = _compose_forecast(product, branch, data, used_default_branch, horizon_clamped)
    return {"answer": text, "mode": "offline", "tool_calls": [], "data": data}


def _load_forecaster(settings: Settings) -> Forecaster:
    """Lazily import and load the real forecaster from settings paths."""
    from src.pipelines.forecast import Forecaster

    return Forecaster.load(settings.data_dir, settings.artifacts_dir)


# ------------------------------------------------------------------- LLM tools
def _coerce_horizon(value: Any, default: int) -> int:
    """Best-effort integer horizon for a model-provided tool argument."""
    try:
        horizon = int(value)
    except (TypeError, ValueError):
        return default
    if horizon < MIN_HORIZON_DAYS:
        return default
    return min(horizon, MAX_HORIZON_DAYS)


def _tool_product_candidates(products: list[dict[str, Any]]) -> list[str]:
    return [str(product.get("name", product.get("product_id", ""))) for product in products][
        :MAX_CANDIDATES
    ]


def _execute_tool(
    name: str, arguments: str | dict[str, Any], forecaster: Forecaster
) -> dict[str, Any]:
    """Execute a tool call locally against the forecaster (never network)."""
    try:
        parsed = json.loads(arguments) if isinstance(arguments, str) else dict(arguments or {})
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        return {"error": f"Invalid tool arguments: {exc}"}
    if not isinstance(parsed, dict):
        return {"error": "Tool arguments must be a JSON object."}

    if name == "list_products":
        products = forecaster.products()
        query = str(parsed.get("query") or "").strip()
        if not query:
            return {"products": products}
        normalized_query = _normalize(query)
        ranked = _rank_products(normalized_query, products)
        matches = [product for product, score in ranked if score >= PRODUCT_MATCH_CUTOFF]
        if not matches:
            return {
                "error": f"No product matched '{query}'.",
                "candidates": _tool_product_candidates(
                    [product for product, _ in ranked[:MAX_CANDIDATES]]
                ),
            }
        return {"products": matches[:MAX_CANDIDATES]}

    if name not in {"forecast", "recommendation"}:
        return {"error": f"Unknown tool '{name}'. Use list_products, forecast or recommendation."}

    query = str(
        parsed.get("product") or parsed.get("product_id") or parsed.get("product_name") or ""
    ).strip()
    products = forecaster.products()
    if not query:
        return {
            "error": "Missing 'product' argument (product name or product_id).",
            "candidates": _tool_product_candidates(products),
        }
    product, candidates, ambiguous = _resolve_product(_normalize(query), products)
    if product is None:
        return {
            "error": (
                "Several products match; ask the user to choose."
                if ambiguous
                else f"No product matched '{query}'."
            ),
            "candidates": _tool_product_candidates(candidates),
        }

    branch_query = str(
        parsed.get("branch") or parsed.get("branch_id") or parsed.get("branch_name") or ""
    ).strip()
    branch, _ = _resolve_branch(_normalize(branch_query), forecaster)
    default_horizon = (
        DEFAULT_RECOMMEND_HORIZON if name == "recommendation" else DEFAULT_FORECAST_HORIZON
    )
    horizon = _coerce_horizon(parsed.get("horizon_days"), default_horizon)
    if name == "recommendation":
        return forecaster.recommend(
            product["product_id"], branch["branch_id"], horizon_days=horizon
        )
    return forecaster.forecast(product["product_id"], branch["branch_id"], horizon_days=horizon)


def _extract_message(body: Any) -> dict[str, Any]:
    """Validate and return the first chat-completion message object."""
    if not isinstance(body, dict):
        raise RuntimeError("LLM response is not a JSON object")
    try:
        message = body["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("LLM response has no choices[0].message") from exc
    if not isinstance(message, dict):
        raise RuntimeError("LLM response message is not an object")
    return message


def _parse_tool_call(call: Any, round_index: int, call_index: int) -> tuple[str, str, str]:
    """Return ``(call_id, tool_name, arguments_json)`` for one tool call."""
    if not isinstance(call, dict):
        raise RuntimeError("Malformed tool call in LLM response")
    function = call.get("function")
    if not isinstance(function, dict):
        raise RuntimeError("Tool call has no function definition")
    name = str(function.get("name") or "").strip()
    if not name:
        raise RuntimeError("Tool call has no function name")
    arguments = function.get("arguments") or "{}"
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments, ensure_ascii=False)
    call_id = str(call.get("id") or f"call_{round_index}_{call_index}")
    return call_id, name, arguments


def _llm_answer(
    question: str,
    forecaster: Forecaster,
    settings: Settings,
    client: httpx.Client | None = None,
) -> dict[str, Any]:
    """Run the OpenAI-compatible tool loop; raise on any failure.

    ``client`` is injectable so tests can use :class:`httpx.MockTransport`
    without touching the network. The caller (:func:`answer`) converts any
    exception into the deterministic offline fallback.
    """
    owns_client = client is None
    if client is None:
        client = httpx.Client(timeout=httpx.Timeout(LLM_TIMEOUT_SECONDS, connect=5.0))
    url = f"{settings.openai_base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {settings.openai_api_key}",
        "Content-Type": "application/json",
    }
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _LLM_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    tool_calls_log: list[dict[str, Any]] = []
    data: dict[str, Any] | None = None

    try:
        for round_index in range(LLM_MAX_TOOL_ROUNDS + 1):
            payload: dict[str, Any] = {
                "model": settings.openai_model,
                "messages": messages,
                "temperature": 0,
            }
            if round_index < LLM_MAX_TOOL_ROUNDS:
                payload["tools"] = TOOL_DEFINITIONS
                payload["tool_choice"] = "auto"
            response = client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            message = _extract_message(response.json())
            tool_calls = message.get("tool_calls") or []
            if not tool_calls:
                content = message.get("content")
                if not isinstance(content, str) or not content.strip():
                    raise RuntimeError("LLM returned an empty answer")
                return {
                    "answer": content.strip(),
                    "mode": "llm",
                    "tool_calls": tool_calls_log,
                    "data": data,
                }
            if round_index >= LLM_MAX_TOOL_ROUNDS:
                raise RuntimeError("LLM exceeded the maximum number of tool rounds")

            messages.append(message)
            for call_index, call in enumerate(tool_calls):
                call_id, name, arguments = _parse_tool_call(call, round_index, call_index)
                try:
                    parsed_arguments: dict[str, Any] = json.loads(arguments)
                except (json.JSONDecodeError, TypeError, ValueError):
                    parsed_arguments = {}
                if not isinstance(parsed_arguments, dict):
                    parsed_arguments = {}
                result = _execute_tool(name, arguments, forecaster)
                tool_calls_log.append(
                    {"name": name, "arguments": parsed_arguments, "result": result}
                )
                data = result
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": name,
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )
        raise RuntimeError("LLM produced no final answer")
    finally:
        if owns_client:
            client.close()


# ----------------------------------------------------------------- public API
def answer(question: str, forecaster: Forecaster | None = None) -> dict[str, Any]:
    """Answer a Spanish inventory question.

    Parameters
    ----------
    question:
        Free-text Spanish question, e.g. ``"¿Cuánto debo pedir de la hojuela de PET?"``.
    forecaster:
        Optional preloaded forecaster (or a test double with the same public
        API). When omitted, :class:`~src.pipelines.forecast.Forecaster` is
        loaded from the configured data/artifacts directories.

    Returns
    -------
    dict
        ``{"answer": str, "mode": str, "tool_calls": list, "data": dict | None}``.
        ``mode`` is ``"offline"``, ``"llm"`` or ``"offline_fallback"``.
        ``data`` is the exact forecaster payload used for the answer; it is
        ``None`` when the assistant needs a clarification.
    """
    if not isinstance(question, str):
        raise TypeError("question must be a string")
    text = question.strip()
    if not text:
        raise ValueError("question must be a non-empty string")

    settings = get_settings()
    if forecaster is None:
        forecaster = _load_forecaster(settings)

    if settings.openai_api_key:
        try:
            return _llm_answer(text, forecaster, settings)
        except Exception as exc:  # noqa: BLE001 - any LLM failure must degrade gracefully
            LOGGER.warning("LLM agent failed (%s); falling back to offline mode", exc)
            result = _offline_answer(text, forecaster)
            result["mode"] = "offline_fallback"
            return result

    return _offline_answer(text, forecaster)
