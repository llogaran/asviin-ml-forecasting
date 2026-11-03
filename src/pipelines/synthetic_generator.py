"""Deterministic synthetic ERP dataset for a Mexican plastic recycler/transformer.

The ML project must never depend on real Asviin data, so this module is itself a
deliverable: it produces the six canonical raw CSVs (products, branches, recipes,
economic events, stock movements, pending sales) plus a ``seed.json`` with the
deterministic generation parameters (no timestamps, so two runs of the same seed
are byte-identical).

Business realism (documented in ``docs/DATA_CONTRACT.md`` by the coordinator):

* One business (``BIZ-001``) with up to three branches and volume multipliers
  1.0 / 0.62 / 0.38 (``B-01`` Matriz Centro, ``B-02`` Sucursal Norte,
  ``B-03`` Bodega Sur).
* 20 catalog products: 6 ingredients (feedstock, flakes, pellets, additive) and
  14 sellables; the first 5 sellables have a bill of materials. Ingredients are
  mostly consumed internally but keep a small direct-sale channel to other
  recyclers, so every product/branch series carries some demand. Reduced runs
  (``n_products`` < 20) take prefixes of each catalog and scale the recipe count.
* Demand follows weekly seasonality (Thu-Sat uplift), a December peak (~1.4x),
  a May dip (~0.85x), a March-April recycling peak, and a +3%/quarter trend.
* Roughly 8% of calendar days run a catalog-wide promotion with a 16-25%
  discount (always below the ETL's ``< 0.85 * price`` promo threshold), 2% of
  sale lines are cancelled and pending sales represent ~10% of latent demand.
* Inventory is simulated per branch/product: sales are capped at available
  stock, replenishment arrives after the product lead time, recipe products are
  produced twice a week from their BOM inputs, and occasional waste is recorded.
  ``stock_after`` is therefore never negative and every foreign key is valid.

Reduced runs remap BOM inputs onto the available ingredient prefix so the
dataset is always FK-consistent.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta, timezone, tzinfo
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pandas as pd

from src.data_contracts.schemas import CSV_COLUMNS, SCHEMA_VERSION

LOGGER = logging.getLogger(__name__)

BUSINESS_ID = "BIZ-001"
DEFAULT_START = "2023-01-01"
TIMEZONE_NAME = "America/Mexico_City"
PROMO_RATE = 0.08
PROMO_DISCOUNT_RANGE = (0.16, 0.25)
CANCEL_RATE = 0.02
PENDING_RATE = 0.10
PRODUCTION_SKIP_RATE = 0.04
PRODUCTION_WEEKDAYS = (0, 3)  # Monday and Thursday batches.
REVIEW_WEEKDAY = 0  # Weekly purchase review on Monday.
WASTE_RATE = 0.0015
QUARTER_GROWTH = 1.03
QUARTER_DAYS = 91.25

_WEEKDAY_FACTORS = (0.92, 0.96, 1.00, 1.12, 1.20, 1.15, 0.55)
_MONTH_FACTORS = {
    1: 1.02,
    2: 0.98,
    3: 1.14,
    4: 1.10,
    5: 0.85,
    6: 1.00,
    7: 0.99,
    8: 1.01,
    9: 1.00,
    10: 1.06,
    11: 1.10,
    12: 1.40,
}
_PAYMENT_METHODS = ("efectivo", "transferencia", "tarjeta", "credito")
_PAYMENT_WEIGHTS = (0.45, 0.30, 0.20, 0.05)
_PENDING_STATUSES = ("convertida", "pendiente", "cancelada")
_PENDING_WEIGHTS = (0.55, 0.30, 0.15)
_INGREDIENT_MIN_LOT = {"kg": 50.0, "l": 10.0, "ml": 1000.0, "pza": 10.0}


@dataclass(frozen=True)
class _ProductSpec:
    """Static catalog entry; recipe positions are 0-based ingredient indexes."""

    name: str
    category: str
    unit: str
    price_mxn: float
    cost_mxn: float
    lead_time_days: int
    base_volume: float
    demand_pattern: str
    recipe: tuple[tuple[int, float], ...] = ()


@dataclass(frozen=True)
class _BranchSpec:
    branch_id: str
    name: str
    region: str
    multiplier: float


@dataclass(frozen=True)
class _SimProduct:
    product_id: str
    name: str
    category: str
    unit: str
    price_mxn: float
    cost_mxn: float
    lead_time_days: int
    base_volume: float
    demand_pattern: str
    is_ingredient: bool
    has_recipe: bool
    abc_class: str
    safety_stock: float
    recipe_inputs: tuple[tuple[str, float], ...] = ()


_BRANCHES: tuple[_BranchSpec, ...] = (
    _BranchSpec("B-01", "Matriz Centro", "Centro", 1.00),
    _BranchSpec("B-02", "Sucursal Norte", "Norte", 0.62),
    _BranchSpec("B-03", "Bodega Sur", "Sur", 0.38),
)

# Ingredients are primarily consumed by recipes, but recyclers also sell flakes,
# pellets and surplus feedstock to other transformers, so each ingredient keeps a
# small direct-sale volume. That keeps every product/branch series alive in the
# modeling grain without turning ingredients into mainstream revenue lines.
_INGREDIENT_CATALOG: tuple[_ProductSpec, ...] = (
    _ProductSpec("Botella PET post-consumo compactada", "Materia Prima", "kg", 6.80, 4.50, 8, 1.20, "smooth"),
    _ProductSpec("Hojuela de PET reciclado transparente", "Materia Prima", "kg", 12.40, 8.20, 6, 0.90, "smooth"),
    _ProductSpec("Hojuela de PEAD reciclado", "Materia Prima", "kg", 14.60, 10.10, 5, 0.70, "intermittent"),
    _ProductSpec("Pellet de polipropileno reciclado", "Materia Prima", "kg", 18.90, 13.20, 12, 0.50, "intermittent"),
    _ProductSpec("Film de polietileno post-industrial", "Materia Prima", "kg", 9.80, 6.10, 9, 0.40, "lumpy"),
    _ProductSpec("Aditivo estabilizante UV", "Insumos", "l", 82.00, 58.00, 14, 0.25, "lumpy"),
)

_SELLABLE_CATALOG: tuple[_ProductSpec, ...] = (
    _ProductSpec(
        "Bolsa de basura PEAD 60 x 90 cm",
        "Producto Terminado",
        "pza",
        3.40,
        2.30,
        5,
        14.0,
        "smooth",
        ((2, 0.03), (5, 0.01)),
    ),
    _ProductSpec(
        "Bolsa de basura PEAD 90 x 120 cm",
        "Producto Terminado",
        "pza",
        4.95,
        3.35,
        5,
        9.0,
        "smooth",
        ((2, 0.05), (5, 0.01)),
    ),
    _ProductSpec(
        "Tarima de plástico reciclado 1.2 x 1.0 m",
        "Producto Terminado",
        "pza",
        248.00,
        176.00,
        10,
        1.6,
        "smooth",
        ((3, 7.50), (4, 1.20)),
    ),
    _ProductSpec(
        "Caja apilable de plástico reciclado 60 L",
        "Producto Terminado",
        "pza",
        172.00,
        121.00,
        8,
        2.2,
        "intermittent",
        ((3, 2.10), (4, 0.50)),
    ),
    _ProductSpec(
        "Contenedor industrial de plástico reciclado 240 L",
        "Producto Terminado",
        "pza",
        895.00,
        642.00,
        15,
        0.35,
        "lumpy",
        ((2, 11.50), (4, 0.90)),
    ),
    _ProductSpec(
        "Tubería PVC sanitaria 1/2 pulgada",
        "Producto Terminado",
        "pza",
        42.50,
        29.20,
        4,
        8.5,
        "smooth",
    ),
    _ProductSpec(
        "Tubería PVC sanitaria 2 pulgadas",
        "Producto Terminado",
        "pza",
        118.00,
        83.00,
        4,
        3.0,
        "intermittent",
    ),
    _ProductSpec("Perfil de poliestireno 2 m", "Producto Terminado", "pza", 36.00, 24.80, 6, 1.2, "lumpy"),
    _ProductSpec(
        "Lámina de PEAD reciclado 1.2 x 2.4 m",
        "Producto Terminado",
        "pza",
        410.00,
        288.00,
        9,
        0.5,
        "lumpy",
    ),
    _ProductSpec(
        "Rollo de fleje PET reciclado 1 kg",
        "Producto Terminado",
        "pza",
        28.00,
        19.40,
        6,
        2.6,
        "intermittent",
    ),
    _ProductSpec(
        "Hojuela de PET grado alimenticio",
        "Materia Prima",
        "kg",
        16.80,
        11.90,
        8,
        6.0,
        "smooth",
    ),
    _ProductSpec(
        "Bolsa reutilizable de supermercado",
        "Consumibles",
        "pza",
        6.90,
        4.65,
        5,
        1.8,
        "intermittent",
    ),
    _ProductSpec(
        "Cubeta de plástico reciclado 20 L",
        "Producto Terminado",
        "pza",
        58.00,
        40.00,
        5,
        1.4,
        "intermittent",
    ),
    _ProductSpec(
        "Bolsa comercial biodegradable 40 x 50 cm",
        "Consumibles",
        "pza",
        2.15,
        1.48,
        6,
        3.2,
        "intermittent",
    ),
)


def _local_timezone() -> tzinfo:
    """America/Mexico_City with a fixed UTC-6 fallback (no DST since 2022)."""
    try:
        return ZoneInfo(TIMEZONE_NAME)
    except ZoneInfoNotFoundError:  # pragma: no cover - exercised only without tzdata
        return timezone(timedelta(hours=-6))


def _iso(day: date, hour: int, minute: int, tz: tzinfo) -> str:
    """UTC-6 ISO-8601 timestamp, e.g. ``2024-12-24T18:30:00-06:00``."""
    return datetime(day.year, day.month, day.day, hour, minute, tzinfo=tz).isoformat()


def _quantize(unit: str, value: float) -> float:
    if unit == "pza":
        return float(int(round(value)))
    return round(value, 2)


def _floor_units(unit: str, value: float) -> float:
    if unit == "pza":
        return float(math.floor(value))
    return math.floor(value * 100.0) / 100.0


def _ceil_units(unit: str, value: float) -> float:
    if unit == "pza":
        return float(math.ceil(value))
    return math.ceil(value * 100.0) / 100.0


def _validate_arguments(days: int, n_products: int, n_branches: int) -> None:
    if days < 1:
        raise ValueError("days must be >= 1")
    if not 2 <= n_products <= len(_INGREDIENT_CATALOG) + len(_SELLABLE_CATALOG):
        raise ValueError(
            f"n_products must be between 2 and {len(_INGREDIENT_CATALOG) + len(_SELLABLE_CATALOG)}"
        )
    if not 1 <= n_branches <= len(_BRANCHES):
        raise ValueError(f"n_branches must be between 1 and {len(_BRANCHES)}")


def _allocate_counts(n_products: int) -> tuple[int, int, int]:
    """Split the catalog into ingredients/sellables/recipes for a given size."""
    n_ingredients = max(1, min(n_products - 1, round(n_products * 0.3)))
    n_sellables = n_products - n_ingredients
    if n_ingredients < 2:
        n_recipes = 0
    else:
        n_recipes = min(5, n_sellables, max(1, round(n_sellables * 5 / 14)))
    return n_ingredients, n_sellables, n_recipes


def _abc_classes(sellables: tuple[_ProductSpec, ...]) -> list[str]:
    """ABC by planned unit volume: roughly 20% A, 30% B, 50% C."""
    count = len(sellables)
    order = sorted(range(count), key=lambda index: (-sellables[index].base_volume, index))
    n_a = min(count, max(1, round(count * 0.2)))
    n_b = min(count - n_a, max(0, round(count * 0.3)))
    classes = ["C"] * count
    for index in order[:n_a]:
        classes[index] = "A"
    for index in order[n_a : n_a + n_b]:
        classes[index] = "B"
    return classes


def _resolve_recipe(
    template: tuple[tuple[int, float], ...], n_ingredients: int
) -> tuple[tuple[int, float], ...]:
    """Map BOM input positions onto the ingredient prefix, keeping inputs distinct."""
    resolved: list[tuple[int, float]] = []
    used: set[int] = set()
    for position, quantity in template:
        index = position % n_ingredients
        while index in used:
            index = (index + 1) % n_ingredients
        used.add(index)
        resolved.append((index, quantity))
    return tuple(resolved)


def _build_products(n_products: int) -> tuple[list[_SimProduct], list[dict[str, object]]]:
    n_ingredients, n_sellables, n_recipes = _allocate_counts(n_products)
    ingredient_specs = _INGREDIENT_CATALOG[:n_ingredients]
    sellable_specs = _SELLABLE_CATALOG[:n_sellables]
    abc_classes = _abc_classes(sellable_specs)

    products: list[_SimProduct] = []
    for spec in ingredient_specs:
        products.append(
            _SimProduct(
                product_id=f"P-{len(products) + 1:03d}",
                name=spec.name,
                category=spec.category,
                unit=spec.unit,
                price_mxn=spec.price_mxn,
                cost_mxn=spec.cost_mxn,
                lead_time_days=spec.lead_time_days,
                base_volume=spec.base_volume,
                demand_pattern=spec.demand_pattern,
                is_ingredient=True,
                has_recipe=False,
                abc_class="C",
                safety_stock=0.0,
            )
        )

    sellable_offset = len(products)
    for index, spec in enumerate(sellable_specs):
        safety = round(0.5 * spec.base_volume * math.sqrt(spec.lead_time_days), 2)
        products.append(
            _SimProduct(
                product_id=f"P-{len(products) + 1:03d}",
                name=spec.name,
                category=spec.category,
                unit=spec.unit,
                price_mxn=spec.price_mxn,
                cost_mxn=spec.cost_mxn,
                lead_time_days=spec.lead_time_days,
                base_volume=spec.base_volume,
                demand_pattern=spec.demand_pattern,
                is_ingredient=False,
                has_recipe=False,
                abc_class=abc_classes[index],
                safety_stock=safety,
            )
        )

    recipes: list[dict[str, object]] = []
    declared_inputs: dict[str, tuple[tuple[str, float], ...]] = {}
    if n_ingredients >= 2:
        for index in range(min(n_recipes, n_sellables)):
            output = products[sellable_offset + index]
            mapped = _resolve_recipe(sellable_specs[index].recipe, n_ingredients)
            inputs = tuple((products[position].product_id, quantity) for position, quantity in mapped)
            declared_inputs[output.product_id] = inputs
            recipe_id = f"R-{index + 1:03d}"
            for input_product_id, quantity_per_unit in inputs:
                recipes.append(
                    {
                        "recipe_id": recipe_id,
                        "business_id": BUSINESS_ID,
                        "output_product_id": output.product_id,
                        "input_product_id": input_product_id,
                        "quantity_per_unit": quantity_per_unit,
                    }
                )

    sellable_by_id = {product.product_id: product for product in products}
    ingredient_consumption: dict[str, float] = {product.product_id: 0.0 for product in products}
    for output_id, inputs in declared_inputs.items():
        output = sellable_by_id[output_id]
        for input_id, quantity_per_unit in inputs:
            ingredient_consumption[input_id] += output.base_volume * quantity_per_unit

    products = [
        replace(
            product,
            has_recipe=product.product_id in declared_inputs,
            recipe_inputs=declared_inputs.get(product.product_id, ()),
            safety_stock=(
                round(
                    (ingredient_consumption[product.product_id] + product.base_volume)
                    * product.lead_time_days
                    * 0.5,
                    2,
                )
                if product.is_ingredient
                else product.safety_stock
            ),
        )
        for product in products
    ]
    return products, recipes


def _poisson(rng: random.Random, lam: float) -> int:
    """Knuth's algorithm for small means; normal approximation for large ones."""
    if lam <= 0:
        return 0
    if lam >= 30:
        return max(0, int(round(rng.gauss(lam, math.sqrt(lam)))))
    limit = math.exp(-lam)
    k = 0
    probability = 1.0
    while True:
        k += 1
        probability *= rng.random()
        if probability <= limit:
            return k - 1


def _draw_demand(rng: random.Random, pattern: str, mean: float, unit: str) -> float:
    """Sample daily demand preserving the requested mean and intermittency."""
    if mean <= 0:
        return 0.0
    if pattern == "smooth":
        quantity = max(0.0, rng.gauss(mean, math.sqrt(mean) * 1.1))
    elif pattern == "intermittent":
        probability = min(0.70, max(0.20, 0.20 + 0.08 * mean))
        if rng.random() >= probability:
            return 0.0
        quantity = float(_poisson(rng, mean / probability))
    else:  # lumpy
        probability = min(0.40, max(0.08, 0.08 + 0.08 * mean))
        if rng.random() >= probability:
            return 0.0
        sigma = 0.8
        log_mean = math.log(max(mean / probability, 0.05)) - sigma * sigma / 2.0
        quantity = rng.lognormvariate(log_mean, sigma)
        quantity = min(quantity, (mean / probability) * 8.0)
    return _quantize(unit, quantity)


class _Simulator:
    """Event-driven inventory simulation; one instance per generated dataset."""

    def __init__(self, products: list[_SimProduct], branches: tuple[_BranchSpec, ...], days: int,
                 start_date: date, seed: int) -> None:
        self.products = products
        self.branches = branches
        self.days = days
        self.start_date = start_date
        self.tz = _local_timezone()
        self.rng = random.Random(seed)
        self.ingredients = [product for product in products if product.is_ingredient]
        self.product_by_id = {product.product_id: product for product in products}
        self.produced = [product for product in products if product.has_recipe]
        self.purchased_sellables = [
            product for product in products if not product.has_recipe and not product.is_ingredient
        ]
        self.purchased_ids = {product.product_id for product in self.purchased_sellables}
        self.waste_products = list(products)
        # Ingredient outflow = BOM consumption + small direct sales to other recyclers.
        self.ingredient_outflow_rates: dict[tuple[str, str], float] = {}
        for branch in branches:
            for ingredient in self.ingredients:
                bom_consumption = sum(
                    output.base_volume * branch.multiplier * quantity_per_unit
                    for output in self.produced
                    for input_id, quantity_per_unit in output.recipe_inputs
                    if input_id == ingredient.product_id
                )
                direct_sales = ingredient.base_volume * branch.multiplier
                self.ingredient_outflow_rates[(branch.branch_id, ingredient.product_id)] = (
                    bom_consumption + direct_sales
                )
        self.stock: dict[tuple[str, str], float] = {}
        self.on_order: dict[tuple[str, str], float] = {}
        self.arrivals: dict[tuple[str, str, int], float] = {}
        self.events: list[dict[str, object]] = []
        self.movements: list[dict[str, object]] = []
        self.pending: list[dict[str, object]] = []
        self.event_ids: dict[tuple[str, str, date], list[str]] = {}
        self.daily_demand: dict[tuple[str, str, date], float] = {}
        self._event_sequence = 0
        self._movement_sequence = 0
        self._pending_sequence = 0

    # ------------------------------------------------------------------ helpers
    def _trend(self, day_index: int) -> float:
        return QUARTER_GROWTH ** (day_index / QUARTER_DAYS)

    @staticmethod
    def _seasonality(day: date) -> float:
        return _WEEKDAY_FACTORS[day.weekday()] * _MONTH_FACTORS[day.month]

    def _safety(self, product: _SimProduct, branch: _BranchSpec) -> float:
        return max(0.0, product.safety_stock * branch.multiplier)

    def _demand_rate(self, product: _SimProduct, branch: _BranchSpec, day_index: int) -> float:
        return product.base_volume * branch.multiplier * self._trend(day_index)

    def _next_event_id(self) -> str:
        self._event_sequence += 1
        return f"E-{self._event_sequence:08d}"

    def _next_movement_id(self) -> str:
        self._movement_sequence += 1
        return f"M-{self._movement_sequence:08d}"

    def _next_pending_id(self) -> str:
        self._pending_sequence += 1
        return f"PS-{self._pending_sequence:08d}"

    def _movement(
        self,
        branch: _BranchSpec,
        product: _SimProduct,
        day: date,
        hour: int,
        direction: str,
        reason: str,
        quantity: float,
        stock_after: float,
        note: str = "",
    ) -> None:
        self.movements.append(
            {
                "movement_id": self._next_movement_id(),
                "business_id": BUSINESS_ID,
                "branch_id": branch.branch_id,
                "product_id": product.product_id,
                "movement_ts": _iso(day, hour, 0, self.tz),
                "direction": direction,
                "reason": reason,
                "quantity": quantity,
                "unit_cost_mxn": product.cost_mxn,
                "stock_after": max(0.0, round(stock_after, 2)),
                "note": note,
            }
        )

    # ------------------------------------------------------------- inventory ops
    def _open_inventory(self) -> None:
        for branch in self.branches:
            for product in self.products:
                if product.is_ingredient:
                    rate = self.ingredient_outflow_rates[(branch.branch_id, product.product_id)]
                    minimum = _INGREDIENT_MIN_LOT.get(product.unit, 10.0)
                    quantity = max(rate * 35.0, minimum)
                else:
                    rate = product.base_volume * branch.multiplier
                    safety = self._safety(product, branch)
                    quantity = rate * (product.lead_time_days + 7.0) + safety
                quantity = max(_ceil_units(product.unit, quantity), 1.0)
                key = (branch.branch_id, product.product_id)
                self.stock[key] = quantity
                self._movement(
                    branch, product, self.start_date, 7, "entrada", "ajuste_manual",
                    quantity, quantity, "inventario inicial",
                )

    def _receive_arrivals(self, branch: _BranchSpec, day_index: int, day: date) -> None:
        for product in self.products:
            key = (branch.branch_id, product.product_id)
            quantity = self.arrivals.pop((branch.branch_id, product.product_id, day_index), 0.0)
            if quantity <= 0:
                continue
            self.on_order[key] = max(0.0, self.on_order.get(key, 0.0) - quantity)
            new_stock = round(self.stock.get(key, 0.0) + quantity, 2)
            self.stock[key] = _quantize(product.unit, new_stock)
            self._movement(
                branch, product, day, 8, "entrada", "compra", quantity, self.stock[key],
                "reabasto de inventario",
            )
            self.events.append(
                {
                    "event_id": self._next_event_id(),
                    "business_id": BUSINESS_ID,
                    "branch_id": branch.branch_id,
                    "event_type": "compra",
                    "event_ts": _iso(day, 8, 0, self.tz),
                    "amount_mxn": round(quantity * product.cost_mxn, 2),
                    "payment_method": "",
                    "customer_id": "",
                    "product_id": product.product_id,
                    "product_name": product.name,
                    "quantity": quantity,
                    "unit_price_mxn": product.cost_mxn,
                    "unit_cost_mxn": product.cost_mxn,
                    "is_cancelled": 0,
                    "note": "",
                }
            )

    def _simulate_production(self, branch: _BranchSpec, day_index: int, day: date) -> None:
        if day.weekday() not in PRODUCTION_WEEKDAYS:
            return
        for product in self.produced:
            key = (branch.branch_id, product.product_id)
            stock = self.stock.get(key, 0.0)
            rate = self._demand_rate(product, branch, day_index)
            reorder_point = rate * product.lead_time_days + self._safety(product, branch)
            if stock >= reorder_point or self.rng.random() < PRODUCTION_SKIP_RATE:
                continue
            target = reorder_point + rate * 4.0
            desired = target - stock
            max_units = min(
                self.stock.get((branch.branch_id, input_id), 0.0) / quantity_per_unit
                for input_id, quantity_per_unit in product.recipe_inputs
            )
            produced = _floor_units(product.unit, min(desired, max_units))
            if produced <= 0:
                continue
            for input_id, quantity_per_unit in product.recipe_inputs:
                ingredient = self.product_by_id[input_id]
                available = self.stock.get((branch.branch_id, input_id), 0.0)
                consumed = min(round(produced * quantity_per_unit, 2), available)
                if consumed <= 0:
                    continue
                new_stock = round(available - consumed, 2)
                self.stock[(branch.branch_id, input_id)] = max(0.0, new_stock)
                self._movement(
                    branch, ingredient, day, 9, "salida", "produccion", consumed,
                    self.stock[(branch.branch_id, input_id)], "consumo de BOM",
                )
            self.stock[key] = _quantize(product.unit, stock + produced)
            self._movement(
                branch, product, day, 9, "entrada", "produccion", produced,
                self.stock[key], "lote de produccion",
            )
            self.events.append(
                {
                    "event_id": self._next_event_id(),
                    "business_id": BUSINESS_ID,
                    "branch_id": branch.branch_id,
                    "event_type": "produccion",
                    "event_ts": _iso(day, 9, 0, self.tz),
                    "amount_mxn": round(produced * product.cost_mxn, 2),
                    "payment_method": "",
                    "customer_id": "",
                    "product_id": product.product_id,
                    "product_name": product.name,
                    "quantity": produced,
                    "unit_price_mxn": product.cost_mxn,
                    "unit_cost_mxn": product.cost_mxn,
                    "is_cancelled": 0,
                    "note": "lote de produccion",
                }
            )

    def _simulate_sales(
        self, branch: _BranchSpec, day_index: int, day: date, promo_day: bool
    ) -> list[dict]:
        lines: list[dict] = []
        for product in self.products:
            key = (branch.branch_id, product.product_id)
            mean = self._demand_rate(product, branch, day_index) * self._seasonality(day)
            demand = _draw_demand(self.rng, product.demand_pattern, mean, product.unit)
            if demand <= 0:
                continue
            stock = self.stock.get(key, 0.0)
            available = max(0.0, stock)
            promo = promo_day
            unit_price = product.price_mxn
            if promo:
                discount = self.rng.uniform(*PROMO_DISCOUNT_RANGE)
                # Clamp to [0.75, 0.85) of the master price so the ETL promo rule
                # (unit_price < 0.85 * price_mxn) is always true and the discount
                # stays within the documented 10-25% band after rounding.
                lower_bound = math.ceil(product.price_mxn * 0.75 * 100.0) / 100.0
                upper_bound = (
                    math.floor(product.price_mxn * 0.85 * 100.0 - 1e-6) / 100.0
                )
                unit_price = min(
                    max(round(product.price_mxn * (1.0 - discount), 2), lower_bound),
                    upper_bound,
                )
            cancelled = self.rng.random() < CANCEL_RATE
            sold = _quantize(product.unit, min(demand, available))
            if sold <= 0:
                continue
            if not cancelled:
                self.stock[key] = _quantize(product.unit, max(0.0, stock - sold))
                self._movement(
                    branch, product, day, 13, "salida", "venta_manual", sold,
                    self.stock[key], "",
                )
            payment_method = self.rng.choices(_PAYMENT_METHODS, weights=_PAYMENT_WEIGHTS)[0]
            customer_id = (
                f"C-{self.rng.randint(1, 240):04d}" if payment_method == "credito" else ""
            )
            self.daily_demand[(branch.branch_id, product.product_id, day)] = demand
            lines.append(
                {
                    "product_id": product.product_id,
                    "product_name": product.name,
                    "quantity": sold,
                    "unit_price_mxn": unit_price,
                    "unit_cost_mxn": product.cost_mxn,
                    "amount_mxn": round(sold * unit_price, 2),
                    "is_cancelled": int(cancelled),
                    "payment_method": payment_method,
                    "customer_id": customer_id,
                    "note": "cancelada por el cliente" if cancelled else "",
                }
            )
        return lines

    def _emit_sale_events(self, branch: _BranchSpec, day: date, lines: list[dict]) -> None:
        index = 0
        while index < len(lines):
            ticket_size = self.rng.choice((1, 1, 1, 2, 2, 3))
            ticket = lines[index : index + ticket_size]
            event_id = self._next_event_id()
            timestamp = _iso(day, self.rng.randint(9, 18), self.rng.randint(0, 59), self.tz)
            for line in ticket:
                self.events.append(
                    {
                        "event_id": event_id,
                        "business_id": BUSINESS_ID,
                        "branch_id": branch.branch_id,
                        "event_type": "venta",
                        "event_ts": timestamp,
                        **line,
                    }
                )
                self.event_ids.setdefault(
                    (branch.branch_id, str(line["product_id"]), day), []
                ).append(event_id)
            index += ticket_size

    def _simulate_waste(self, branch: _BranchSpec, day_index: int, day: date) -> None:
        for product in self.waste_products:
            key = (branch.branch_id, product.product_id)
            stock = self.stock.get(key, 0.0)
            if stock <= 0 or self.rng.random() >= WASTE_RATE:
                continue
            quantity = _quantize(product.unit, stock * self.rng.uniform(0.003, 0.02))
            quantity = min(quantity, stock)
            if quantity <= 0:
                continue
            self.stock[key] = _quantize(product.unit, max(0.0, stock - quantity))
            self._movement(
                branch, product, day, 17, "salida", "merma", quantity,
                self.stock[key], "merma de almacen",
            )
            self.events.append(
                {
                    "event_id": self._next_event_id(),
                    "business_id": BUSINESS_ID,
                    "branch_id": branch.branch_id,
                    "event_type": "merma",
                    "event_ts": _iso(day, 17, 0, self.tz),
                    "amount_mxn": round(quantity * product.cost_mxn, 2),
                    "payment_method": "",
                    "customer_id": "",
                    "product_id": product.product_id,
                    "product_name": product.name,
                    "quantity": quantity,
                    "unit_price_mxn": product.cost_mxn,
                    "unit_cost_mxn": product.cost_mxn,
                    "is_cancelled": 0,
                    "note": "merma de almacen",
                }
            )

    def _review_orders(self, branch: _BranchSpec, day_index: int, day: date) -> None:
        if day.weekday() != REVIEW_WEEKDAY:
            return
        for product in self.products:
            key = (branch.branch_id, product.product_id)
            safety = self._safety(product, branch)
            if product.is_ingredient:
                rate = (
                    self.ingredient_outflow_rates[(branch.branch_id, product.product_id)]
                    * self._trend(day_index)
                )
                if rate <= 0:
                    continue
                reorder_point = rate * (product.lead_time_days + 10.0) + safety
                minimum = _INGREDIENT_MIN_LOT.get(product.unit, 10.0)
                target = rate * (product.lead_time_days + 45.0) + safety + minimum
            elif product.product_id in self.purchased_ids:
                rate = self._demand_rate(product, branch, day_index)
                # Weekly review: cover lead time plus the review period, then
                # order one more review period of demand (classic (s, S) policy).
                reorder_point = rate * (product.lead_time_days + 7.0) + safety
                target = reorder_point + rate * 7.0
            else:
                continue
            stock = self.stock.get(key, 0.0)
            on_order = self.on_order.get(key, 0.0)
            if stock + on_order >= reorder_point:
                continue
            quantity = _ceil_units(product.unit, target - stock - on_order)
            if quantity <= 0:
                continue
            arrival_index = day_index + product.lead_time_days
            if arrival_index >= self.days:
                continue
            self.arrivals[(branch.branch_id, product.product_id, arrival_index)] = (
                self.arrivals.get((branch.branch_id, product.product_id, arrival_index), 0.0)
                + quantity
            )
            self.on_order[key] = on_order + quantity

    def _generate_pending_sales(self) -> None:
        for (branch_id, product_id, created_day), demand in sorted(
            self.daily_demand.items(), key=lambda item: (item[0][0], item[0][1], item[0][2])
        ):
            if self.rng.random() >= PENDING_RATE:
                continue
            product = self.product_by_id[product_id]
            quantity = _quantize(product.unit, demand)
            if quantity <= 0:
                continue
            status = self.rng.choices(_PENDING_STATUSES, weights=_PENDING_WEIGHTS)[0]
            converted_event_id = ""
            if status == "convertida":
                candidates = self.event_ids.get((branch_id, product_id, created_day), [])
                if not candidates:
                    status = "pendiente"
                else:
                    converted_event_id = candidates[0]
            promised_day = created_day + timedelta(days=self.rng.randint(2, 10))
            self.pending.append(
                {
                    "pending_sale_id": self._next_pending_id(),
                    "business_id": BUSINESS_ID,
                    "branch_id": branch_id,
                    "product_id": product_id,
                    "created_ts": _iso(created_day, 11, 0, self.tz),
                    "promised_ts": _iso(promised_day, 12, 0, self.tz),
                    "quantity": quantity,
                    "unit_price_mxn": product.price_mxn,
                    "status": status,
                    "converted_event_id": converted_event_id,
                }
            )

    # ---------------------------------------------------------------- simulation
    def run(self) -> None:
        self._open_inventory()
        for day_index in range(self.days):
            day = self.start_date + timedelta(days=day_index)
            # Promotions are calendar campaigns: about 8% of days apply to the
            # whole catalog, so the ETL promo flag is consistent across series.
            promo_day = self.rng.random() < PROMO_RATE
            for branch in self.branches:
                self._receive_arrivals(branch, day_index, day)
                self._simulate_production(branch, day_index, day)
                lines = self._simulate_sales(branch, day_index, day, promo_day)
                self._emit_sale_events(branch, day, lines)
                self._simulate_waste(branch, day_index, day)
                self._review_orders(branch, day_index, day)
        self._generate_pending_sales()
        self._add_monthly_expenses()

    def _add_monthly_expenses(self) -> None:
        expenses = (
            ("renta", 22000.0),
            ("energia", 8500.0),
            ("agua", 1200.0),
            ("nomina", 48000.0),
            ("mantenimiento", 3500.0),
            ("fletes", 6200.0),
        )
        cursor = self.start_date
        while cursor <= self.start_date + timedelta(days=self.days - 1):
            if cursor.day == 1:
                for branch in self.branches:
                    for label, base_amount in expenses:
                        self.events.append(
                            {
                                "event_id": self._next_event_id(),
                                "business_id": BUSINESS_ID,
                                "branch_id": branch.branch_id,
                                "event_type": "gasto",
                                "event_ts": _iso(cursor, 10, 0, self.tz),
                                "amount_mxn": round(base_amount * branch.multiplier, 2),
                                "payment_method": "",
                                "customer_id": "",
                                "product_id": "",
                                "product_name": "",
                                "quantity": "",
                                "unit_price_mxn": "",
                                "unit_cost_mxn": "",
                                "is_cancelled": 0,
                                "note": f"gasto mensual: {label}",
                            }
                        )
            next_month = cursor.month % 12 + 1
            next_year = cursor.year + (cursor.month // 12)
            cursor = date(next_year, next_month, 1)


def _write_csv(rows: list[dict[str, object]], path: Path, schema_name: str) -> Path:
    """Write a contract CSV with a stable column order and 2-decimal floats."""
    frame = pd.DataFrame(rows, columns=list(CSV_COLUMNS[schema_name]))
    frame.to_csv(
        path,
        index=False,
        encoding="utf-8",
        lineterminator="\n",
        float_format="%.2f",
        na_rep="",
    )
    return path


def generate_dataset(
    out_dir: Path | str,
    *,
    days: int = 1096,
    n_products: int = 20,
    n_branches: int = 3,
    seed: int = 42,
    start: str = DEFAULT_START,
) -> dict[str, Path]:
    """Generate the synthetic raw dataset and return the written paths."""
    _validate_arguments(days, n_products, n_branches)
    start_date = date.fromisoformat(start)
    end_date = start_date + timedelta(days=days - 1)
    products, recipes = _build_products(n_products)
    branches = _BRANCHES[:n_branches]

    simulator = _Simulator(products, branches, days, start_date, seed)
    simulator.run()

    raw_dir = Path(out_dir) / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    product_rows = [
        {
            "product_id": product.product_id,
            "name": product.name,
            "category": product.category,
            "unit": product.unit,
            "price_mxn": product.price_mxn,
            "cost_mxn": product.cost_mxn,
            "is_ingredient": int(product.is_ingredient),
            "has_recipe": int(product.has_recipe),
            "safety_stock": product.safety_stock,
            "lead_time_days": product.lead_time_days,
            "demand_pattern": product.demand_pattern,
            "abc_class": product.abc_class,
        }
        for product in products
    ]
    branch_rows = [
        {
            "branch_id": branch.branch_id,
            "business_id": BUSINESS_ID,
            "name": branch.name,
            "region": branch.region,
        }
        for branch in branches
    ]

    events = sorted(simulator.events, key=lambda row: (row["event_ts"], row["event_id"]))
    for row in events:
        for column in ("amount_mxn", "quantity", "unit_price_mxn", "unit_cost_mxn"):
            value = row[column]
            row[column] = "" if value is None or value == "" else f"{float(value):.2f}"
    movements = sorted(
        simulator.movements, key=lambda row: (row["movement_ts"], row["movement_id"])
    )
    pending = sorted(simulator.pending, key=lambda row: (row["created_ts"], row["pending_sale_id"]))

    for movement in movements:
        if float(movement["stock_after"]) < 0:  # pragma: no cover - invariant guard
            raise RuntimeError(f"negative stock_after generated: {movement}")

    paths = {
        "raw_dir": raw_dir,
        "products": _write_csv(product_rows, raw_dir / "products.csv", "products"),
        "branches": _write_csv(branch_rows, raw_dir / "branches.csv", "branches"),
        "recipes": _write_csv(recipes, raw_dir / "recipes.csv", "recipes"),
        "economic_events": _write_csv(events, raw_dir / "economic_events.csv", "economic_events"),
        "stock_movements": _write_csv(
            movements, raw_dir / "stock_movements.csv", "stock_movements"
        ),
        "pending_sales": _write_csv(pending, raw_dir / "pending_sales.csv", "pending_sales"),
    }

    seed_payload = {
        "generator": "synthetic_generator",
        "schema_version": SCHEMA_VERSION,
        "seed": seed,
        "business_id": BUSINESS_ID,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "days": days,
        "n_products": n_products,
        "n_branches": n_branches,
        "currency": "MXN",
        "timezone": TIMEZONE_NAME,
        "branch_multipliers": {
            branch.branch_id: branch.multiplier for branch in branches
        },
        "row_counts": {
            "products": len(product_rows),
            "branches": len(branch_rows),
            "recipes": len(recipes),
            "economic_events": len(events),
            "stock_movements": len(movements),
            "pending_sales": len(pending),
        },
    }
    seed_path = raw_dir / "seed.json"
    seed_path.write_text(
        json.dumps(seed_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    paths["seed"] = seed_path
    LOGGER.info(
        "Generated %s rows under %s", {name: row_count for name, row_count in seed_payload["row_counts"].items()}, raw_dir
    )
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the deterministic Asviin synthetic dataset.")
    parser.add_argument("--out", required=True, help="Output directory; raw CSVs are written to OUT/raw.")
    parser.add_argument("--days", type=int, default=1096)
    parser.add_argument("--products", type=int, default=20)
    parser.add_argument("--branches", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start", default=DEFAULT_START)
    args = parser.parse_args(argv)
    paths = generate_dataset(
        args.out,
        days=args.days,
        n_products=args.products,
        n_branches=args.branches,
        seed=args.seed,
        start=args.start,
    )
    print(
        f"Generated {args.products} products / {args.branches} branches / {args.days} days "
        f"under {paths['raw_dir']}"
    )
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    raise SystemExit(main())
