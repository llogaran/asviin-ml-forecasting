"""Canonical data contracts mirroring the Asviin ERP v2 schema.

Asviin keeps its operational data in Firestore (NoSQL). These Pydantic models are
the *canonical projection* of the fields this project consumes; CSV files under
``data/`` are the transport format. Any change here must be reflected in
``docs/DATA_CONTRACT.md`` and bump ``SCHEMA_VERSION``.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field

SCHEMA_VERSION = "1.0.0"

UNITS = ("pza", "kg", "l", "ml")
DEMAND_PATTERNS = ("smooth", "intermittent", "lumpy")
ABC_CLASSES = ("A", "B", "C")
PAYMENT_METHODS = ("efectivo", "transferencia", "tarjeta", "credito")


class EventType(StrEnum):
    """Economic event types used by Asviin v2 (``economicEvents.type``)."""

    VENTA = "venta"
    COMPRA = "compra"
    GASTO = "gasto"
    MERMA = "merma"
    PRODUCCION = "produccion"
    TRANSFERENCIA = "transferencia"


class MovementDirection(StrEnum):
    ENTRADA = "entrada"
    SALIDA = "salida"


class MovementReason(StrEnum):
    """Allowed ``reason`` values for ``products/{id}/movimientos``."""

    VENTA_MANUAL = "venta_manual"
    COMPRA = "compra"
    AJUSTE_MANUAL = "ajuste_manual"
    MERMA = "merma"
    PRODUCCION = "produccion"
    TRANSFERENCIA_ENTRANTE = "transferencia_entrante"
    TRANSFERENCIA_SALIENTE = "transferencia_saliente"


class PendingSaleStatus(StrEnum):
    PENDIENTE = "pendiente"
    CONVERTIDA = "convertida"
    CANCELADA = "cancelada"


Unit = Literal["pza", "kg", "l", "ml"]
DemandPattern = Literal["smooth", "intermittent", "lumpy"]
AbcClass = Literal["A", "B", "C"]
PaymentMethod = Literal["efectivo", "transferencia", "tarjeta", "credito"]


class Product(BaseModel):
    """Product master (``businesses/{id}/products/{id}`` + ``data`` fields)."""

    product_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    category: str = Field(min_length=1)
    unit: Unit
    price_mxn: float = Field(gt=0)
    cost_mxn: float = Field(ge=0)
    is_ingredient: bool = False
    has_recipe: bool = False
    safety_stock: float = Field(ge=0)
    lead_time_days: int = Field(ge=0)
    demand_pattern: DemandPattern
    abc_class: AbcClass


class Branch(BaseModel):
    """Sucursal (``businesses/{id}/branches/{id}``)."""

    branch_id: str = Field(min_length=1)
    business_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    region: str = Field(min_length=1)


class Recipe(BaseModel):
    """BOM line: one input product consumed per unit of an output product."""

    recipe_id: str = Field(min_length=1)
    business_id: str = Field(min_length=1)
    output_product_id: str = Field(min_length=1)
    input_product_id: str = Field(min_length=1)
    quantity_per_unit: float = Field(gt=0)


class EconomicEventLine(BaseModel):
    """One line of an economic event (Asviin stores items inside metadata).

    ``venta`` rows are item-level: ``event_id`` is repeated for every ticket line
    and ``amount_mxn`` is the line subtotal. Non-sale rows (``gasto``) may carry
    no product.
    """

    event_id: str = Field(min_length=1)
    business_id: str = Field(min_length=1)
    branch_id: str = Field(min_length=1)
    event_type: EventType
    event_ts: datetime
    amount_mxn: float
    payment_method: PaymentMethod | None = None
    customer_id: str | None = None
    product_id: str | None = None
    product_name: str | None = None
    quantity: float | None = Field(default=None, ge=0)
    unit_price_mxn: float | None = Field(default=None, ge=0)
    unit_cost_mxn: float | None = Field(default=None, ge=0)
    is_cancelled: bool = False
    note: str | None = None


class StockMovement(BaseModel):
    """Inventory movement (``products/{id}/movimientos``)."""

    movement_id: str = Field(min_length=1)
    business_id: str = Field(min_length=1)
    branch_id: str = Field(min_length=1)
    product_id: str = Field(min_length=1)
    movement_ts: datetime
    direction: MovementDirection
    reason: MovementReason
    quantity: float = Field(ge=0)
    unit_cost_mxn: float = Field(ge=0)
    stock_after: float = Field(ge=0)
    note: str | None = None


class PendingSale(BaseModel):
    """Pedido pendiente (``businesses/{id}/pendingSales/{id}``)."""

    pending_sale_id: str = Field(min_length=1)
    business_id: str = Field(min_length=1)
    branch_id: str = Field(min_length=1)
    product_id: str = Field(min_length=1)
    created_ts: datetime
    promised_ts: datetime | None = None
    quantity: float = Field(gt=0)
    unit_price_mxn: float = Field(ge=0)
    status: PendingSaleStatus
    converted_event_id: str | None = None


class DailyDemand(BaseModel):
    """Modeling table: one row per business/branch/product/day."""

    date: date
    business_id: str = Field(min_length=1)
    branch_id: str = Field(min_length=1)
    product_id: str = Field(min_length=1)
    category: str = Field(min_length=1)
    unit: Unit
    units_sold: float = Field(ge=0)
    revenue_mxn: float = Field(ge=0)
    cost_mxn: float = Field(ge=0)
    cancelled_units: float = Field(ge=0)
    pending_units: float = Field(ge=0)
    stock_on_hand: float = Field(ge=0)
    stockout_flag: int = Field(ge=0, le=1)
    promo: int = Field(ge=0, le=1)
    unit_price_mxn: float = Field(ge=0)
    unit_cost_mxn: float = Field(ge=0)
    lead_time_days: int = Field(ge=0)
    demand_pattern: DemandPattern
    abc_class: AbcClass


# Ordered column contracts for every CSV in the project.
CSV_COLUMNS: dict[str, tuple[str, ...]] = {
    "products": (
        "product_id",
        "name",
        "category",
        "unit",
        "price_mxn",
        "cost_mxn",
        "is_ingredient",
        "has_recipe",
        "safety_stock",
        "lead_time_days",
        "demand_pattern",
        "abc_class",
    ),
    "branches": ("branch_id", "business_id", "name", "region"),
    "recipes": (
        "recipe_id",
        "business_id",
        "output_product_id",
        "input_product_id",
        "quantity_per_unit",
    ),
    "economic_events": (
        "event_id",
        "business_id",
        "branch_id",
        "event_type",
        "event_ts",
        "amount_mxn",
        "payment_method",
        "customer_id",
        "product_id",
        "product_name",
        "quantity",
        "unit_price_mxn",
        "unit_cost_mxn",
        "is_cancelled",
        "note",
    ),
    "stock_movements": (
        "movement_id",
        "business_id",
        "branch_id",
        "product_id",
        "movement_ts",
        "direction",
        "reason",
        "quantity",
        "unit_cost_mxn",
        "stock_after",
        "note",
    ),
    "pending_sales": (
        "pending_sale_id",
        "business_id",
        "branch_id",
        "product_id",
        "created_ts",
        "promised_ts",
        "quantity",
        "unit_price_mxn",
        "status",
        "converted_event_id",
    ),
    "demand_daily": (
        "date",
        "business_id",
        "branch_id",
        "product_id",
        "category",
        "unit",
        "units_sold",
        "revenue_mxn",
        "cost_mxn",
        "cancelled_units",
        "pending_units",
        "stock_on_hand",
        "stockout_flag",
        "promo",
        "unit_price_mxn",
        "unit_cost_mxn",
        "lead_time_days",
        "demand_pattern",
        "abc_class",
    ),
}

CSV_MODELS: dict[str, type[BaseModel]] = {
    "products": Product,
    "branches": Branch,
    "recipes": Recipe,
    "economic_events": EconomicEventLine,
    "stock_movements": StockMovement,
    "pending_sales": PendingSale,
    "demand_daily": DailyDemand,
}
