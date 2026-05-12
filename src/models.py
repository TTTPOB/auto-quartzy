from typing import Any, List
from pydantic import BaseModel, field_validator, model_validator
from datetime import date as Date
from config import QUARTZY_LAB_ID, QUARTZY_TYPE_ID


class Item(BaseModel):
    name: str | None = None
    quantity: int | None = None
    unit: str | None = None
    price: float | None = None
    stock_id: str | None = None
    vendor: str | None = None
    comment: str | None = None

    @field_validator("quantity", "price", mode="before")
    @classmethod
    def empty_number_to_none(cls, value: Any) -> Any:
        if value == "":
            return None
        return value

    def has_any_value(self) -> bool:
        return any(
            getattr(self, field_name) not in (None, "")
            for field_name in self.model_fields
        )


class Receipt(BaseModel):
    date: Date | None = None
    total_amount: float | None = None
    items: List[Item] = []

    @field_validator("date", "total_amount", mode="before")
    @classmethod
    def empty_value_to_none(cls, value: Any) -> Any:
        if value == "":
            return None
        return value

    @model_validator(mode="after")
    def remove_empty_items(self) -> "Receipt":
        self.items = [item for item in self.items if item.has_any_value()]
        return self


class QuartzyRequest(BaseModel):
    lab_id: str
    type_id: str
    name: str
    vendor_name: str
    catalog_number: str
    price: dict  # {"amount": "1250", "currency": "CNY"}
    quantity: int
    notes: str
