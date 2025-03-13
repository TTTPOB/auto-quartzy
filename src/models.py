from typing import List, Optional
from pydantic import BaseModel
from datetime import date
from config import QUARTZY_LAB_ID, QUARTZY_TYPE_ID


class Item(BaseModel):
    name: str
    quantity: int
    unit: str
    price: float
    stock_id: str
    vendor: str
    comment: str


class Receipt(BaseModel):
    date: date
    total_amount: float
    items: List[Item]


class QuartzyRequest(BaseModel):
    lab_id: str
    type_id: str
    name: str
    vendor_name: str
    catalog_number: str
    price: dict  # {"amount": "1250", "currency": "CNY"}
    quantity: int
    notes: str
