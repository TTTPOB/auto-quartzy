import base64
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import gradio as gr
import httpx
import pandas as pd
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from PIL import Image

from config import (
    OPENROUTER_API_KEY,
    OPENROUTER_MODEL,
    QUARTZY_API_BASE,
    QUARTZY_API_TOKEN,
    QUARTZY_LAB_ID,
    QUARTZY_TYPE_ID,
)
from models import QuartzyRequest, Receipt

DF_COLNAMES = ["名称", "货号", "数量", "单位", "单价", "供应商", "备注", "时间"]

img_prompt = """
请解析这张收据图片中的信息，
包括日期、供应商、商品列表（名称、数量、货号、单位、单价）和总金额, 备注。
如果你发现某项解析不对，在里面填上对应类型的错误值。
一些常见的供应商
生工
赛音图
NEB
Thermo
金石百优
如果碰到泽平 你要填QSP
如果碰到恒诺创新 你要填卓一航
以上提到的这些你就写我提到的简称 全名太长了没必要
但是如果品牌里有东西就按品牌里面的填啊，品牌里面写杂牌或者国产，或者根本没这栏
才按我上面说的来
有疑虑的东西填comment里，收据大抬头也填comment

单位这个field要把规格也一起放进来
"""


def parse_receipt_image(image_path: str) -> Receipt:
    model = init_chat_model(
        model=OPENROUTER_MODEL,
        api_key=OPENROUTER_API_KEY,
        model_provider="openai",
    ).with_structured_output(Receipt, method="function_calling")

    ext = os.path.splitext(image_path)[-1].lower().replace(".", "") or "jpeg"
    mime_type = f"image/{'jpeg' if ext == 'jpg' else ext}"

    with open(image_path, "rb") as img_file:
        img_bytes = img_file.read()
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")

    message = HumanMessage(
        content=[
            {"type": "text", "text": img_prompt},
            {"type": "image_url", "image_url": f"data:{mime_type};base64,{img_base64}"},
        ]
    )
    response = model.invoke([message])
    return response


def to_dataframe(receipt: Receipt) -> pd.DataFrame:
    df_factory = {}
    df_factory["名称"] = [item.name for item in receipt.items]
    df_factory["货号"] = [item.stock_id for item in receipt.items]
    df_factory["数量"] = [item.quantity for item in receipt.items]
    df_factory["单位"] = [item.unit for item in receipt.items]
    df_factory["单价"] = [item.price for item in receipt.items]
    df_factory["供应商"] = [item.vendor for item in receipt.items]
    df_factory["备注"] = [item.comment for item in receipt.items]
    df_factory["时间"] = [receipt.date for _ in receipt.items]
    return pd.DataFrame(df_factory)


def df_to_quartzy_requests(df: pd.DataFrame) -> List[QuartzyRequest]:
    requests = []
    for _, row in df.iterrows():
        request = QuartzyRequest(
            lab_id=QUARTZY_LAB_ID,
            type_id=QUARTZY_TYPE_ID,
            name=row["名称"],
            vendor_name=row["供应商"],
            catalog_number=row["货号"],
            price={"amount": str(row["单价"] * 100), "currency": "N/A"},
            quantity=row["数量"],
            # notes: date, unit, comment
            notes=f"Date: {row['时间']}, comment: {row['备注']}, unit: {row['单位']}",
        )
        requests.append(request)
    return requests


def show_uploaded_image(file: gr.File) -> Image.Image:
    return Image.open(file.name)


def process_receipts(file: gr.File) -> tuple:
    file_path = file.name
    receipt = parse_receipt_image(file_path)
    df_data = to_dataframe(receipt)
    image_pil = Image.open(file_path)
    return image_pil, df_data, receipt.model_dump(), receipt


def submit_all(edited_table: pd.DataFrame) -> Dict:
    results = []
    for req in df_to_quartzy_requests(pd.DataFrame(edited_table, columns=DF_COLNAMES)):
        response = httpx.post(
            f"{QUARTZY_API_BASE}/order-requests",
            headers={
                "Access-Token": f"{QUARTZY_API_TOKEN}",
                "Content-Type": "application/json",
            },
            json=req.model_dump(),
        )
        results.append(response.json())
    return {"submitted": results}


with gr.Blocks() as demo:
    gr.Markdown("# 收据识别与库存提交")

    with gr.Row():
        file_input = gr.File(
            label="上传收据图片", file_types=[".jpg", ".jpeg", ".png", ".webp"]
        )
        image_view = gr.Image(label="原始收据预览")

    file_input.change(fn=show_uploaded_image, inputs=[file_input], outputs=[image_view])

    with gr.Row():
        table = gr.Dataframe(
            headers=DF_COLNAMES,
            label="识别商品列表（可修改）",
            interactive=True,
            row_count="dynamic",
            col_count=8,
        )

    hidden_json = gr.JSON(visible=False)
    process_btn = gr.Button("识别收据")
    submit_btn = gr.Button("提交至 Quartzy")
    result_box = gr.JSON(label="API提交结果")

    process_btn.click(
        fn=process_receipts,
        inputs=[file_input],
        outputs=[image_view, table, hidden_json],
    )

    submit_btn.click(fn=submit_all, inputs=[table], outputs=[result_box])

if __name__ == "__main__":
    demo.launch()
