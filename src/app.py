import io
import json
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Dict, List
from uuid import uuid4

import gradio as gr
import httpx
import pandas as pd
from PIL import Image

from config import (
    DEEPSEEK_API_BASE,
    DEEPSEEK_API_KEY,
    DEEPSEEK_MODEL,
    MINERU_API_BASE,
    MINERU_API_KEY,
    QUARTZY_API_BASE,
    QUARTZY_API_TOKEN,
    QUARTZY_LAB_ID,
    QUARTZY_TYPE_ID,
)
from models import QuartzyRequest, Receipt

DF_COLNAMES = ["名称", "货号", "数量", "单位", "单价", "供应商", "备注", "时间"]
MINERU_POLL_INTERVAL_SECONDS = 5
MINERU_MAX_POLLS = 60

receipt_markdown_prompt = """
请从下面的收据 Markdown 中抽取信息，输出必须严格符合给定 JSON schema。

需要抽取日期、供应商、商品列表（名称、数量、货号、单位、单价）和总金额、备注。
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


def gr_img_to_bytes(img: gr.Image) -> bytes:
    pil_img = Image.fromarray(img.astype("uint8"), "RGB")
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG")
    return buf.getvalue()


def mineru_parse_markdown(img: gr.Image) -> str:
    if not MINERU_API_KEY:
        raise ValueError("MINERU_API_KEY is not configured")

    base_url = MINERU_API_BASE.rstrip("/")
    headers = {
        "Authorization": f"Bearer {MINERU_API_KEY}",
        "Content-Type": "application/json",
    }
    filename = f"receipt-{uuid4().hex}.jpg"
    payload = {
        "files": [{"name": filename, "data_id": filename, "is_ocr": True}],
        "model_version": "vlm",
        "language": "ch",
        "enable_table": True,
        "enable_formula": False,
    }

    with httpx.Client(timeout=60.0, follow_redirects=True) as client:
        response = client.post(
            f"{base_url}/api/v4/file-urls/batch",
            headers=headers,
            json=payload,
        )
        response.raise_for_status()
        created = response.json()
        if created.get("code") != 0:
            raise RuntimeError(f"MinerU create task failed: {created}")

        batch_id = created["data"]["batch_id"]
        upload_url = created["data"]["file_urls"][0]
        upload_response = client.put(upload_url, content=gr_img_to_bytes(img))
        upload_response.raise_for_status()

        result = None
        for _ in range(MINERU_MAX_POLLS):
            poll_response = client.get(
                f"{base_url}/api/v4/extract-results/batch/{batch_id}",
                headers=headers,
            )
            poll_response.raise_for_status()
            result = poll_response.json()
            if result.get("code") != 0:
                raise RuntimeError(f"MinerU parse failed: {result}")

            items = result.get("data", {}).get("extract_result") or []
            if items and all(item.get("state") in {"done", "failed"} for item in items):
                break
            time.sleep(MINERU_POLL_INTERVAL_SECONDS)
        else:
            raise TimeoutError(f"MinerU parse timed out: {batch_id}")

        item = (result.get("data", {}).get("extract_result") or [])[0]
        if item.get("state") != "done":
            raise RuntimeError(f"MinerU parse did not complete: {item}")

        zip_response = client.get(item["full_zip_url"])
        zip_response.raise_for_status()

    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = Path(tmpdir) / "mineru_output.zip"
        zip_path.write_bytes(zip_response.content)
        with zipfile.ZipFile(zip_path) as archive:
            try:
                return archive.read("full.md").decode("utf-8")
            except KeyError as exc:
                raise RuntimeError("MinerU output did not include full.md") from exc


def extract_receipt_from_markdown(markdown: str) -> Receipt:
    if not DEEPSEEK_API_KEY:
        raise ValueError("DEEPSEEK_API_KEY is not configured")

    schema = Receipt.model_json_schema()
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You extract receipt data from OCR markdown. "
                    "Return only valid JSON matching the provided schema."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"{receipt_markdown_prompt}\n\n"
                    f"JSON schema:\n{json.dumps(schema, ensure_ascii=False)}\n\n"
                    f"Markdown:\n{markdown}"
                ),
            },
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0,
    }

    with httpx.Client(timeout=60.0, follow_redirects=True) as client:
        response = client.post(
            f"{DEEPSEEK_API_BASE.rstrip('/')}/chat/completions",
            headers={
                "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        response.raise_for_status()
        result = response.json()

    content = result["choices"][0]["message"]["content"]
    return Receipt.model_validate_json(content)


def parse_receipt_image(img) -> Receipt:
    markdown = mineru_parse_markdown(img)
    return extract_receipt_from_markdown(markdown)


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
            price={"amount": str(row["单价"] * 100), "currency": "CNY"},
            quantity=row["数量"],
            # notes: date, unit, comment
            notes=f"Date: {row['时间']}, comment: {row['备注']}, unit: {row['单位']}",
        )
        requests.append(request)
    return requests


def show_uploaded_image(file: gr.File) -> Image.Image:
    return Image.open(file.name)


def process_receipts(img: gr.Image) -> tuple:
    receipt = parse_receipt_image(img)
    df_data = to_dataframe(receipt)
    return df_data, receipt.model_dump()


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
    return results


with gr.Blocks() as demo:
    gr.Markdown("# 收据识别与库存提交")

    with gr.Row():
        img_input = gr.Image(label="上传收据图片")

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
        inputs=[img_input],
        outputs=[table, hidden_json],
    )

    submit_btn.click(fn=submit_all, inputs=[table], outputs=[result_box])

if __name__ == "__main__":
    demo.launch()
