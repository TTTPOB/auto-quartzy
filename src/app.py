import base64
import hashlib
import html
import io
import json
import tempfile
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List
from uuid import uuid4

import httpx
import pandas as pd
import streamlit as st
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
from quartzy_upload import (
    QuartzyUploadedFile,
    attach_uploaded_file_to_order_request,
    attachment_filename,
    upload_receipt_image,
)

DF_COLNAMES = ["名称", "货号", "数量", "单位", "单价", "供应商", "备注", "时间"]
MINERU_POLL_INTERVAL_SECONDS = 5
MINERU_MAX_POLLS = 60
GALLERY_COLUMNS = 4
MAX_PARSE_CONCURRENCY = 5
TABLE_HEADER_HEIGHT = 38
TABLE_ROW_HEIGHT = 36
TABLE_EXTRA_HEIGHT = 18
TABLE_MIN_HEIGHT = 96
TABLE_MAX_HEIGHT = 360
GALLERY_SCROLL_HEIGHT = 660

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


def image_to_jpeg_bytes(img: Image.Image) -> bytes:
    pil_img = img.convert("RGB")
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG")
    return buf.getvalue()


def image_to_data_url(img: Image.Image, max_size: tuple[int, int] = (360, 240)) -> str:
    thumbnail = img.copy()
    thumbnail.thumbnail(max_size)
    encoded = base64.b64encode(image_to_jpeg_bytes(thumbnail)).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def mineru_parse_markdown(img: Image.Image) -> str:
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
        upload_response = client.put(upload_url, content=image_to_jpeg_bytes(img))
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


def process_receipts(img: Image.Image) -> tuple:
    receipt = parse_receipt_image(img)
    df_data = to_dataframe(receipt)
    return df_data, receipt.model_dump()


def submit_all(
    edited_table: pd.DataFrame,
    receipt_image: Image.Image,
    receipt_name: str,
    uploaded_file: QuartzyUploadedFile | None = None,
) -> tuple[List[Dict], QuartzyUploadedFile | None]:
    results = []
    if uploaded_file is None:
        try:
            uploaded_file = upload_receipt_image(
                image_to_jpeg_bytes(receipt_image),
                attachment_filename(receipt_name),
                "image/jpeg",
            )
        except Exception as exc:
            return [
                {
                    "order_request": None,
                    "attachment": None,
                    "error": f"Receipt image upload failed before creating order requests: {exc}",
                }
            ], None

    for req in df_to_quartzy_requests(pd.DataFrame(edited_table, columns=DF_COLNAMES)):
        response = httpx.post(
            f"{QUARTZY_API_BASE}/order-requests",
            headers={
                "Access-Token": f"{QUARTZY_API_TOKEN}",
                "Content-Type": "application/json",
            },
            json=req.model_dump(),
        )
        try:
            response.raise_for_status()
            order_result = response.json()
        except Exception as exc:
            results.append(
                {
                    "request": req.model_dump(),
                    "order_request": None,
                    "attachment": None,
                    "error": str(exc),
                }
            )
            continue

        attachment_result = None
        attachment_error = None
        order_request_id, order_request_id_source = extract_order_request_uuid(order_result)
        if uploaded_file is not None and order_request_id:
            if not uploaded_file.uuid:
                attachment_error = "Quartzy createFile response did not include file uuid."
            else:
                try:
                    attachment_result = attach_uploaded_file_to_order_request(
                        uploaded_file.uuid,
                        order_request_id,
                    )
                    attachment_error = None
                except Exception as exc:
                    attachment_error = str(exc)
        elif uploaded_file is not None:
            attachment_error = (
                "Could not find the UUID required by "
                "attachFileToOrderRequest.input.orderRequestId in Quartzy API response."
            )

        results.append(
            {
                "request": req.model_dump(),
                "order_request": order_result,
                "order_request_id": order_request_id,
                "order_request_id_source": order_request_id_source,
                "attachment": attachment_result,
                "attachment_error": attachment_error,
                "reused_file_id": uploaded_file.file_id if uploaded_file else None,
                "reused_file_uuid": uploaded_file.uuid if uploaded_file else None,
            }
        )
    return results, uploaded_file


def extract_order_request_uuid(value) -> tuple[str | None, str | None]:
    if not isinstance(value, dict) or not is_uuid_like(value.get("id")):
        return None, None
    return value["id"], "id"


def is_uuid_like(value) -> bool:
    if not isinstance(value, str) or len(value) != 36:
        return False
    parts = value.split("-")
    return [len(part) for part in parts] == [8, 4, 4, 4, 12]


def uploaded_file_id(file) -> str:
    return hashlib.sha256(file.getvalue()).hexdigest()


def dataframe_editor_height(df: pd.DataFrame) -> int:
    visible_rows = max(len(df), 1)
    height = TABLE_HEADER_HEIGHT + visible_rows * TABLE_ROW_HEIGHT + TABLE_EXTRA_HEIGHT
    return min(max(height, TABLE_MIN_HEIGHT), TABLE_MAX_HEIGHT)


def render_html_text(content: str, class_name: str, min_height: int) -> None:
    # st.html lets the browser size this text block directly; st.markdown inside
    # fixed-height containers can clip CJK text and long filenames at the top.
    st.html(
        (
            f'<div class="{class_name}" '
            f'style="min-height:{min_height}px;">'
            f"{html.escape(content)}</div>"
        )
    )


def empty_receipt_record(file_id: str, name: str, image: Image.Image) -> Dict:
    return {
        "id": file_id,
        "name": name,
        "image": image,
        "thumbnail_url": image_to_data_url(image),
        "df": pd.DataFrame(columns=DF_COLNAMES),
        "json": None,
        "submit_result": None,
        "quartzy_uploaded_file": None,
        "editor_version": 0,
        "parse_future": None,
        "parse_status": "未识别",
        "parse_error": None,
    }


def get_parse_executor() -> ThreadPoolExecutor:
    if "parse_executor" not in st.session_state:
        st.session_state.parse_executor = ThreadPoolExecutor(
            max_workers=MAX_PARSE_CONCURRENCY
        )
    return st.session_state.parse_executor


def collect_parse_results() -> None:
    for record in st.session_state.receipts.values():
        future = record.get("parse_future")
        if future is None or not future.done():
            continue

        try:
            df_data, receipt_json = future.result()
        except Exception as exc:
            record["parse_status"] = "识别失败"
            record["parse_error"] = str(exc)
        else:
            record["df"] = df_data
            record["json"] = receipt_json
            record["submit_result"] = None
            record["editor_version"] += 1
            record["parse_status"] = "已识别"
            record["parse_error"] = None
        finally:
            record["parse_future"] = None


def submit_parse_task(record: Dict) -> bool:
    future = record.get("parse_future")
    if future is not None and not future.done():
        return False

    record["parse_status"] = "识别中"
    record["parse_error"] = None
    record["submit_result"] = None
    record["parse_future"] = get_parse_executor().submit(
        process_receipts,
        record["image"].copy(),
    )
    return True


def select_receipt(file_id: str) -> None:
    st.session_state.selected_receipt_id = file_id


def main() -> None:
    st.set_page_config(page_title="收据识别与库存提交", layout="wide")
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.25rem;
            padding-bottom: 1.25rem;
            max-width: 100%;
        }
        .receipt-title {
            display: flex;
            align-items: center;
            font-size: 1.95rem;
            font-weight: 700;
            line-height: 1.45;
            white-space: normal;
            overflow-wrap: anywhere;
            padding: 0.85rem 0 0.75rem;
            margin: 0;
            box-sizing: border-box;
        }
        .empty-state {
            display: flex;
            align-items: center;
            padding: 1rem 1.1rem;
            border-radius: 0.5rem;
            background: rgba(46, 134, 193, 0.16);
            color: #1c7ed6;
            line-height: 1.6;
            overflow-wrap: anywhere;
            box-sizing: border-box;
        }
        div[data-testid="stImage"] img {
            object-fit: contain;
        }
        .receipt-gallery-card {
            border: 1px solid rgba(128, 128, 128, 0.28);
            border-radius: 7px;
            padding: 7px;
            box-sizing: border-box;
            height: 220px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }
        .receipt-gallery-card.selected {
            border-color: #ff4b4b;
            box-shadow: inset 0 0 0 2px #ff4b4b;
        }
        .receipt-gallery-card img {
            height: 155px;
            width: 100%;
            object-fit: contain;
        }
        .receipt-gallery-caption {
            margin-top: 0.45rem;
            font-size: 0.92rem;
            line-height: 1.35;
            overflow-wrap: anywhere;
            max-height: 46px;
            overflow: hidden;
        }
        .receipt-gallery-select button {
            min-height: 220px;
            height: 220px;
            padding: 0.5rem 0.25rem;
            white-space: normal;
            word-break: keep-all;
            line-height: 1.35;
        }
        .receipt-preview div[data-testid="stImage"] img {
            max-height: 520px;
        }
        div[data-testid="stDataFrame"] div[role="columnheader"],
        div[data-testid="stDataFrame"] div[role="gridcell"] {
            min-height: 34px;
            line-height: 1.25;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    left_col, work_col = st.columns([1.15, 4.02], gap="large")

    if "receipts" not in st.session_state:
        st.session_state.receipts = {}
    if "receipt_order" not in st.session_state:
        st.session_state.receipt_order = []
    if "selected_receipt_id" not in st.session_state:
        st.session_state.selected_receipt_id = None

    collect_parse_results()

    with left_col:
        uploaded_files = st.file_uploader(
            "上传收据图片",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=True,
        )

    current_ids = []
    for uploaded_file in uploaded_files:
        file_id = uploaded_file_id(uploaded_file)
        current_ids.append(file_id)
        if file_id not in st.session_state.receipts:
            image = Image.open(uploaded_file).convert("RGB")
            st.session_state.receipts[file_id] = empty_receipt_record(
                file_id,
                uploaded_file.name,
                image,
            )

    st.session_state.receipt_order = current_ids
    for file_id in list(st.session_state.receipts):
        if file_id not in current_ids:
            del st.session_state.receipts[file_id]

    if (
        st.session_state.selected_receipt_id not in st.session_state.receipts
        and current_ids
    ):
        st.session_state.selected_receipt_id = current_ids[0]
    if not current_ids:
        st.session_state.selected_receipt_id = None

    if not current_ids:
        with work_col:
            render_html_text(
                "上传一张或多张收据图片后开始识别。",
                "empty-state",
                86,
            )
        return

    active_parse_count = sum(
        1
        for record in st.session_state.receipts.values()
        if record.get("parse_future") is not None
    )

    with left_col:
        gallery_actions = st.columns(2)
        with gallery_actions[0]:
            all_disabled = not any(
                st.session_state.receipts[file_id]["parse_status"]
                in {"未识别", "识别失败"}
                for file_id in current_ids
            )
            if st.button("识别全部", disabled=all_disabled, width="stretch"):
                for file_id in current_ids:
                    record = st.session_state.receipts[file_id]
                    if record["parse_status"] in {"未识别", "识别失败"}:
                        submit_parse_task(record)
                st.rerun()
        with gallery_actions[1]:
            if st.button(
                "刷新",
                disabled=active_parse_count == 0,
                width="stretch",
            ):
                st.rerun()

        with st.container(height=GALLERY_SCROLL_HEIGHT, border=False):
            for file_id in current_ids:
                gallery_record = st.session_state.receipts[file_id]
                selected = file_id == st.session_state.selected_receipt_id
                card_class = "receipt-gallery-card selected" if selected else "receipt-gallery-card"
                image_col, select_col = st.columns([5, 1], gap="small")
                with image_col:
                    st.markdown(
                        (
                            f'<div class="{card_class}">'
                            f'<img src="{gallery_record["thumbnail_url"]}" alt="">'
                            f'<div class="receipt-gallery-caption">'
                            f'{html.escape(gallery_record["name"])} '
                            f'{html.escape(gallery_record["parse_status"])}'
                            "</div></div>"
                        ),
                        unsafe_allow_html=True,
                    )
                with select_col:
                    st.markdown(
                        '<div class="receipt-gallery-select">',
                        unsafe_allow_html=True,
                    )
                    st.button(
                        "选择",
                        key=f"select_{file_id}",
                        disabled=selected,
                        width="stretch",
                        on_click=select_receipt,
                        args=(file_id,),
                    )
                    st.markdown("</div>", unsafe_allow_html=True)

    record = st.session_state.receipts[st.session_state.selected_receipt_id]

    with work_col:
        render_html_text(record["name"], "receipt-title", 92)

        action_bar = st.container()

        if record["parse_error"]:
            st.error(record["parse_error"])

        edited_table = st.data_editor(
            record["df"],
            num_rows="dynamic",
            width="stretch",
            height=dataframe_editor_height(record["df"]),
            key=f"receipt_editor_{record['id']}_{record['editor_version']}",
        )
        record["df"] = edited_table

        st.markdown('<div class="receipt-preview">', unsafe_allow_html=True)
        if record["submit_result"] is not None:
            st.json(record["submit_result"])
        else:
            st.image(record["image"], width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)

        with action_bar:
            parse_col, submit_col, spacer_col = st.columns([1, 1, 4], gap="small")
            parsing_current = record.get("parse_future") is not None
            with parse_col:
                if st.button(
                    "识别当前收据",
                    disabled=parsing_current,
                    width="stretch",
                ):
                    submit_parse_task(record)
                    st.rerun()
            with submit_col:
                if st.button(
                    "提交",
                    disabled=edited_table.empty,
                    width="stretch",
                ):
                    with st.spinner("正在提交至 Quartzy..."):
                        (
                            record["submit_result"],
                            record["quartzy_uploaded_file"],
                        ) = submit_all(
                            edited_table,
                            record["image"],
                            record["name"],
                            record.get("quartzy_uploaded_file"),
                        )
                    st.rerun()

    if active_parse_count:
        time.sleep(1)
        st.rerun()


if __name__ == "__main__":
    main()
