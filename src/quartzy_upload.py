from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from config import QUARTZY_AUTH0_ACCESS_TOKEN, QUARTZY_GRAPHQL_URL


CREATE_FILE_MUTATION = """
mutation Mutation($input: CreateFileInput) {
  createFile(input: $input) {
    id
    uuid
    temporaryUploadUrl
    type
    name
    __typename
  }
}
"""

ATTACH_FILE_TO_ORDER_REQUEST_MUTATION = """
mutation Mutation($input: AttachFileToOrderRequestInput) {
  attachFileToOrderRequest(input: $input) {
    attachment {
      ...AttachmentFragment
      __typename
    }
    __typename
  }
}

fragment AttachmentFragment on Attachment {
  id
  internalId
  file {
    id
    url
    name
    type
    uuid
    __typename
  }
  createdAt
  createdBy {
    id
    __typename
  }
  __typename
}
"""


@dataclass(frozen=True)
class QuartzyUploadedFile:
    file_id: str
    uuid: str | None
    name: str
    temporary_upload_url: str


def upload_receipt_image(
    image_bytes: bytes,
    filename: str,
    content_type: str = "image/jpeg",
) -> QuartzyUploadedFile:
    if not QUARTZY_AUTH0_ACCESS_TOKEN:
        raise RuntimeError("QUARTZY_AUTH0_ACCESS_TOKEN is required to upload attachments.")

    file_info = _create_file(filename, len(image_bytes), content_type)
    upload_url = _required_str(file_info, "temporaryUploadUrl")

    with httpx.Client(timeout=60) as client:
        response = client.put(
            upload_url,
            content=image_bytes,
            headers={"Content-Type": content_type},
        )
        response.raise_for_status()

    return QuartzyUploadedFile(
        file_id=_required_str(file_info, "id"),
        uuid=file_info.get("uuid"),
        name=_required_str(file_info, "name"),
        temporary_upload_url=upload_url,
    )


def attach_uploaded_file_to_order_request(
    file_uuid: str,
    order_request_id: str,
) -> dict[str, Any]:
    data = _graphql(
        ATTACH_FILE_TO_ORDER_REQUEST_MUTATION,
        {
            "input": {
                "fileId": file_uuid,
                "orderRequestId": order_request_id,
            }
        },
    )
    attachment = data.get("attachFileToOrderRequest")
    if not isinstance(attachment, dict):
        raise RuntimeError(f"Unexpected attachFileToOrderRequest response: {data}")
    return attachment


def attachment_filename(receipt_name: str) -> str:
    stem = Path(receipt_name).stem or "receipt"
    return f"{stem}.jpg"


def _create_file(filename: str, size: int, content_type: str) -> dict[str, Any]:
    data = _graphql(
        CREATE_FILE_MUTATION,
        {
            "input": {
                "name": filename,
                "type": content_type,
                "size": size,
            }
        },
    )
    file_info = data.get("createFile")
    if not isinstance(file_info, dict):
        raise RuntimeError(f"Unexpected createFile response: {data}")
    return file_info


def _graphql(query: str, variables: dict[str, Any]) -> dict[str, Any]:
    response = httpx.post(
        QUARTZY_GRAPHQL_URL,
        params={"op": "Mutation"},
        headers={
            "auth0-access-token": QUARTZY_AUTH0_ACCESS_TOKEN or "",
            "Content-Type": "application/json",
            "Origin": "https://app.quartzy.com",
            "Referer": "https://app.quartzy.com/",
            "apollographql-client-name": "lab-frontend",
        },
        json={
            "operationName": "Mutation",
            "variables": variables,
            "query": query,
        },
        timeout=30,
    )
    response.raise_for_status()
    body = response.json()
    if body.get("errors"):
        raise RuntimeError(f"Quartzy GraphQL error: {body['errors']}")
    data = body.get("data")
    if not isinstance(data, dict):
        raise RuntimeError(f"Unexpected Quartzy GraphQL response: {body}")
    return data


def _required_str(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"Quartzy response missing {key}: {data}")
    return value
