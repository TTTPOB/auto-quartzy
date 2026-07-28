import sys
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import app
from quartzy_upload import QuartzyUploadedFile


class FakeResponse:
    def __init__(self, body):
        self.body = body

    def raise_for_status(self):
        return None

    def json(self):
        return self.body


class SubmitAllTest(unittest.TestCase):
    def test_attaches_receipt_to_first_item_and_references_it_from_later_items(self):
        first_id = str(uuid4())
        second_id = str(uuid4())
        requests = []

        def create_order_request(*_, **kwargs):
            requests.append(kwargs["json"])
            order_id = first_id if len(requests) == 1 else second_id
            return FakeResponse(
                {"id": order_id, "app_url": f"https://app.quartzy.com/requests/{order_id}"}
            )

        table = pd.DataFrame(
            [
                ["Buffer", "BUF-1", 1, "bottle", 100, "Vendor", "", "2026-07-28"],
                ["Tube", "TUBE-1", 2, "pack", 200, "Vendor", "", "2026-07-28"],
            ],
            columns=app.DF_COLNAMES,
        )
        uploaded_file = QuartzyUploadedFile("file-id", "file-uuid", "receipt.jpg", "https://upload")

        with (
            patch("app.httpx.post", side_effect=create_order_request),
            patch("app.upload_receipt_image", return_value=uploaded_file) as upload,
            patch("app.attach_uploaded_file_to_order_request", return_value={"attachment": {}}) as attach,
        ):
            results, returned_file = app.submit_all(
                table,
                Image.new("RGB", (1, 1)),
                "receipt.png",
            )

        upload.assert_called_once()
        attach.assert_called_once_with("file-uuid", first_id)
        self.assertIs(returned_file, uploaded_file)
        self.assertNotIn("Receipt image attached to", requests[0]["notes"])
        self.assertIn("Receipt image attached to: Buffer (Date: 2026-07-28)", requests[1]["notes"])
        self.assertIn(f"https://app.quartzy.com/requests/{first_id}", requests[1]["notes"])
        self.assertIsNotNone(results[0]["attachment"])
        self.assertIsNone(results[1]["attachment"])


if __name__ == "__main__":
    unittest.main()
