import os
from dotenv import load_dotenv

load_dotenv()

# DeepSeek API settings
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_API_BASE = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")

# MinerU API settings
MINERU_API_KEY = os.getenv("MINERU_API_KEY")
MINERU_API_BASE = os.getenv("MINERU_API_BASE", "https://mineru.net")

# Quartzy API settings
QUARTZY_API_TOKEN = os.getenv("QUARTZY_API_TOKEN")
QUARTZY_AUTH0_ACCESS_TOKEN = os.getenv("QUARTZY_AUTH0_ACCESS_TOKEN") or os.getenv(
    "QUARTZY_GRAPHQL_JWT"
)
QUARTZY_API_BASE = "https://api.quartzy.com"
QUARTZY_GRAPHQL_URL = os.getenv("QUARTZY_GRAPHQL_URL", "https://graphql.quartzy.com/")
QUARTZY_LAB_ID = os.getenv("QUARTZY_LAB_ID")
QUARTZY_TYPE_ID = os.getenv("QUARTZY_TYPE_ID")

# Local settings
RECEIPT_ARCHIVE_DIR = "receipts"
