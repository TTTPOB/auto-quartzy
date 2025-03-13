import os
from dotenv import load_dotenv

load_dotenv()

# OpenRouter API settings
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "anthropic/claude-3-opus-20240229")

# Quartzy API settings
QUARTZY_API_TOKEN = os.getenv("QUARTZY_API_TOKEN")
QUARTZY_API_BASE = "https://api.quartzy.com"
QUARTZY_LAB_ID = os.getenv("QUARTZY_LAB_ID")
QUARTZY_TYPE_ID = os.getenv("QUARTZY_TYPE_ID")

# Local settings
RECEIPT_ARCHIVE_DIR = "receipts"