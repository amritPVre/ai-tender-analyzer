"""Application configuration and constants."""

import os

from dotenv import load_dotenv

load_dotenv()

# NVIDIA NIM API keys (local: .env | Streamlit Cloud: app Secrets)
NVIDIA_MINIMAX_API_KEY = os.getenv("NVIDIA_MINIMAX_API_KEY", "")
NVIDIA_DEEPSEEK_API_KEY = os.getenv("NVIDIA_DEEPSEEK_API_KEY", "")

# NVIDIA integrate API
NVIDIA_API_BASE_URL = "https://integrate.api.nvidia.com/v1"
MINIMAX_MODEL = "minimaxai/minimax-m3"
DEEPSEEK_MODEL = "deepseek-ai/deepseek-v4-flash"

# RAG settings (~4 chars per token)
CHUNK_CHARS = 3200  # ~800 tokens
OVERLAP_CHARS = 600  # ~150 tokens
RETRIEVE_K = 10

# LLM / API settings (tuned for Streamlit Cloud timeouts)
API_TIMEOUT_SECONDS = int(os.getenv("API_TIMEOUT_SECONDS", "180"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "8192"))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "24000"))
API_MAX_RETRIES = int(os.getenv("API_MAX_RETRIES", "3"))
API_CALL_DELAY_SECONDS = int(os.getenv("API_CALL_DELAY_SECONDS", "4"))
# Thinking mode is slow and often causes NVIDIA gateway 504 timeouts on Cloud
DEEPSEEK_ENABLE_THINKING = os.getenv("DEEPSEEK_ENABLE_THINKING", "false").lower() == "true"

# PDF detection
SCANNED_THRESHOLD = 80  # avg chars per page
SAMPLE_PAGES = 5
OCR_DPI = 200

# Brand colors
COLORS = {
    "navy": "#0B1F3A",
    "teal": "#14B8A6",
    "amber": "#FF8C32",
    "cream": "#FAF3DC",
    "slate": "#8C97AC",
}

SYSTEM_PROMPT = (
    "You are a Senior Solar EPC Tender Analysis Expert with deep knowledge of "
    "MNRE, SECI, CPWD, and international solar procurement standards. "
    "You analyze tender documents with engineering precision."
)
