"""Application configuration and constants."""

import os

from dotenv import load_dotenv

load_dotenv()


def _clean_key(value: str) -> str:
    """Strip whitespace and accidental surrounding quotes from API keys."""
    if not value:
        return ""
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
        value = value[1:-1].strip()
    return value


def _load_secret(name: str) -> str:
    """Load from environment variables or Streamlit Cloud secrets."""
    value = _clean_key(os.getenv(name, ""))
    if value:
        return value

    try:
        import streamlit as st

        if name in st.secrets:
            return _clean_key(str(st.secrets[name]))
    except Exception:
        pass

    return ""


def _resolve_keys() -> tuple[str, str]:
    """Resolve MiniMax and DeepSeek keys with optional shared fallback."""
    minimax = _load_secret("NVIDIA_MINIMAX_API_KEY")
    deepseek = _load_secret("NVIDIA_DEEPSEEK_API_KEY")
    shared = _load_secret("NVIDIA_API_KEY")

    if not minimax and shared:
        minimax = shared
    if not deepseek and shared:
        deepseek = shared

    return minimax, deepseek


NVIDIA_MINIMAX_API_KEY, NVIDIA_DEEPSEEK_API_KEY = _resolve_keys()

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
TECHNICAL_MAX_TOKENS = int(os.getenv("TECHNICAL_MAX_TOKENS", "4096"))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "24000"))
TECHNICAL_CONTEXT_CHARS = int(os.getenv("TECHNICAL_CONTEXT_CHARS", "14000"))
API_MAX_RETRIES = int(os.getenv("API_MAX_RETRIES", "4"))
API_CALL_DELAY_SECONDS = int(os.getenv("API_CALL_DELAY_SECONDS", "6"))
TECHNICAL_RETRIEVE_K = int(os.getenv("TECHNICAL_RETRIEVE_K", "6"))
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


def key_fingerprint(key: str) -> str:
    """Safe preview for UI/debugging without exposing full secret."""
    if not key:
        return "not set"
    if len(key) <= 12:
        return "set (too short — check key)"
    return f"{key[:8]}…{key[-4:]}"


def keys_configured() -> dict[str, bool]:
    return {
        "minimax": bool(NVIDIA_MINIMAX_API_KEY),
        "deepseek": bool(NVIDIA_DEEPSEEK_API_KEY),
    }
