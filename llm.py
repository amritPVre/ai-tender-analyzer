"""LLM routing wrapper for MiniMax-M3 and DeepSeek via NVIDIA NIM API."""

import json
import re
import time
from typing import Callable, Literal, Optional

from openai import APIConnectionError, APITimeoutError, OpenAI, RateLimitError

try:
    from openai import APIStatusError
except ImportError:
    APIStatusError = Exception  # type: ignore

from config import (
    API_MAX_RETRIES,
    API_TIMEOUT_SECONDS,
    DEEPSEEK_ENABLE_THINKING,
    DEEPSEEK_MODEL,
    LLM_MAX_TOKENS,
    MAX_CONTEXT_CHARS,
    MINIMAX_MODEL,
    NVIDIA_API_BASE_URL,
    NVIDIA_DEEPSEEK_API_KEY,
    NVIDIA_MINIMAX_API_KEY,
    SYSTEM_PROMPT,
)

RoutingMode = Literal["text", "scanned"]

_RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}


def _make_client(api_key: str) -> OpenAI:
    return OpenAI(
        api_key=api_key,
        base_url=NVIDIA_API_BASE_URL,
        timeout=API_TIMEOUT_SECONDS,
        max_retries=0,
    )


def get_minimax_client() -> OpenAI:
    if not NVIDIA_MINIMAX_API_KEY:
        raise ValueError("NVIDIA_MINIMAX_API_KEY is not set. Add it to your .env or Streamlit Secrets.")
    return _make_client(NVIDIA_MINIMAX_API_KEY)


def get_deepseek_client() -> OpenAI:
    if not NVIDIA_DEEPSEEK_API_KEY:
        raise ValueError("NVIDIA_DEEPSEEK_API_KEY is not set. Add it to your .env or Streamlit Secrets.")
    return _make_client(NVIDIA_DEEPSEEK_API_KEY)


def _truncate_context(chunks: list[str]) -> str:
    """Cap total retrieved context to avoid oversized prompts and API timeouts."""
    if not chunks:
        return "No context available."

    parts: list[str] = []
    total = 0
    for chunk in chunks:
        if total + len(chunk) > MAX_CONTEXT_CHARS:
            remaining = MAX_CONTEXT_CHARS - total
            if remaining > 400:
                parts.append(chunk[:remaining])
            break
        parts.append(chunk)
        total += len(chunk)

    return "\n\n---\n\n".join(parts)


def _chat_with_retry(client: OpenAI, **kwargs):
    """Retry transient NVIDIA gateway failures (502/503/504) with backoff."""
    last_error: Exception | None = None

    for attempt in range(API_MAX_RETRIES):
        try:
            return client.chat.completions.create(**kwargs)
        except (APIStatusError, APITimeoutError, APIConnectionError, RateLimitError) as exc:
            last_error = exc
            status = getattr(exc, "status_code", None)
            if status not in _RETRYABLE_STATUS and not isinstance(
                exc, (APITimeoutError, APIConnectionError, RateLimitError)
            ):
                raise
            if attempt >= API_MAX_RETRIES - 1:
                raise
            # Longer backoff for rate-limit / resource exhaustion (503)
            wait = min(2 ** attempt * (8 if status == 503 else 4), 45)
            time.sleep(wait)

    if last_error:
        raise last_error
    raise RuntimeError("LLM request failed after retries")


def extract_text_from_page_image(image_b64: str, page_num: int) -> str:
    """Extract text from a single scanned page using MiniMax-M3 vision."""
    client = get_minimax_client()
    response = _chat_with_retry(
        client,
        model=MINIMAX_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"Extract ALL text from this tender document page (page {page_num}). "
                            "Preserve structure, tables, headings, and numbered lists. "
                            "Return only the extracted text with no commentary."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                    },
                ],
            }
        ],
        max_tokens=LLM_MAX_TOKENS,
        temperature=1.0,
        top_p=0.95,
    )
    return response.choices[0].message.content or ""


def call_llm(
    prompt: str,
    context_chunks: list[str],
    routing_mode: RoutingMode,
    stream_callback: Optional[Callable[[str], None]] = None,
) -> str:
    """
    Call the appropriate LLM based on routing mode.
    Scanned PDFs → MiniMax-M3 | Text PDFs → DeepSeek-V4-Flash (both via NVIDIA NIM).
    """
    context = _truncate_context(context_chunks)
    user_message = (
        "Use the following excerpts from the tender document as your primary source. "
        "If information is not present, state 'Not specified in document'.\n\n"
        f"DOCUMENT EXCERPTS:\n{context}\n\n"
        f"TASK:\n{prompt}"
    )

    if routing_mode == "scanned":
        return _call_minimax(user_message, stream_callback)
    return _call_deepseek(user_message, stream_callback)


def _call_minimax(user_message: str, stream_callback: Optional[Callable[[str], None]] = None) -> str:
    client = get_minimax_client()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]
    kwargs = {
        "model": MINIMAX_MODEL,
        "messages": messages,
        "max_tokens": LLM_MAX_TOKENS,
        "temperature": 1.0,
        "top_p": 0.95,
    }

    if stream_callback:
        full_text = ""
        stream = _chat_with_retry(client, **kwargs, stream=True)
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            full_text += delta
            stream_callback(delta)
        return full_text

    response = _chat_with_retry(client, **kwargs)
    return response.choices[0].message.content or ""


def _call_deepseek(user_message: str, stream_callback: Optional[Callable[[str], None]] = None) -> str:
    client = get_deepseek_client()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]
    kwargs = {
        "model": DEEPSEEK_MODEL,
        "messages": messages,
        "temperature": 1.0,
        "top_p": 0.95,
        "max_tokens": LLM_MAX_TOKENS,
    }
    if DEEPSEEK_ENABLE_THINKING:
        kwargs["extra_body"] = {
            "chat_template_kwargs": {"thinking": True, "reasoning_effort": "high"},
        }

    if stream_callback:
        full_text = ""
        stream = _chat_with_retry(client, **kwargs, stream=True)
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            full_text += delta
            stream_callback(delta)
        return full_text

    response = _chat_with_retry(client, **kwargs)
    return response.choices[0].message.content or ""


def parse_json_response(text: str) -> dict:
    """Extract JSON object from LLM response."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return {"raw_text": text}
