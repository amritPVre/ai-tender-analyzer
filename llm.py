"""LLM routing wrapper for MiniMax-M3 and DeepSeek via NVIDIA NIM API."""

import json
import re
from typing import Callable, Literal, Optional

from openai import OpenAI

from config import (
    DEEPSEEK_MODEL,
    MINIMAX_MODEL,
    NVIDIA_API_BASE_URL,
    NVIDIA_DEEPSEEK_API_KEY,
    NVIDIA_MINIMAX_API_KEY,
    SYSTEM_PROMPT,
)

RoutingMode = Literal["text", "scanned"]


def get_minimax_client() -> OpenAI:
    if not NVIDIA_MINIMAX_API_KEY:
        raise ValueError("NVIDIA_MINIMAX_API_KEY is not set. Add it to your .env or Streamlit Secrets.")
    return OpenAI(api_key=NVIDIA_MINIMAX_API_KEY, base_url=NVIDIA_API_BASE_URL)


def get_deepseek_client() -> OpenAI:
    if not NVIDIA_DEEPSEEK_API_KEY:
        raise ValueError("NVIDIA_DEEPSEEK_API_KEY is not set. Add it to your .env or Streamlit Secrets.")
    return OpenAI(api_key=NVIDIA_DEEPSEEK_API_KEY, base_url=NVIDIA_API_BASE_URL)


def extract_text_from_page_image(image_b64: str, page_num: int) -> str:
    """Extract text from a single scanned page using MiniMax-M3 vision."""
    client = get_minimax_client()
    response = client.chat.completions.create(
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
        max_tokens=8192,
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
    context = "\n\n---\n\n".join(context_chunks) if context_chunks else "No context available."
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

    if stream_callback:
        full_text = ""
        stream = client.chat.completions.create(
            model=MINIMAX_MODEL,
            messages=messages,
            max_tokens=8192,
            temperature=1.0,
            top_p=0.95,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            full_text += delta
            stream_callback(delta)
        return full_text

    response = client.chat.completions.create(
        model=MINIMAX_MODEL,
        messages=messages,
        max_tokens=8192,
        temperature=1.0,
        top_p=0.95,
    )
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
        "max_tokens": 16384,
        "extra_body": {"chat_template_kwargs": {"thinking": True, "reasoning_effort": "high"}},
    }

    if stream_callback:
        full_text = ""
        stream = client.chat.completions.create(**kwargs, stream=True)
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            full_text += delta
            stream_callback(delta)
        return full_text

    response = client.chat.completions.create(**kwargs, stream=False)
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
