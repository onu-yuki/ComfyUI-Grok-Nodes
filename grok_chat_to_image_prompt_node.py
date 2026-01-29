import os
import json
import re
import time
import random
import hashlib
from pathlib import Path
from threading import Lock
from email.utils import parsedate_to_datetime
from datetime import datetime, timezone
from io import BytesIO
import base64

import requests

# optional image support
import numpy as np
import torch
from PIL import Image


# =========================
# 設定
# =========================
API_URL = "https://api.x.ai/v1/chat/completions"
MIN_INTERVAL_SEC = 3.0
MAX_RETRIES = 3
MIN_WAIT_429 = 15.0
MAX_WAIT_429 = 120.0
CONV_ID = "lb-grok-chat-single-image-001"

CACHE_DIR = Path("custom_nodes/ComfyUI-LB-Grok/.cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

_RATE_LOCK = Lock()
_LAST_CALL = 0.0


# =========================
# Utils
# =========================
def enforce_min_interval(min_interval_sec: float):
    global _LAST_CALL
    with _RATE_LOCK:
        now = time.time()
        wait = (_LAST_CALL + min_interval_sec) - now
        if wait > 0:
            time.sleep(wait)
        _LAST_CALL = time.time()


def make_cache_key_from_parts(payload_no_image: dict, image_digest: str) -> str:
    """
    画像base64をキャッシュキーに直接入れると巨大になるので、
    画像はsha256 digestだけで識別する。
    """
    s = json.dumps(payload_no_image, sort_keys=True, ensure_ascii=False)
    h = hashlib.sha256()
    h.update(s.encode("utf-8"))
    h.update(b"|img|")
    h.update((image_digest or "").encode("utf-8"))
    return h.hexdigest()


def cache_get(key: str):
    p = CACHE_DIR / f"{key}.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return None


def cache_set(key: str, value: dict):
    p = CACHE_DIR / f"{key}.json"
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def is_credit_limit_error(text: str) -> bool:
    t = (text or "").lower()
    return ("used all available credits" in t) or ("monthly spending limit" in t)


def parse_retry_after(ra: str, default_sec: float) -> float:
    if not ra:
        return default_sec
    ra = ra.strip()

    # seconds
    try:
        return max(0.0, float(ra))
    except ValueError:
        pass

    # HTTP-date
    try:
        dt = parsedate_to_datetime(ra)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return max(0.0, (dt - datetime.now(timezone.utc)).total_seconds())
    except Exception:
        return default_sec


def post_with_retry_429_heavy(url, headers, payload, timeout=60, max_retries=MAX_RETRIES):
    for attempt in range(max_retries + 1):
        enforce_min_interval(MIN_INTERVAL_SEC)

        r = requests.post(url, headers=headers, json=payload, timeout=timeout)

        if r.status_code < 400:
            return r

        if r.status_code == 429:
            # クレジット/上限到達は即停止
            if is_credit_limit_error(r.text):
                raise RuntimeError(
                    "xAI API: credits exhausted or monthly spending limit reached.\n"
                    "Please purchase more credits or raise your spending limit.\n"
                    f"Response: {r.text}"
                )

            ra = r.headers.get("Retry-After")
            if ra:
                wait = parse_retry_after(ra, MIN_WAIT_429)
            else:
                base = MIN_WAIT_429 * (2 ** attempt)  # 15, 30, 60, 120...
                wait = min(MAX_WAIT_429, base) + random.uniform(0.0, 1.0)

            if attempt == max_retries:
                r.raise_for_status()

            time.sleep(wait)
            continue

        if 500 <= r.status_code <= 599:
            wait = min(60.0, (2 ** attempt)) + random.uniform(0.0, 0.5)
            if attempt == max_retries:
                r.raise_for_status()
            time.sleep(wait)
            continue

        r.raise_for_status()

    raise RuntimeError("Retries exhausted unexpectedly")


def _extract_json(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    m = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if m:
        return json.loads(m.group(1))

    m = re.search(r"(\{.*\})", text, flags=re.DOTALL)
    if m:
        return json.loads(m.group(1))

    raise ValueError("Failed to extract JSON from model output")


def _comfy_image_to_png_bytes(image_tensor: torch.Tensor) -> bytes:
    """
    ComfyUI IMAGE(Tensor) -> PNG bytes
    想定: [H,W,C] or [N,H,W,C] (float32, 0..1)
    """
    if image_tensor is None:
        raise ValueError("reference_image is None")

    if isinstance(image_tensor, torch.Tensor) and image_tensor.ndim == 4:
        # take first image in batch
        image_tensor = image_tensor[0]

    if not isinstance(image_tensor, torch.Tensor) or image_tensor.ndim != 3:
        raise ValueError("reference_image must be a ComfyUI IMAGE tensor with shape [H,W,C] or [N,H,W,C]")

    t = image_tensor.detach().cpu().clamp(0, 1)
    arr = (t.numpy() * 255.0).astype(np.uint8)

    # ensure RGB
    if arr.shape[-1] == 4:
        arr = arr[:, :, :3]
    pil = Image.fromarray(arr, mode="RGB")

    buf = BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()


def _png_bytes_to_data_url(png_bytes: bytes) -> str:
    b64 = base64.b64encode(png_bytes).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def _digest_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# =========================
# Node
# =========================
# =========================
# Node
# =========================
class GrokChatToImagePrompt:
    '''
    Grok Chat を使って、単発画像生成用の prompt を1本だけ生成するノード（シンプル版）

    - 入力は Chat 1本だけ（ユーザーフレンドリー）
    - 'Avoid:' / 'Negative:' のようなラベル表現は使わず、自然文で失敗を回避する
    '''

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "chat": (
                    "STRING",
                    {
                        "multiline": True,
                        "tooltip": "チャット感覚で入力してください。\n"
                                   "例：白い服を着た少女が部屋の中で静かに立っている。アニメ調で。"
                    }
                ),
            },
            "optional": {
                "reference_image": (
                    "IMAGE",
                    {
                        "tooltip": "参照画像（任意）。\n"
                                   "接続すると、この画像を見た上で単発プロンプトを生成します。\n"
                                   "未接続の場合はテキストのみで生成します。"
                    }
                ),
                # "temperature": (
                #     "FLOAT",
                #     {
                #         "default": 0.3,
                #         "min": 0.0,
                #         "max": 1.5,
                #         "tooltip": "どれくらい冒険するか\n"
                #                    "低い → かしこい・堅い・毎回似た答え\n"
                #                    "高い → 発想豊か・ブレる・脱線しやすい\n"
                #                    "低めにすると出力が安定します（推奨 0.2〜0.3）。"
                #     }
                # ),
                "model": (
                    "STRING",
                    {
                        "default": "grok-4-1-fast-reasoning",
                        "tooltip": "プロンプト生成に使用する Grok モデルです。"
                    }
                ),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("image_prompt", "debug_json")
    FUNCTION = "generate"
    CATEGORY = "LimitBreak/Grok"

    def generate(
        self,
        chat,
        reference_image=None,
        # temperature=0.3,
        model="grok-4-1-fast-reasoning"
    ):
        api_key = os.getenv("XAI_API_KEY")
        if not api_key:
            raise ValueError("XAI_API_KEY is not set")

        system_prompt = (
            "You are a prompt engineer for an image generation model.\n"
            "Return ONLY valid JSON. No extra text.\n"
            "Make ONE single-image prompt suitable for an image generator that has only one 'prompt' field.\n"
            "Write a concise but specific prompt in English.\n"
            "Do NOT use labels like 'Avoid:' or 'Negative:' or any 'key: value' blocks.\n"
            "Instead, include one short natural-language safety sentence at the end, e.g. "
            "'No text, no watermark, no extra fingers, no deformed anatomy.'\n"
            "JSON schema:\n"
            "{\n"
            "  \"image_prompt\": \"string\"\n"
            "}\n"
        )

        user_prompt = (
            f"Chat:\n{chat}\n\n"
            "Base requirements (always apply):\n"
            "- single image, single scene\n"
            "- coherent composition\n"
            "- prevent common artifacts (extra fingers, bad anatomy, text, watermark) WITHOUT using labels\n\n"
            "Now output ONLY the JSON."
        )

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "x-grok-conv-id": CONV_ID,
        }

        # --- optional image handling ---
        image_digest = ""
        if reference_image is not None:
            png_bytes = _comfy_image_to_png_bytes(reference_image)
            image_digest = _digest_bytes(png_bytes)
            image_data_url = _png_bytes_to_data_url(png_bytes)

            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": image_data_url}},
                    ],
                },
            ]
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

        payload = {
            "model": model,
            "messages": messages,
            # "temperature": float(temperature),
        }

        # ---- 永続キャッシュ（画像のdigestだけで識別） ----
        payload_no_image_for_key = {
            "model": model,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            # "temperature": float(temperature),
            "has_image": reference_image is not None,
        }
        key = make_cache_key_from_parts(payload_no_image_for_key, image_digest)

        hit = cache_get(key)
        if hit:
            image_prompt = (hit.get("image_prompt") or "").strip()
            return (image_prompt, json.dumps(hit, ensure_ascii=False, indent=2))

        r = post_with_retry_429_heavy(
            API_URL,
            headers=headers,
            payload=payload,
            timeout=60,
            max_retries=MAX_RETRIES,
        )

        data = r.json()
        content = data["choices"][0]["message"]["content"]
        j = _extract_json(content)

        image_prompt = (j.get("image_prompt") or "").strip()
        if not image_prompt:
            raise ValueError(f"Missing image_prompt in model output. raw={content}")

        cache_set(key, j)
        return (image_prompt, json.dumps(j, ensure_ascii=False, indent=2))
