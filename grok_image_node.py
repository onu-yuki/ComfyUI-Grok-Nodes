import os
import time
import random
import base64
import hashlib
from io import BytesIO
from email.utils import parsedate_to_datetime
from datetime import datetime, timezone

import requests
from PIL import Image
import numpy as np
import torch


# =========================
# 設定
# =========================
IMAGE_API_URL = "https://api.x.ai/v1/images/generations"
CHAT_API_URL = "https://api.x.ai/v1/chat/completions"

MIN_INTERVAL_SEC = 3.0
MAX_RETRIES = 3
MIN_WAIT_429 = 15.0
MAX_WAIT_429 = 120.0

# Chat(画像理解) 側の会話ID（任意）
VISION_CONV_ID = "lb-grok-vision-ref-001"


# =========================
# Helpers
# =========================
_RATE_LAST_CALL = 0.0


def _enforce_min_interval(min_interval_sec: float):
    global _RATE_LAST_CALL
    now = time.time()
    wait = (_RATE_LAST_CALL + min_interval_sec) - now
    if wait > 0:
        time.sleep(wait)
    _RATE_LAST_CALL = time.time()


def _is_credit_limit_error(text: str) -> bool:
    t = (text or "").lower()
    return ("used all available credits" in t) or ("monthly spending limit" in t)


def _parse_retry_after(ra: str, default_sec: float) -> float:
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


def _post_with_retry_429_heavy(url, headers, payload, timeout=60, max_retries=MAX_RETRIES):
    """
    429/5xx を中心にリトライ。
    - 429: クレジット/上限到達は即停止、それ以外はRetry-After優先 + backoff
    """
    for attempt in range(max_retries + 1):
        _enforce_min_interval(MIN_INTERVAL_SEC)

        r = requests.post(url, headers=headers, json=payload, timeout=timeout)

        if r.status_code < 400:
            return r

        if r.status_code == 429:
            if _is_credit_limit_error(r.text):
                raise RuntimeError(
                    "xAI API: credits exhausted or monthly spending limit reached.\n"
                    "Please purchase more credits or raise your spending limit.\n"
                    f"Response: {r.text}"
                )

            ra = r.headers.get("Retry-After")
            if ra:
                wait = _parse_retry_after(ra, MIN_WAIT_429)
            else:
                base = MIN_WAIT_429 * (2 ** attempt)
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


def pil_to_comfy_image(pil: Image.Image) -> torch.Tensor:
    arr = np.array(pil.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(arr)


def _comfy_image_to_png_bytes(image_tensor: torch.Tensor) -> bytes:
    """
    ComfyUI IMAGE(Tensor) -> PNG bytes
    想定: [H,W,C] or [N,H,W,C] (float32, 0..1)
    """
    if image_tensor is None:
        raise ValueError("reference_image is None")

    if isinstance(image_tensor, torch.Tensor) and image_tensor.ndim == 4:
        image_tensor = image_tensor[0]

    if not isinstance(image_tensor, torch.Tensor) or image_tensor.ndim != 3:
        raise ValueError("reference_image must be a ComfyUI IMAGE tensor with shape [H,W,C] or [N,H,W,C]")

    t = image_tensor.detach().cpu().clamp(0, 1)
    arr = (t.numpy() * 255.0).astype(np.uint8)

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


def _describe_reference_image(
    api_key: str,
    reference_image: torch.Tensor,
    vision_model: str,
    temperature: float = 0.2,
) -> str:
    """
    参照画像を Chat(画像理解) に渡して、画像生成promptに使える短い説明文を作る。
    """
    png_bytes = _comfy_image_to_png_bytes(reference_image)
    data_url = _png_bytes_to_data_url(png_bytes)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "x-grok-conv-id": VISION_CONV_ID,
    }

    system_prompt = (
        "You are an assistant that describes a reference image for image generation.\n"
        "Return a concise English description focusing on:\n"
        "- character appearance (hair, outfit, accessories)\n"
        "- background/environment\n"
        "- camera angle/composition\n"
        "- lighting/style\n"
        "No extra commentary."
    )

    user_text = (
        "Describe this image for use as a reference in an image generation prompt.\n"
        "Keep it concise (2-5 sentences)."
    )

    payload = {
        "model": vision_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ],
        "temperature": float(temperature),
    }

    r = _post_with_retry_429_heavy(CHAT_API_URL, headers=headers, payload=payload, timeout=60, max_retries=MAX_RETRIES)
    data = r.json()
    text = data["choices"][0]["message"]["content"]
    return (text or "").strip()


# =========================
# Node
# =========================
class GrokImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "tooltip": "生成したい内容を自然文で記述します。\n"
                               "（必要なら Avoid: もprompt内に書いてください）"
                }),
            },
            "optional": {
                "reference_image": ("IMAGE", {
                    "tooltip": "参照画像（任意）。\n"
                               "接続すると、この画像を Chat(画像理解) で説明文に変換し、promptへ追記してから生成します。\n"
                               "未接続の場合は prompt のみで生成します。"
                }),
                "vision_model": ("STRING", {
                    "default": "grok-4-1-fast-reasoning",
                    "tooltip": "参照画像を説明するための Chat/Vision 用モデルです。"
                }),
                "model": ("STRING", {
                    "default": "grok-2-image",
                    "tooltip": "画像生成に使用する Grok Image モデルです。"
                }),
                "n": ("INT", {"default": 1, "min": 1, "max": 10}),
                "response_format": (["url", "b64_json"], {"default": "b64_json"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("images", "info",)
    FUNCTION = "generate"
    CATEGORY = "LimitBreak/Grok"

    def generate(self, prompt, reference_image=None, vision_model="grok-4-1-fast-reasoning",
                 model="grok-2-image", n=1, response_format="b64_json"):
        api_key = os.getenv("XAI_API_KEY")
        if not api_key:
            raise ValueError("XAI_API_KEY is not set in environment variables")

        # --- optional reference image -> caption -> prompt append ---
        ref_info = ""
        if reference_image is not None:
            ref_desc = _describe_reference_image(
                api_key=api_key,
                reference_image=reference_image,
                vision_model=vision_model,
                temperature=0.2,
            )
            ref_hash = _digest_bytes(_comfy_image_to_png_bytes(reference_image))
            ref_info = f"[reference_image sha256={ref_hash}]\n{ref_desc}"

            prompt = (
                f"{prompt}\n\n"
                "Reference image (must match as closely as possible):\n"
                f"{ref_desc}\n\n"
                "Keep the same character, outfit, background, camera angle, and composition."
            )

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": model,
            "prompt": prompt,
            "n": int(n),
            "response_format": response_format,
        }

        try:
            print("Sending payload to xAI images/generations:")
            print(payload)
            r = _post_with_retry_429_heavy(
                IMAGE_API_URL,
                headers=headers,
                payload=payload,
                timeout=120,
                max_retries=MAX_RETRIES
            )
            data = r.json()

            images = []
            infos = []

            for idx, item in enumerate(data.get("data", [])):
                if response_format == "url":
                    url = item.get("url")
                    if not url:
                        raise ValueError(f"No url in response for image {idx}")
                    img_response = requests.get(url, timeout=60)
                    img_response.raise_for_status()
                    pil = Image.open(BytesIO(img_response.content))
                    info = f"Generated image {idx+1}: URL={url}"
                else:
                    b64 = item.get("b64_json")
                    if not b64:
                        raise ValueError(f"No b64_json in response for image {idx}")
                    img_bytes = base64.b64decode(b64)
                    pil = Image.open(BytesIO(img_bytes))
                    info = f"Generated image {idx+1}: base64 (b64_json)"

                img_tensor = pil_to_comfy_image(pil)
                images.append(img_tensor.unsqueeze(0))  # [1,H,W,C]
                infos.append(info)

            if not images:
                raise ValueError("No images generated in response")

            batch_images = torch.cat(images, dim=0)  # [N,H,W,C]

            extra = ""
            if ref_info:
                extra = "\n\n---\n" + ref_info

            return (batch_images, "\n".join(infos) + extra)

        except requests.exceptions.HTTPError as e:
            error_detail = r.text if 'r' in locals() else str(e)
            raise RuntimeError(f"API request failed: {e}\nResponse: {error_detail}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error in Grok image generation: {str(e)}")
