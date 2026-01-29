import os
import time
import json
import random
from pathlib import Path
from threading import Lock
from io import BytesIO
import base64

import requests
import numpy as np
import torch
from PIL import Image

# import folder_paths


# =========================
# xAI endpoints
# =========================
BASE_URL = "https://api.x.ai"
VIDEO_GENERATIONS_URL = f"{BASE_URL}/v1/videos/generations"
VIDEO_RESULT_URL = f"{BASE_URL}/v1/videos"  # + /{request_id}

# Polling
MAX_POLL_ATTEMPTS = 120  # 120 * 4s = 8min by default

# Rate limiting (client-side)
MIN_INTERVAL_SEC = 1.0
_RATE_LOCK = Lock()
_LAST_CALL = 0.0


def _enforce_min_interval(min_interval_sec: float):
    global _LAST_CALL
    with _RATE_LOCK:
        now = time.time()
        wait = (_LAST_CALL + min_interval_sec) - now
        if wait > 0:
            time.sleep(wait)
        _LAST_CALL = time.time()


def _post_with_retry(url: str, headers: dict, payload: dict, timeout: int = 180, max_retries: int = 3):
    """
    Light retry for 429 / 5xx.
    For 422, raise with body (most useful for payload validation).
    """
    for attempt in range(max_retries + 1):
        _enforce_min_interval(MIN_INTERVAL_SEC)
        r = requests.post(url, headers=headers, json=payload, timeout=timeout)

        if r.status_code < 400:
            return r

        if r.status_code == 422:
            raise RuntimeError(f"422 Unprocessable Entity\nurl={url}\nresponse={r.text}")

        if r.status_code == 429:
            ra = r.headers.get("Retry-After")
            if ra:
                try:
                    wait = float(ra)
                except ValueError:
                    wait = 10.0
            else:
                wait = min(60.0, 5.0 * (2 ** attempt)) + random.uniform(0.0, 1.0)

            if attempt == max_retries:
                r.raise_for_status()

            time.sleep(wait)
            continue

        if 500 <= r.status_code <= 599:
            wait = min(30.0, (2 ** attempt)) + random.uniform(0.0, 0.5)
            if attempt == max_retries:
                r.raise_for_status()
            time.sleep(wait)
            continue

        r.raise_for_status()

    raise RuntimeError("Retries exhausted unexpectedly")


def _get_with_retry(url: str, headers: dict, timeout: int = 60, max_retries: int = 3):
    for attempt in range(max_retries + 1):
        _enforce_min_interval(MIN_INTERVAL_SEC)
        r = requests.get(url, headers=headers, timeout=timeout)
        if r.status_code < 400:
            return r
        if r.status_code == 429 or (500 <= r.status_code <= 599):
            wait = min(30.0, (2 ** attempt)) + random.uniform(0.0, 0.5)
            if attempt == max_retries:
                r.raise_for_status()
            time.sleep(wait)
            continue
        r.raise_for_status()
    raise RuntimeError("Retries exhausted unexpectedly")


def _comfy_image_to_png_data_url(image_tensor: torch.Tensor) -> str:
    """
    ComfyUI IMAGE tensor: [H,W,C] or [N,H,W,C] float(0..1) -> data:image/png;base64,...
    """
    if image_tensor is None:
        raise ValueError("reference_image is None")

    if isinstance(image_tensor, torch.Tensor) and image_tensor.ndim == 4:
        image_tensor = image_tensor[0]

    if not isinstance(image_tensor, torch.Tensor) or image_tensor.ndim != 3:
        raise ValueError("reference_image must be IMAGE tensor [H,W,C] or [N,H,W,C]")

    t = image_tensor.detach().cpu().clamp(0, 1)
    arr = (t.numpy() * 255.0).astype(np.uint8)

    if arr.shape[-1] == 4:
        arr = arr[:, :, :3]

    pil = Image.fromarray(arr, mode="RGB")
    buf = BytesIO()
    pil.save(buf, format="PNG")

    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def _extract_request_id(resp_json: dict) -> str:
    for k in ("request_id", "id"):
        v = resp_json.get(k)
        if v:
            return str(v)
    raise ValueError(f"Could not find request id in response: {resp_json}")


def _extract_status(result_json: dict) -> str:
    return str(result_json.get("status") or result_json.get("state") or "").lower().strip()


def _extract_video_url(result_json: dict) -> str:
    """
    Common patterns:
      { video: { url: ... } }
      { response: { video: { url: ... } } }
      { video_url: ... }
    """
    vid = result_json.get("video")
    if isinstance(vid, dict) and isinstance(vid.get("url"), str) and vid["url"]:
        return vid["url"]

    if isinstance(result_json.get("video_url"), str) and result_json["video_url"]:
        return result_json["video_url"]

    resp = result_json.get("response")
    if isinstance(resp, dict):
        vid = resp.get("video")
        if isinstance(vid, dict) and isinstance(vid.get("url"), str) and vid["url"]:
            return vid["url"]

    out = result_json.get("output")
    if isinstance(out, dict):
        for k in ("url", "video_url"):
            if isinstance(out.get(k), str) and out.get(k):
                return out[k]

    data = result_json.get("data")
    if isinstance(data, list) and data:
        item = data[0]
        if isinstance(item, dict):
            for k in ("url", "video_url"):
                if isinstance(item.get(k), str) and item.get(k):
                    return item[k]

    raise ValueError(f"Could not find video url in result: {result_json}")


def _stream_download_to_path(url: str, path: Path, timeout: int = 300, chunk_size: int = 1024 * 1024):
    """
    Stream download to avoid holding the whole mp4 in memory.
    """
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)


class GrokImagineVideo:
    """
    Grok Imagine Video:
      prompt (+ optional reference image) -> generate mp4 -> save to ComfyUI output directory

    Outputs:
      - mp4_path (STRING)
      - info (STRING)

    Preview tip:
      Connect mp4_path -> VideoHelperSuite "Load Video (Path)" to get an animated preview in the UI.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "model": ("STRING", {"default": "grok-imagine-video"}),
            },
            "optional": {
                "reference_image": ("IMAGE", {"tooltip": "Optional. If connected, it will be sent as image conditioning."}),
                "duration_sec": ("INT", {"default": 6, "min": 1, "max": 30, "tooltip": "Requested duration (seconds)."}),
                "poll_interval_sec": ("FLOAT", {"default": 4.0, "min": 1.0, "max": 30.0}),
                "filename_prefix": ("STRING", {"default": "grok_video", "tooltip": "Output filename prefix (no extension)."}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("mp4_path",)
    FUNCTION = "generate"
    CATEGORY = "LimitBreak/Grok"

    def generate(self, prompt, model, reference_image=None, duration_sec=6, poll_interval_sec=4.0, filename_prefix="grok_video"):
        api_key = os.getenv("XAI_API_KEY")
        if not api_key:
            raise ValueError("XAI_API_KEY is not set")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": model,
            "prompt": prompt,
            "duration": int(duration_sec),
        }

        # Optional image: keep endpoint as /videos/generations, only add payload field.
        if reference_image is not None:
            image_data_url = _comfy_image_to_png_data_url(reference_image)
            payload["image"] = {"url": image_data_url}

        # Submit generation
        r = _post_with_retry(VIDEO_GENERATIONS_URL, headers=headers, payload=payload, timeout=180, max_retries=3)
        submit_json = r.json()
        request_id = _extract_request_id(submit_json)

        # Poll result; IMPORTANT: some responses don't include status, but include video.url directly.
        last_json = None
        for _ in range(MAX_POLL_ATTEMPTS):
            time.sleep(float(poll_interval_sec))
            rs = _get_with_retry(f"{VIDEO_RESULT_URL}/{request_id}", headers=headers, timeout=60, max_retries=3)
            last_json = rs.json()

            # If mp4 url is present, treat as done even if status is missing
            video_url = None
            try:
                video_url = _extract_video_url(last_json)
            except Exception:
                video_url = None

            status = _extract_status(last_json)
            if video_url and (not status or status in ("done", "completed", "succeeded", "success")):
                # Save mp4 under output/
                out_dir = Path("output") / "grok_videos"
                out_dir.mkdir(parents=True, exist_ok=True)

                safe_prefix = "".join(c for c in (filename_prefix or "grok_video") if c.isalnum() or c in ("-", "_"))
                if not safe_prefix:
                    safe_prefix = "grok_video"

                mp4_path = out_dir / f"{safe_prefix}_{request_id}.mp4"
                _stream_download_to_path(video_url, mp4_path)

                info = json.dumps(
                    {
                        "request_id": request_id,
                        "status": status,
                        "video_url": video_url,
                        "saved_mp4": str(mp4_path),
                        "duration_requested": int(duration_sec),
                        "duration_reported": (last_json.get("video") or {}).get("duration") if isinstance(last_json.get("video"), dict) else None,
                        "preview_hint": "Use VideoHelperSuite 'Load Video (Path)' to preview this mp4 in the UI.",
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                return (str(mp4_path),)

            if status in ("failed", "error", "canceled", "cancelled"):
                raise RuntimeError(
                    "Video generation failed.\n"
                    f"status={status}\nraw={json.dumps(last_json, ensure_ascii=False)}"
                )

        raise TimeoutError(
            "Video generation timed out.\n"
            f"request_id={request_id}\nlast={json.dumps(last_json, ensure_ascii=False) if last_json else None}"
        )


NODE_CLASS_MAPPINGS = {
    "GrokImagineVideo": GrokImagineVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GrokImagineVideo": "Grok Imagine Video",
}
