import base64
import io
import os
import requests
import numpy as np
from PIL import Image
import torch


def _comfy_image_to_png_bytes(image_tensor: torch.Tensor) -> bytes:
    """
    ComfyUI IMAGE tensor: [B,H,W,C] float 0..1
    """
    if image_tensor is None:
        raise ValueError("image_tensor is None")

    if image_tensor.dim() != 4 or image_tensor.size(-1) not in (3, 4):
        raise ValueError(f"Unexpected IMAGE tensor shape: {tuple(image_tensor.shape)}")

    img = image_tensor[0].detach().cpu().numpy()
    img = np.clip(img, 0.0, 1.0)
    img8 = (img * 255.0).round().astype(np.uint8)

    pil = Image.fromarray(img8, mode="RGBA" if img8.shape[-1] == 4 else "RGB")
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()


def _png_bytes_to_comfy_image(png_bytes: bytes) -> torch.Tensor:
    pil = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    arr = np.array(pil).astype(np.float32) / 255.0  # HWC
    t = torch.from_numpy(arr)[None, ...]  # 1,H,W,C
    return t


class GrokImagineImage:
    """
    Image-to-Image (Edit) node for xAI grok-imagine-image via /v1/images/edits
    If no image is connected, it falls back to /v1/images/generations (text-to-image).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "Swap the cat in the picture with a dog."}),
            },
            "optional": {
                "image": ("IMAGE",),
                "api_key": ("STRING", {"multiline": False, "default": os.getenv("XAI_API_KEY", "")}),
                "model": ("STRING", {"multiline": False, "default": "grok-imagine-image"}),
                "aspect_ratio": ("STRING", {"multiline": False, "default": "1:1"}),  # e.g. "4:3"
                "n": ("INT", {"default": 1, "min": 1, "max": 10}),
                "timeout_sec": ("INT", {"default": 120, "min": 10, "max": 600}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "LimitBreak/Grok"

    def generate(self, prompt, image=None, api_key="", model="grok-imagine-image", aspect_ratio="1:1", n=1, timeout_sec=120):
        if not api_key:
            raise ValueError("XAI API key is empty. Set api_key input or environment variable XAI_API_KEY.")

        headers = {
            "Authorization": f"Bearer {api_key}",
        }

        # Prefer base64 output so we can return IMAGE directly
        response_format = "b64_json"

        if image is not None:
            # ---- Image Edit (I2I) ----
            url = "https://api.x.ai/v1/images/edits"

            png_bytes = _comfy_image_to_png_bytes(image)

            data = {
                "model": model,
                "prompt": prompt,
                "response_format": response_format,
                "n": str(int(n)),
                "aspect_ratio": str(aspect_ratio),
            }

            files = {
                # OpenAI-compatible "image" form field
                "image": ("input.png", png_bytes, "image/png"),
            }

            r = requests.post(url, headers=headers, data=data, files=files, timeout=int(timeout_sec))
        else:
            # ---- Text to Image fallback ----
            url = "https://api.x.ai/v1/images/generations"
            payload = {
                "model": model,
                "prompt": prompt,
                "response_format": response_format,
                "n": int(n),
                "aspect_ratio": str(aspect_ratio),
            }
            r = requests.post(url, headers={**headers, "Content-Type": "application/json"}, json=payload, timeout=int(timeout_sec))

        if r.status_code >= 400:
            raise RuntimeError(f"xAI API error {r.status_code}: {r.text}")

        j = r.json()

        # Expected OpenAI-like shape: {"data":[{"b64_json":"..."}]}
        data_list = j.get("data") or []
        if not data_list:
            raise RuntimeError(f"Unexpected response (no data): {j}")

        # Return the first image
        b64 = data_list[0].get("b64_json")
        if not b64:
            raise RuntimeError(f"Unexpected response (no b64_json): {j}")

        out_bytes = base64.b64decode(b64)
        out_img = _png_bytes_to_comfy_image(out_bytes)
        return (out_img,)


NODE_CLASS_MAPPINGS = {
    "GrokImagineImage": GrokImagineImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GrokImagineImage": "Grok Imagine Image",
}
