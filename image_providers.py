from __future__ import annotations

import os
from enum import Enum
from typing import Optional, Union

from PIL import Image

from grok_client import GrokClient


class ImageBackend(str, Enum):
    GROK = "grok"
    STABLE_DIFFUSION = "stable_diffusion"


def _get_env(name: str, default: Optional[str] = None) -> str:
    value = os.getenv(name, default)
    if value is None:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


_sd_pipeline = None


def _get_sd_pipeline():
    """
    Lazily load a Stable Diffusion pipeline using diffusers.

    This requires a reasonably powerful machine; on CPU it will be slow
    but still functional for occasional images.
    """
    global _sd_pipeline
    if _sd_pipeline is not None:
        return _sd_pipeline

    from diffusers import StableDiffusionPipeline  # type: ignore
    import torch

    model_id = os.getenv("SD_MODEL_ID", "runwayml/stable-diffusion-v1-5")

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device)

    # For CPU-only environments, enable some offloading to reduce memory usage.
    if not torch.cuda.is_available():
        try:
            pipe.enable_sequential_cpu_offload()
        except Exception:
            # Not critical; just means higher memory usage.
            pass

    _sd_pipeline = pipe
    return _sd_pipeline


ImageResult = Union[str, Image.Image]


def generate_image_with_backend(
    prompt: str,
    backend: ImageBackend,
    grok_client: GrokClient,
) -> Optional[ImageResult]:
    """
    Generate an image using the selected backend.

    - For GROK: returns a URL string.
    - For STABLE_DIFFUSION: returns a PIL.Image.
    """
    if not prompt.strip():
        return None

    if backend == ImageBackend.STABLE_DIFFUSION:
        pipe = _get_sd_pipeline()
        result = pipe(prompt)
        if not result.images:
            return None
        return result.images[0]

    # Default: Grok image model via the existing client.
    return grok_client.generate_image(prompt)


__all__ = ["ImageBackend", "generate_image_with_backend"]


