import os
from typing import Generator, Iterable, List, Optional

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


def _get_env(name: str, default: Optional[str] = None) -> str:
    value = os.getenv(name, default)
    if value is None:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


class GrokClient:
    """
    Thin wrapper around an OpenAI-compatible client configured for xAI Grok.

    This assumes:
    - XAI_API_KEY holds your xAI API key
    - XAI_BASE_URL is something like https://api.x.ai/v1
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        text_model: Optional[str] = None,
        image_model: Optional[str] = None,
    ) -> None:
        self.api_key = api_key or _get_env("XAI_API_KEY")
        self.base_url = base_url or _get_env("XAI_BASE_URL", "https://api.x.ai/v1")
        # Defaults align with your chosen models but can be overridden via env vars
        self.text_model = text_model or _get_env(
            "GROK_TEXT_MODEL", "grok-4-latest"
        )
        self.image_model = image_model or _get_env(
            "GROK_IMAGE_MODEL", "grok-2-image-1212"
        )

        self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def stream_story(
        self,
        system_prompt: str,
        user_prompt: str,
        history: Optional[List[dict]] = None,
    ) -> Generator[str, None, None]:
        """
        Stream a story continuation using Grok's chat completions API.

        Yields raw text chunks as they arrive.
        """
        messages: List[dict] = [{"role": "system", "content": system_prompt}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_prompt})

        stream = self._client.chat.completions.create(
            model=self.text_model,
            messages=messages,
            stream=True,
        )

        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content

    def generate_image(self, prompt: str) -> Optional[str]:
        """
        Generate an image URL from a text prompt using Grok's image endpoint.

        Returns:
            The URL of the generated image, or None if generation fails.
        """
        # xAI's grok-2-image-1212 currently does not accept a `size` argument,
        # so we only send the minimal required parameters.
        response = self._client.images.generate(
            model=self.image_model,
            prompt=prompt,
        )

        if not response.data:
            return None
        # OpenAI-compatible: response.data[0].url or .b64_json
        return response.data[0].url


def build_image_prompt_from_story(chunk: str) -> str:
    """
    Very simple heuristic to turn a story chunk into an image prompt.
    You can make this smarter later (LLM-based prompt extraction, etc.).
    """
    chunk = chunk.strip()
    return (
        "Highly detailed, cinematic illustration of the current scene from this story, "
        f"vibrant lighting, cohesive character design, digital art style:\n\n{chunk}"
    )


__all__ = ["GrokClient", "build_image_prompt_from_story"]


