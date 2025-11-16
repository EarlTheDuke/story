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

        # Primary path: streaming
        try:
            stream = self._client.chat.completions.create(
                model=self.text_model,
                messages=messages,
                stream=True,
            )

            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    yield delta.content
            return
        except Exception:
            # Some environments (or transient network issues) can cause
            # streaming to fail with protocol errors. Fall back to a
            # single non-streaming response so the app still works.
            response = self._client.chat.completions.create(
                model=self.text_model,
                messages=messages,
                stream=False,
            )
            content = response.choices[0].message.content or ""
            if content:
                yield content

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


def build_image_prompt_from_story(
    chunk: str,
    scene_index: int = 0,
    tone: str = "Neutral",
    visual_style: str = "Cinematic",
) -> str:
    """
    Turn a story chunk into a *short* image prompt.

    xAI's image endpoint has a maximum prompt length (e.g. 1024 chars), so we:
    - focus on the *end* of the chunk (most recent scene)
    - shape the prompt using shot type + tone + style presets
    - trim to a conservative character limit.
    """
    chunk = chunk.strip()
    if not chunk:
        return "A simple abstract illustration."

    # Naive sentence split; keeps the last 1–2 sentences.
    sentences = [s.strip() for s in chunk.replace("?", ".").split(".") if s.strip()]
    if not sentences:
        scene_text = chunk[-400:]
    else:
        scene_text = ". ".join(sentences[-2:])  # last two sentences

    # Choose a simple shot type based on how many scenes have come before.
    if scene_index <= 0:
        shot_type = "Wide establishing shot"
    elif scene_index <= 2:
        shot_type = "Medium shot"
    else:
        shot_type = "Dramatic close-up"

    # Map tone & style presets to compact descriptors.
    tone = (tone or "Neutral").lower()
    visual_style = (visual_style or "Cinematic").lower()

    tone_descriptions = {
        "neutral": "balanced mood, natural lighting",
        "cozy": "warm, comforting, soft lighting",
        "epic": "grand, cinematic scale, dramatic lighting",
        "spooky": "dark, high-contrast, eerie lighting",
        "whimsical": "playful, colorful, imaginative atmosphere",
        "kids": "bright, friendly, simple shapes, soft edges",
        "scifi": "futuristic, cool lighting, sleek technology",
        "horror": "ominous, high-contrast, unsettling details",
    }
    tone_phrase = tone_descriptions.get(tone, tone_descriptions["neutral"])

    style_descriptions = {
        "cinematic": "cinematic digital painting, realistic proportions",
        "storybook": "soft storybook illustration, painterly textures",
        "anime": "anime style, clean lines, expressive faces",
        "comic": "comic-book illustration, bold lines, strong contrast",
        "watercolor": "watercolor illustration, soft washes, subtle lines",
    }
    style_phrase = style_descriptions.get(visual_style, style_descriptions["cinematic"])

    # Hard cap on scene text to avoid hitting the model's max prompt length.
    max_scene_chars = 300
    if len(scene_text) > max_scene_chars:
        scene_text = scene_text[-max_scene_chars:]

    base_prompt = (
        f"{shot_type} of the key moment in this scene: {scene_text}. "
        f"Mood: {tone_phrase}. Visual style: {style_phrase}. "
        "Highly detailed, cohesive character design."
    )

    # Final safety cap.
    max_total_chars = 600
    if len(base_prompt) <= max_total_chars:
        return base_prompt

    # If it's still too long, trim the scene text further and rebuild once.
    overflow = len(base_prompt) - max_total_chars
    trimmed_scene = scene_text[:-overflow] if overflow < len(scene_text) else scene_text[: max_scene_chars // 2]
    trimmed_scene = trimmed_scene.rstrip(" .,;:") + "..."

    return (
        f"{shot_type} of the key moment in this scene: {trimmed_scene}. "
        f"Mood: {tone_phrase}. Visual style: {style_phrase}. "
        "Highly detailed, cohesive character design."
    )


__all__ = ["GrokClient", "build_image_prompt_from_story"]


