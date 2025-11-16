import os
from typing import Dict, Generator, Iterable, List, Optional

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

    def extract_image_moment(self, chunk: str) -> Optional[str]:
        """
        Ask Grok to describe the single best visual moment to illustrate
        from a given story chunk, in one short, concrete sentence.
        """
        chunk = (chunk or "").strip()
        if not chunk:
            return None

        system_msg = (
            "You are a visual director for an illustrated storybook. "
            "Your job is to pick the single most cinematic visual moment "
            "from a story passage for an illustration."
        )
        user_msg = (
            "Given the story text below, describe ONE visual moment to illustrate "
            "in a single short sentence (under 120 characters). "
            "Use concrete visual language, mention key character(s) and setting. "
            "Do NOT ask questions. Do NOT include dialogue.\n\n"
            f"Story passage:\n{chunk}\n\n"
            "Illustration moment:"
        )

        try:
            response = self._client.chat.completions.create(
                model=self.text_model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                stream=False,
            )
        except Exception:
            return None

        text = (response.choices[0].message.content or "").strip()
        if not text:
            return None

        # Take only the first line and strip any leading labels like
        # "Illustration moment:" that the model might add.
        first_line = text.splitlines()[0].strip()
        lowered = first_line.lower()
        for prefix in ("illustration moment:", "illustration:", "image prompt:", "image:"):
            if lowered.startswith(prefix):
                first_line = first_line[len(prefix) :].lstrip(" -:")
                break

        if not first_line:
            first_line = text

        max_chars = 160
        if len(first_line) > max_chars:
            first_line = first_line[: max_chars - 3].rstrip(" .,;:") + "..."

        return first_line

    def extract_characters(self, story_text: str) -> Optional[Dict[str, str]]:
        """
        Ask Grok to identify the main characters and give each a short,
        visually focused description to use for consistent illustrations.

        Returns a dict mapping character name -> short visual description.
        """
        story_text = (story_text or "").strip()
        if not story_text:
            return None

        system_msg = (
            "You are a character designer for an illustrated story. "
            "You summarize characters with short visual descriptions "
            "that help artists draw them consistently."
        )
        user_msg = (
            "From the story text below, list up to 4 MAIN characters who are likely "
            "to appear repeatedly. For each character, write ONLY a short visual "
            "description (30–120 characters) focusing on age, species, build, "
            "hair/fur, clothing, distinctive items, overall vibe.\n\n"
            "Return your answer as valid JSON of the form:\n"
            "{\n"
            '  \"Elena\": \"lean woman in a black trench coat, short dark hair, tech goggles\",\n'
            '  \"Finn\": \"small russet fox with bright eyes and a torn blue scarf\"\n'
            "}\n\n"
            "Do NOT include any additional text outside the JSON.\n\n"
            f"Story text:\n{story_text}"
        )

        try:
            response = self._client.chat.completions.create(
                model=self.text_model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                stream=False,
            )
        except Exception:
            return None

        raw = (response.choices[0].message.content or "").strip()
        if not raw:
            return None

        import json  # local import to avoid forcing json on consumers

        try:
            data = json.loads(raw)
        except Exception:
            return None

        if not isinstance(data, dict):
            return None

        cleaned: Dict[str, str] = {}
        for name, desc in data.items():
            if not isinstance(name, str) or not isinstance(desc, str):
                continue
            name_clean = name.strip()
            desc_clean = desc.strip()
            if not name_clean or not desc_clean:
                continue
            # Conservative length cap on description
            max_len = 160
            if len(desc_clean) > max_len:
                desc_clean = desc_clean[: max_len - 3].rstrip(" .,;:") + "..."
            cleaned[name_clean] = desc_clean

        return cleaned or None

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
    scene_summary: Optional[str] = None,
    character_bible: Optional[Dict[str, str]] = None,
) -> str:
    """
    Turn a story chunk into a *short* image prompt.

    xAI's image endpoint has a maximum prompt length (e.g. 1024 chars), so we:
    - focus on the *end* of the chunk (most recent scene)
    - shape the prompt using shot type + tone + style presets
    - trim to a conservative character limit.
    """
    chunk = (chunk or "").strip()
    if not chunk and not scene_summary:
        return "A simple abstract illustration."

    # Prefer a compact scene summary if provided; otherwise derive from the text.
    if scene_summary:
        scene_text = scene_summary.strip()
    else:
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

    # Basic tone map, with a bit more drama for action-oriented scenes.
    tone_descriptions = {
        "neutral": "balanced mood, natural lighting",
        "cozy": "warm, comforting, soft lighting",
        "epic": "grand, cinematic, high-energy lighting",
        "spooky": "dark, high-contrast, eerie lighting",
        "whimsical": "playful, colorful, imaginative atmosphere",
        "kids": "bright, friendly, simple shapes, soft edges",
        "scifi": "futuristic, cool lighting, sleek technology",
        "horror": "ominous, high-contrast, unsettling details",
    }
    tone_phrase = tone_descriptions.get(tone, tone_descriptions["neutral"])

    # Heuristic tweak: if the scene text clearly describes a chase, battle,
    # or dangerous moment, lean into a more intense mood unless the user
    # explicitly chose a very soft tone.
    intense_keywords = ("chase", "pursuit", "run", "running", "battle", "fight", "explosion", "cliff", "edge", "gun", "rifle")
    if any(k in scene_text.lower() for k in intense_keywords):
        if tone in ("neutral", "scifi", "epic"):
            tone_phrase = "high-energy, tense atmosphere, dramatic lighting"

    style_descriptions = {
        "cinematic": "cinematic digital painting, realistic proportions",
        "storybook": "soft storybook illustration, painterly textures",
        "anime": "anime style, clean lines, expressive faces",
        "comic": "comic-book illustration, bold lines, strong contrast",
        "watercolor": "watercolor illustration, soft washes, subtle lines",
    }
    style_phrase = style_descriptions.get(visual_style, style_descriptions["cinematic"])

    # Optionally enrich with character descriptions when their names appear.
    char_phrase = ""
    if character_bible:
        lowered_scene = (scene_text or chunk).lower()
        snippets = []
        for name, desc in character_bible.items():
            if not name or not desc:
                continue
            if name.lower() in lowered_scene:
                snippets.append(f"{name}: {desc}")
            if len(snippets) >= 2:
                break
        # If none were detected in the scene text, fall back to the first
        # character in the bible so we at least keep the main hero present.
        if not snippets:
            name, desc = next(iter(character_bible.items()))
            snippets.append(f"{name}: {desc}")
        if snippets:
            joined = "; ".join(snippets)
            char_phrase = f" Character focus: {joined}."

    # Hard cap on scene text to avoid hitting the model's max prompt length.
    max_scene_chars = 300
    if len(scene_text) > max_scene_chars:
        scene_text = scene_text[-max_scene_chars:]

    base_prompt = (
        f"{shot_type} of the key moment in this scene: {scene_text}. "
        f"Mood: {tone_phrase}. Visual style: {style_phrase}. "
        "Highly detailed, cohesive character design."
        f"{char_phrase}"
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
        f"{char_phrase}"
    )


__all__ = ["GrokClient", "build_image_prompt_from_story"]


