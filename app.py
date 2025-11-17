import json
import os

import streamlit as st
import streamlit.components.v1 as components

from grok_client import GrokClient, build_image_prompt_from_story
from image_providers import ImageBackend, generate_image_with_backend


st.set_page_config(page_title="Real-Time AI Storyteller", layout="wide")


@st.cache_resource
def get_grok_client(api_key: str) -> GrokClient:
    """
    Cache the Grok client, keyed by API key so we recreate it
    if the user changes keys in the UI.
    """
    return GrokClient(api_key=api_key)


STORY_SYSTEM_PROMPT = """
You are an imaginative, cinematic storyteller.
Write in vivid, visual language that is easy to turn into illustrations.

Rules:
- Continue the story in 120–200 words per response.
- End each response with a small cliffhanger or open question.
- Keep the narration focused on concrete scenes, characters, and actions.
"""

FACT_SYSTEM_PROMPT = """
You are a knowledgeable, clear explainer.
Give accurate, concise answers about real history, science, technology, and events.
Use paragraphs, not bullet points. Do NOT invent fictional facts.
If you are unsure, say what is and is not known.
"""


def init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "character_bible" not in st.session_state:
        st.session_state.character_bible = {}


def speak_text_with_browser_tts(text: str) -> None:
    """
    Trigger client-side text-to-speech using the browser's Web Speech API.

    This keeps things simple (no extra API keys) and works both locally
    and on Streamlit Cloud, as long as the user's browser supports TTS.
    """
    text = text.strip()
    if not text:
        return

    safe_text = json.dumps(text)
    components.html(
        f"""
        <script>
        const text = {safe_text};
        if ("speechSynthesis" in window) {{
            const utterance = new SpeechSynthesisUtterance(text);
            window.speechSynthesis.cancel();
            window.speechSynthesis.speak(utterance);
        }}
        </script>
        """,
        height=0,
    )


def _preset_api_key() -> str:
    """
    Resolve an initial API key from (highest to lowest priority):
    - session state (so it survives reruns)
    - Streamlit secrets (Streamlit Cloud or local .streamlit/secrets.toml)
    - environment variables (incl. .env if loaded)
    """
    # Session state takes priority
    saved = st.session_state.get("xai_api_key")
    if saved:
        return saved

    # Streamlit secrets (does not exist outside Streamlit runtime)
    try:
        if "XAI_API_KEY" in st.secrets:
            return str(st.secrets["XAI_API_KEY"])
    except Exception:
        # st.secrets may not be available in some contexts
        pass

    # Environment variable as final fallback
    return os.getenv("XAI_API_KEY", "")


def ask_for_api_key() -> str | None:
    """
    Resolve the xAI API key, prefilling from secrets/env when available,
    and letting the user override it via the sidebar.
    """
    preset = _preset_api_key()

    api_key = st.sidebar.text_input(
        "xAI API key",
        value=preset,
        type="password",
        help=(
            "If running on Streamlit Cloud, you can set XAI_API_KEY in app secrets "
            "so you don't have to paste it here each time."
        ),
    ).strip()

    # If the user doesn't change the field, keep the preset value.
    resolved = api_key or preset
    st.session_state["xai_api_key"] = resolved
    return resolved or None


def main() -> None:
    init_session_state()

    api_key = ask_for_api_key()
    if not api_key:
        st.warning("Please paste your xAI API key in the left sidebar to begin.")
        return

    client = get_grok_client(api_key)

    # Mode selection: story vs. real-world factual answers
    st.sidebar.markdown("### Mode")
    mode = st.sidebar.radio(
        "Response type",
        ["Story", "Real-world answer"],
        index=0,
        help="Choose between imaginative storytelling or factual, real-world explanations.",
    )
    is_story_mode = mode == "Story"
    system_prompt = STORY_SYSTEM_PROMPT if is_story_mode else FACT_SYSTEM_PROMPT

    # Sidebar controls for prompt behavior and style
    st.sidebar.markdown("### Image Prompt Settings")
    use_advanced_prompts = st.sidebar.checkbox(
        "Use advanced image prompts (extra Grok calls)",
        value=True,
        help="When enabled, uses extra Grok calls to pick key moments, beats, and emotions for each scene.",
    )
    keep_characters_consistent = st.sidebar.checkbox(
        "Keep characters visually consistent",
        value=True,
        help="When enabled, builds a character bible and injects short visual descriptions into image prompts.",
    )
    image_backend_label = st.sidebar.selectbox(
        "Image backend",
        [
            "Grok (xAI API)",
            "Local SDXL Turbo (fast)",
            "Local SDXL Base (high quality)",
        ],
        index=0,
        help="Choose which image generator to use.",
    )
    sd_model_id: Optional[str] = None
    if "SDXL" in image_backend_label:
        image_backend = ImageBackend.STABLE_DIFFUSION
        if "Base" in image_backend_label:
            sd_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        else:
            sd_model_id = "stabilityai/sdxl-turbo"
    else:
        image_backend = ImageBackend.GROK

    st.sidebar.markdown("### Story & Image Style")
    story_tone = st.sidebar.selectbox(
        "Story tone",
        ["Neutral", "Cozy", "Epic", "Spooky", "Whimsical", "Kids", "SciFi", "Horror"],
        index=0,
    )
    image_style = st.sidebar.selectbox(
        "Image style",
        ["Cinematic", "Photorealistic", "Diagram", "Storybook", "Anime", "Comic", "Watercolor"],
        index=0,
    )
    show_image_prompts = st.sidebar.checkbox(
        "Show image prompts (debug)",
        value=False,
        help="Display the exact prompt sent to the image model for each scene.",
    )
    st.sidebar.markdown("### Custom image prompt (optional)")
    custom_prompt_default = st.session_state.get("custom_image_prompt", "")
    custom_prompt = st.sidebar.text_area(
        "Custom image prompt for next scene (optional)",
        value=custom_prompt_default,
        height=80,
    )
    existing_mode = st.session_state.get("custom_image_prompt_mode", "off")
    mode_to_label = {
        "off": "Off",
        "enhance": "Enhance auto prompt",
        "replace": "Replace auto prompt",
    }
    label_to_mode = {v: k for k, v in mode_to_label.items()}
    custom_mode_label = st.sidebar.radio(
        "How to use the custom prompt",
        ["Off", "Enhance auto prompt", "Replace auto prompt"],
        index=["Off", "Enhance auto prompt", "Replace auto prompt"].index(
            mode_to_label.get(existing_mode, "Off")
        ),
    )
    custom_mode = label_to_mode[custom_mode_label]
    st.session_state["story_tone"] = story_tone
    st.session_state["image_style"] = image_style
    st.session_state["show_image_prompts"] = show_image_prompts
    st.session_state["use_advanced_prompts"] = use_advanced_prompts
    st.session_state["keep_characters_consistent"] = keep_characters_consistent
    st.session_state["image_backend"] = image_backend
    st.session_state["sd_model_id"] = sd_model_id
    st.session_state["custom_image_prompt"] = custom_prompt
    st.session_state["custom_image_prompt_mode"] = custom_mode

    # Sidebar view of current character bible
    with st.sidebar.expander("Character bible (auto)", expanded=False):
        if keep_characters_consistent:
            if st.session_state.character_bible:
                for name, desc in st.session_state.character_bible.items():
                    st.markdown(f"- **{name}**: {desc}")
            else:
                st.caption("No characters extracted yet. They will appear after the first scene.")
        else:
            st.caption("Character consistency is turned off.")

    if is_story_mode:
        st.title("🎭 Real-Time AI Storyteller with Grok")
        st.markdown(
            "Enter a story idea or continuation. Grok will stream the story while generating images for each scene."
        )
    else:
        st.title("📘 Real-Time Illustrated Answers with Grok")
        st.markdown(
            "Ask a question about real history, science, or anything factual. "
            "Grok will answer while generating an illustration that helps explain it."
        )

    # Show history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if img := msg.get("image"):
                st.image(img, use_container_width=True)
            if (
                show_image_prompts
                and msg.get("role") == "assistant"
                and (msg.get("image_prompt") or msg.get("final_image_prompt"))
            ):
                auto_prompt = msg.get("image_prompt")
                final_prompt = msg.get("final_image_prompt") or auto_prompt
                if not final_prompt:
                    continue
                with st.expander("Image prompt for this scene", expanded=False):
                    if image_moment := msg.get("image_moment"):
                        st.markdown(f"**Key visual moment (from Grok):** {image_moment}")
                        st.markdown("---")
                    if beat_type := msg.get("beat_type"):
                        st.markdown(f"**Beat type:** `{beat_type}`")
                    if emotion := msg.get("emotion_word"):
                        st.markdown(f"**Emotion:** `{emotion}`")
                    if beat_type or emotion:
                        st.markdown("---")
                    if auto_prompt and auto_prompt != final_prompt:
                        st.markdown("**Auto-generated prompt:**")
                        st.markdown(auto_prompt)
                        st.markdown("---")
                    st.markdown("**Final image prompt sent to model:**")
                    st.markdown(final_prompt)

    prompt = st.chat_input("Start a new story or continue the current one...")
    if not prompt:
        return

    # Record user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Prepare history for the model (assistant + user messages only)
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages
        if m["role"] in ("user", "assistant")
    ]
    # Determine which numbered scene this chunk represents
    scene_index = sum(1 for m in st.session_state.messages if m["role"] == "assistant")

    # Stream Grok's story response
    with st.chat_message("assistant"):
        text_placeholder = st.empty()
        full_story_chunk = ""

        for text_piece in client.stream_story(
            system_prompt=system_prompt,
            user_prompt=prompt,
            history=history,
        ):
            full_story_chunk += text_piece
            text_placeholder.markdown(full_story_chunk)

        # Once the chunk is complete, trigger client-side TTS.
        speak_text_with_browser_tts(full_story_chunk)

        # Update character bible once we have at least one full scene.
        if (
            keep_characters_consistent
            and full_story_chunk.strip()
            and not st.session_state.character_bible
        ):
            # Use all assistant content so far (including this chunk) as context.
            story_so_far = "\n\n".join(
                m["content"]
                for m in st.session_state.messages
                if m["role"] == "assistant"
            ) + f"\n\n{full_story_chunk}"
            try:
                characters = client.extract_characters(story_so_far)
                if characters:
                    st.session_state.character_bible.update(characters)
            except Exception as e:  # noqa: BLE001
                if show_image_prompts:
                    st.warning(f"Character extraction failed: {e}")

        # After streaming finishes, generate an image for this chunk
        img_obj = None
        image_prompt = None
        final_image_prompt = None
        image_moment = None
        beat_type = None
        emotion_word = None
        if full_story_chunk.strip():
            # Ask Grok for beat type & emotion, then a visual moment
            # to illustrate when advanced prompts are enabled.
            if use_advanced_prompts:
                try:
                    beat_info = client.extract_beat_and_emotion(full_story_chunk)
                except Exception as e:  # noqa: BLE001
                    if show_image_prompts:
                        st.warning(f"Beat/emotion extraction failed: {e}")
                    beat_info = None

                if beat_info:
                    beat_type = beat_info.get("beat")
                    emotion_word = beat_info.get("emotion")

                try:
                    image_moment = client.extract_image_moment(
                        full_story_chunk,
                        beat_type=beat_type,
                        mode="story" if is_story_mode else "real",
                    )
                except Exception as e:  # noqa: BLE001
                    if show_image_prompts:
                        st.warning(f"Image moment extraction failed: {e}")
                    image_moment = None

            image_prompt = build_image_prompt_from_story(
                full_story_chunk,
                scene_index=scene_index,
                tone=story_tone,
                visual_style=image_style,
                scene_summary=image_moment,
                character_bible=st.session_state.character_bible
                if keep_characters_consistent
                else None,
                beat_type=beat_type,
                emotion_word=emotion_word,
            )
            # Apply custom prompt override/enhancement if provided.
            custom_prompt = st.session_state.get("custom_image_prompt", "").strip()
            custom_mode = st.session_state.get("custom_image_prompt_mode", "off")
            final_image_prompt = image_prompt
            if custom_prompt:
                if custom_mode == "replace":
                    final_image_prompt = custom_prompt
                elif custom_mode == "enhance":
                    final_image_prompt = f"{image_prompt} Extra details: {custom_prompt}"

            # Reset one-shot custom prompt after use.
            st.session_state["custom_image_prompt"] = ""
            st.session_state["custom_image_prompt_mode"] = "off"

            try:
                img_obj = generate_image_with_backend(
                    final_image_prompt or image_prompt,
                    backend=image_backend,
                    grok_client=client,
                    sd_model_id=sd_model_id,
                )
            except Exception as e:  # noqa: BLE001
                st.error(f"Image generation failed: {e}")
                img_obj = None

            if img_obj is not None:
                st.image(img_obj, caption="Scene Illustration", use_container_width=True)
            else:
                st.info("No image was generated for this scene.")

            if show_image_prompts and (final_image_prompt or image_prompt):
                with st.expander("Image prompt for this scene", expanded=False):
                    if image_moment:
                        st.markdown(f"**Key visual moment (from Grok):** {image_moment}")
                        st.markdown("---")
                    if beat_type:
                        st.markdown(f"**Beat type:** `{beat_type}`")
                    if emotion_word:
                        st.markdown(f"**Emotion:** `{emotion_word}`")
                    if beat_type or emotion_word:
                        st.markdown("---")
                    if image_prompt and final_image_prompt and image_prompt != final_image_prompt:
                        st.markdown("**Auto-generated prompt:**")
                        st.markdown(image_prompt)
                        st.markdown("---")
                    st.markdown("**Final image prompt sent to model:**")
                    st.markdown(final_image_prompt or image_prompt)

    # Save assistant message (with image if any)
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": full_story_chunk,
            "image": img_obj,
            "image_prompt": image_prompt,
            "final_image_prompt": final_image_prompt or image_prompt,
            "image_moment": image_moment,
            "beat_type": beat_type,
            "emotion_word": emotion_word,
        }
    )


if __name__ == "__main__":
    main()


