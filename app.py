import json
import os

import streamlit as st
import streamlit.components.v1 as components

from grok_client import GrokClient, build_image_prompt_from_story


st.set_page_config(page_title="Real-Time AI Storyteller", layout="wide")


@st.cache_resource
def get_grok_client(api_key: str) -> GrokClient:
    """
    Cache the Grok client, keyed by API key so we recreate it
    if the user changes keys in the UI.
    """
    return GrokClient(api_key=api_key)


SYSTEM_PROMPT = """
You are an imaginative, cinematic storyteller.
Write in vivid, visual language that is easy to turn into illustrations.

Rules:
- Continue the story in 120–200 words per response.
- End each response with a small cliffhanger or open question.
- Keep the narration focused on concrete scenes, characters, and actions.
"""


def init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []


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

    # Sidebar controls for story tone and image style
    st.sidebar.markdown("### Story & Image Style")
    story_tone = st.sidebar.selectbox(
        "Story tone",
        ["Neutral", "Cozy", "Epic", "Spooky", "Whimsical", "Kids", "SciFi", "Horror"],
        index=0,
    )
    image_style = st.sidebar.selectbox(
        "Image style",
        ["Cinematic", "Storybook", "Anime", "Comic", "Watercolor"],
        index=0,
    )
    st.session_state["story_tone"] = story_tone
    st.session_state["image_style"] = image_style

    st.title("🎭 Real-Time AI Storyteller with Grok")
    st.markdown(
        "Enter a story idea or continuation. Grok will stream the story while generating images for each scene."
    )

    # Show history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if img_url := msg.get("image_url"):
                st.image(img_url, use_column_width=True)

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
            system_prompt=SYSTEM_PROMPT,
            user_prompt=prompt,
            history=history,
        ):
            full_story_chunk += text_piece
            text_placeholder.markdown(full_story_chunk)

        # Once the chunk is complete, trigger client-side TTS.
        speak_text_with_browser_tts(full_story_chunk)

        # After streaming finishes, generate an image for this chunk
        img_url = None
        if full_story_chunk.strip():
            image_prompt = build_image_prompt_from_story(
                full_story_chunk,
                scene_index=scene_index,
                tone=story_tone,
                visual_style=image_style,
            )
            try:
                img_url = client.generate_image(image_prompt)
            except Exception as e:  # noqa: BLE001
                st.error(f"Image generation failed: {e}")
                img_url = None

            if img_url:
                st.image(img_url, caption="Scene Illustration", use_column_width=True)
            else:
                st.info("No image was generated for this scene.")

    # Save assistant message (with image if any)
    st.session_state.messages.append(
        {"role": "assistant", "content": full_story_chunk, "image_url": img_url}
    )


if __name__ == "__main__":
    main()


