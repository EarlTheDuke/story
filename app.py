import streamlit as st

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


def ask_for_api_key() -> str | None:
    """
    Let the user paste their xAI API key directly in the UI.
    This bypasses any issues with .env loading on their machine.
    """
    saved = st.session_state.get("xai_api_key", "")
    api_key = st.sidebar.text_input(
        "xAI API key",
        value=saved,
        type="password",
        help="Paste your xAI key from the x.ai console (it stays local on your machine).",
    ).strip()
    st.session_state["xai_api_key"] = api_key
    return api_key or None


def main() -> None:
    init_session_state()

    api_key = ask_for_api_key()
    if not api_key:
        st.warning("Please paste your xAI API key in the left sidebar to begin.")
        return

    client = get_grok_client(api_key)

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

        # After streaming finishes, generate an image for this chunk
        img_url = None
        if full_story_chunk.strip():
            image_prompt = build_image_prompt_from_story(full_story_chunk)
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


