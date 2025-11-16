## Real-Time AI Storyteller (Grok + Images)

An AI storyteller that streams an evolving story while generating **relevant images in real time** as the narrative unfolds.  
This MVP uses:

- **Grok (xAI) API** for story generation (chat completions, streamed).
- **Grok image model** (or any OpenAI-compatible image endpoint) for scene illustrations.
- **Python + Streamlit** for a lightweight, browser-based, real-time UI.

### 1. Features (MVP)

- **Interactive story chat**: You give a topic (e.g., “a lonely robot on Mars”) and the AI writes the story in chunks.
- **Streaming text**: Story text appears progressively for a “live narration” feeling.
- **Dynamic images**: After each story chunk, an image is generated for that scene and shown inline.
- **Session history**: Conversation (story + images) persists while the app is running.

### 2. Tech Stack

- **Language**: Python 3.10+
- **UI**: Streamlit
- **LLM + Images**: Grok via xAI API (OpenAI-compatible client)

### 3. Setup

1. **Create and activate a virtual environment (recommended)**

   ```bash
   python -m venv .venv
   .venv\Scripts\activate   # On Windows
   # source .venv/bin/activate  # On macOS/Linux
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**

   Create a `.env` file in the project root:

   ```bash
   XAI_API_KEY=your_xai_grok_api_key_here
   XAI_BASE_URL=https://api.x.ai/v1
   GROK_TEXT_MODEL=grok-4-latest           # Your chosen default Grok 4 text model
   GROK_IMAGE_MODEL=grok-2-image-1212      # Your chosen default Grok 2 image model
   ```

   You can find keys and available model names in the xAI console.

### 4. Running the App

From the project root:

```bash
streamlit run app.py
```

This will open the UI in your browser (or give you a local URL to open).

### 5. High-Level Architecture

- `app.py`
  - Streamlit UI (chat-style interface).
  - Manages session state, user prompts, and displays streaming text + images.
- `grok_client.py`
  - Thin wrapper around the OpenAI-compatible client pointed at xAI’s Grok API.
  - Functions for:
    - **streaming story text** (chat completions, `stream=True`).
    - **generating images** from text prompts.

Flow for each user prompt:

1. User enters a **story topic or continuation**.
2. `grok_client.stream_story_chunk(...)` streams story text tokens/chunks.
3. The UI updates a live text placeholder as chunks arrive.
4. When the chunk finishes:
   - We derive an **image prompt** from the story chunk.
   - `grok_client.generate_image(...)` creates an image.
   - The image is displayed under the corresponding text.

### 6. Next Steps / Extensions

- Add **branching choices** (“What should the hero do next?”).
- Add **voice narration** (e.g., ElevenLabs or other TTS).
- Support **different storytelling styles** (horror, cozy, sci-fi, kids).
- Add **persistence** (save stories + images to a database).

We’ll start with the MVP defined above and iterate from there.


