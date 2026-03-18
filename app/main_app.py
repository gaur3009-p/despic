import gradio as gr
from pipelines.streaming_pipeline import run_pipeline, reset_stream, LANGUAGES
from pipelines.sentence_pipeline  import run_travel_pipeline

LANG_NAMES = list(LANGUAGES.keys())

# ---------------------------------------------------------------------------
# Gradio callback wrappers
# ---------------------------------------------------------------------------

def _interpreter_cb(audio, target_lang, transcript_state):
    """Streaming callback – fires on every audio chunk from the microphone."""
    if audio is None:
        return transcript_state, "", None, ""

    sr, data = audio
    transcript, translation, speech, latency = run_pipeline(data, sr, target_lang)
    return transcript, translation, speech, latency


def _new_session_cb():
    """Reset internal streaming state when the user starts a new session."""
    reset_stream()
    return "", "", None, ""


def _translator_cb(audio, target_lang):
    """One-shot callback for the recorded-sentence tab."""
    if audio is None:
        return "", "", None, ""

    sr, data = audio
    text, translated, speech, timings = run_travel_pipeline(data, sr, target_lang)

    if not timings:
        return text, translated, speech, ""

    latency = (
        f"ASR {timings.get('asr', 0)} ms  |  "
        f"Translate {timings.get('translation', 0)} ms  |  "
        f"TTS {timings.get('tts', 0)} ms  |  "
        f"Total {timings.get('total', 0)} ms"
    )
    return text, translated, speech, latency


# ---------------------------------------------------------------------------
# CSS – dark, minimal, production-quality
# ---------------------------------------------------------------------------

CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@400;600;800&display=swap');

:root {
    --bg:       #0d0f14;
    --surface:  #151820;
    --border:   #252a35;
    --accent:   #4f8ef7;
    --accent2:  #a78bfa;
    --success:  #34d399;
    --text:     #e2e8f0;
    --muted:    #64748b;
    --radius:   10px;
    --font-ui:  'Syne', sans-serif;
    --font-mono:'DM Mono', monospace;
}

body, .gradio-container {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--font-ui) !important;
}

/* Header */
.app-header {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.5rem;
}
.app-header h1 {
    font-size: 2rem;
    font-weight: 800;
    letter-spacing: -0.5px;
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent2) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.4rem;
}
.app-header p {
    color: var(--muted);
    font-size: 0.9rem;
    font-weight: 400;
    margin: 0;
}

/* Tabs */
.tab-nav button {
    background: transparent !important;
    border: none !important;
    color: var(--muted) !important;
    font-family: var(--font-ui) !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    padding: 0.6rem 1.2rem !important;
    border-bottom: 2px solid transparent !important;
    transition: all 0.2s !important;
}
.tab-nav button.selected {
    color: var(--accent) !important;
    border-bottom-color: var(--accent) !important;
}

/* Panels */
.panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.25rem;
}

/* Labels */
label span {
    font-family: var(--font-ui) !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    color: var(--muted) !important;
}

/* Textboxes */
textarea, input[type="text"] {
    background: var(--bg) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: var(--radius) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.9rem !important;
}
textarea:focus, input[type="text"]:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(79,142,247,0.15) !important;
}

/* Latency bar */
.latency-box textarea {
    color: var(--success) !important;
    font-size: 0.78rem !important;
    font-family: var(--font-mono) !important;
}

/* Dropdown */
.wrap-inner {
    background: var(--bg) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
    font-family: var(--font-ui) !important;
}

/* Primary button */
button.primary {
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent2) 100%) !important;
    border: none !important;
    color: #fff !important;
    font-family: var(--font-ui) !important;
    font-weight: 700 !important;
    letter-spacing: 0.04em !important;
    border-radius: var(--radius) !important;
    padding: 0.65rem 1.6rem !important;
    transition: opacity 0.2s !important;
}
button.primary:hover { opacity: 0.88 !important; }

/* Reset button */
button.secondary {
    background: transparent !important;
    border: 1px solid var(--border) !important;
    color: var(--muted) !important;
    font-family: var(--font-ui) !important;
    font-size: 0.8rem !important;
    border-radius: var(--radius) !important;
}
button.secondary:hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
}

/* Status dot for live tab */
.live-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 0.75rem;
    color: var(--success);
    font-family: var(--font-mono);
    margin-bottom: 0.75rem;
}
.live-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--success);
    animation: pulse 1.5s infinite;
}
@keyframes pulse {
    0%,100% { opacity:1; transform:scale(1); }
    50%      { opacity:0.4; transform:scale(1.3); }
}

/* Grid layout helpers */
.two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
@media (max-width: 720px) { .two-col { grid-template-columns: 1fr; } }
"""

# ---------------------------------------------------------------------------
# Build the app
# ---------------------------------------------------------------------------

with gr.Blocks(css=CSS, title="AI Speech Interpreter") as demo:

    # ── Header ──────────────────────────────────────────────────────────────
    gr.HTML("""
    <div class="app-header">
      <h1>🌐 AI Speech Interpreter</h1>
      <p>Real-time speech translation across 16 Indic + world languages</p>
    </div>
    """)

    # ── Tabs ─────────────────────────────────────────────────────────────────
    with gr.Tabs():

        # ── Tab 1: Live Streaming ──────────────────────────────────────────
        with gr.Tab("⚡ Live Interpreter"):

            gr.HTML('<div class="live-badge"><span class="live-dot"></span>Streams in real-time · chunk-triggered · low latency</div>')

            with gr.Row():
                with gr.Column(scale=1):
                    target_live = gr.Dropdown(
                        LANG_NAMES, value="Hindi",
                        label="Translate to"
                    )
                    mic_stream = gr.Audio(
                        sources=["microphone"],
                        streaming=True,
                        type="numpy",
                        label="Microphone"
                    )
                    reset_btn = gr.Button("↺  New Session", variant="secondary", size="sm")

                with gr.Column(scale=2):
                    transcript_live = gr.Textbox(
                        label="Transcript (original)",
                        lines=4, interactive=False
                    )
                    translation_live = gr.Textbox(
                        label="Translation",
                        lines=4, interactive=False
                    )
                    audio_live = gr.Audio(
                        label="Dubbed audio",
                        autoplay=True, interactive=False
                    )
                    latency_live = gr.Textbox(
                        label="Latency",
                        lines=1, interactive=False,
                        elem_classes=["latency-box"]
                    )

            # Wiring
            mic_stream.stream(
                _interpreter_cb,
                inputs=[mic_stream, target_live, transcript_live],
                outputs=[transcript_live, translation_live, audio_live, latency_live],
            )
            reset_btn.click(
                _new_session_cb,
                outputs=[transcript_live, translation_live, audio_live, latency_live],
            )

        # ── Tab 2: Sentence Translator ──────────────────────────────────────
        with gr.Tab("🎙 Speech Translator"):

            gr.Markdown(
                "_Record a sentence – uses the high-accuracy Whisper **medium** model "
                "with full beam search for best transcription quality._",
            )

            with gr.Row():
                with gr.Column(scale=1):
                    target_sent = gr.Dropdown(
                        LANG_NAMES, value="Hindi",
                        label="Translate to"
                    )
                    mic_record = gr.Audio(
                        sources=["microphone"],
                        type="numpy",
                        label="Record your voice"
                    )
                    translate_btn = gr.Button("Translate & Dub", variant="primary")

                with gr.Column(scale=2):
                    transcript_sent = gr.Textbox(
                        label="Transcript (original)",
                        lines=4, interactive=False
                    )
                    translation_sent = gr.Textbox(
                        label="Translation",
                        lines=4, interactive=False
                    )
                    audio_sent = gr.Audio(
                        label="Dubbed audio",
                        autoplay=True, interactive=False
                    )
                    latency_sent = gr.Textbox(
                        label="Latency",
                        lines=1, interactive=False,
                        elem_classes=["latency-box"]
                    )

            translate_btn.click(
                _translator_cb,
                inputs=[mic_record, target_sent],
                outputs=[transcript_sent, translation_sent, audio_sent, latency_sent],
            )


if __name__ == "__main__":
    demo.launch(share=True)
