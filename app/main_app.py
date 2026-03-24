import gradio as gr
 
from pipelines.streaming_pipeline import (
    run_pipeline       as run_stream,
    reset_stream,
    LANGUAGES,
)
from pipelines.live_pipeline import (
    run_live_pipeline  as run_live,
    reset_live,
)
from pipelines.sentence_pipeline import run_travel_pipeline
 
LANG_NAMES = list(LANGUAGES.keys())
 
# ── Gradio callbacks ──────────────────────────────────────────────────────────
 
def _stream_cb(audio, target_lang, _state):
    if audio is None:
        return _state, "", None, ""
    sr, data = audio
    transcript, translation, speech, latency = run_stream(data, sr, target_lang)
    return transcript, translation, speech, latency
 
def _stream_reset():
    reset_stream()
    return "", "", None, ""
 
def _live_cb(audio, target_lang):
    # Pass None audio through — pipeline preserves display state internally
    if audio is None:
        transcript, translation, latency = run_live(None, 16000, target_lang)
    else:
        sr, data = audio
        transcript, translation, latency = run_live(data, sr, target_lang)
    return transcript, translation, latency
 
def _live_reset():
    reset_live()
    return "", "", ""
 
def _sentence_cb(audio, target_lang):
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
 
# ── CSS ───────────────────────────────────────────────────────────────────────
 
CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@400;600;800&display=swap');
 
:root {
    --bg:        #0d0f14;
    --surface:   #151820;
    --surface2:  #1c2030;
    --border:    #252a35;
    --accent:    #4f8ef7;
    --accent2:   #a78bfa;
    --live:      #f97316;
    --success:   #34d399;
    --text:      #e2e8f0;
    --muted:     #64748b;
    --radius:    10px;
    --font-ui:   'Syne', sans-serif;
    --font-mono: 'DM Mono', monospace;
}
 
body, .gradio-container {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--font-ui) !important;
}
 
.app-header {
    text-align: center;
    padding: 2.2rem 1rem 1.4rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.5rem;
}
.app-header h1 {
    font-size: 2rem; font-weight: 800; letter-spacing: -0.5px;
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent2) 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin: 0 0 0.35rem;
}
.app-header p { color: var(--muted); font-size: 0.88rem; margin: 0; }
 
.tab-nav button {
    background: transparent !important; border: none !important;
    color: var(--muted) !important; font-family: var(--font-ui) !important;
    font-weight: 600 !important; font-size: 0.88rem !important;
    padding: 0.6rem 1.1rem !important;
    border-bottom: 2px solid transparent !important; transition: all 0.2s !important;
}
.tab-nav button.selected { color: var(--accent) !important; border-bottom-color: var(--accent) !important; }
 
label span {
    font-family: var(--font-ui) !important; font-size: 0.74rem !important;
    font-weight: 600 !important; text-transform: uppercase !important;
    letter-spacing: 0.08em !important; color: var(--muted) !important;
}
 
textarea, input[type="text"] {
    background: var(--bg) !important; border: 1px solid var(--border) !important;
    color: var(--text) !important; border-radius: var(--radius) !important;
    font-family: var(--font-mono) !important; font-size: 0.88rem !important;
    transition: border-color 0.15s !important;
}
textarea:focus, input[type="text"]:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(79,142,247,0.12) !important; outline: none !important;
}
 
.latency-box textarea { color: var(--success) !important; font-size: 0.76rem !important; }
.trans-box textarea   { background: var(--surface2) !important; color: #c4b5fd !important; }
 
.wrap-inner {
    background: var(--bg) !important; border-color: var(--border) !important;
    color: var(--text) !important; font-family: var(--font-ui) !important;
}
 
button.primary {
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent2) 100%) !important;
    border: none !important; color: #fff !important;
    font-family: var(--font-ui) !important; font-weight: 700 !important;
    letter-spacing: 0.04em !important; border-radius: var(--radius) !important;
    padding: 0.65rem 1.6rem !important; transition: opacity 0.2s !important;
}
button.primary:hover { opacity: 0.85 !important; }
 
button.secondary {
    background: transparent !important; border: 1px solid var(--border) !important;
    color: var(--muted) !important; font-family: var(--font-ui) !important;
    font-size: 0.8rem !important; border-radius: var(--radius) !important;
    transition: all 0.15s !important;
}
button.secondary:hover { border-color: var(--accent) !important; color: var(--accent) !important; }
 
.badge {
    display: inline-flex; align-items: center; gap: 7px;
    font-size: 0.74rem; font-family: var(--font-mono);
    margin-bottom: 0.9rem; padding: 0.3rem 0.9rem;
    border-radius: 99px; border: 1px solid;
}
.badge-stream { color: var(--accent);  border-color: rgba(79,142,247,0.3);  background: rgba(79,142,247,0.06); }
.badge-live   { color: var(--live);    border-color: rgba(249,115,22,0.3);  background: rgba(249,115,22,0.06); }
.badge-rec    { color: var(--success); border-color: rgba(52,211,153,0.3);  background: rgba(52,211,153,0.06); }
 
.dot { width:7px; height:7px; border-radius:50%; animation: pulse 1.6s infinite; }
.dot-blue   { background: var(--accent); }
.dot-orange { background: var(--live); }
.dot-green  { background: var(--success); }
 
@keyframes pulse {
    0%,100% { opacity:1; transform:scale(1);   }
    50%      { opacity:.4; transform:scale(1.4); }
}
 
.note {
    font-size: 0.78rem; color: var(--muted);
    background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 0.55rem 0.9rem;
    margin-bottom: 1rem; font-family: var(--font-mono);
}
 
.side-by-side { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
@media (max-width: 740px) { .side-by-side { grid-template-columns: 1fr; } }
"""
 
# ── App ───────────────────────────────────────────────────────────────────────
 
with gr.Blocks(css=CSS, title="AI Speech Interpreter") as demo:
 
    gr.HTML("""
    <div class="app-header">
      <h1>🌐 AI Speech Interpreter</h1>
      <p>Real-time speech translation across 16 Indic &amp; world languages</p>
    </div>
    """)
 
    with gr.Tabs():
 
        # ── Tab 1: Live Interpreter (streaming + dubbing) ─────────────────────
        with gr.Tab("⚡ Live Interpreter"):
            gr.HTML("""<div class="badge badge-stream">
              <span class="dot dot-blue"></span>
              Streaming · Whisper small · chunk-triggered · with audio dubbing
            </div>""")
 
            with gr.Row():
                with gr.Column(scale=1):
                    tgt_stream = gr.Dropdown(LANG_NAMES, value="Hindi", label="Translate to")
                    mic_stream = gr.Audio(sources=["microphone"], streaming=True,
                                         type="numpy", label="Microphone")
                    reset_stream_btn = gr.Button("↺  New Session", variant="secondary", size="sm")
 
                with gr.Column(scale=2):
                    with gr.Row(elem_classes=["side-by-side"]):
                        transcript_stream  = gr.Textbox(label="Transcript (original)",
                                                        lines=5, interactive=False)
                        translation_stream = gr.Textbox(label="Translation",
                                                        lines=5, interactive=False,
                                                        elem_classes=["trans-box"])
                    audio_stream   = gr.Audio(label="Dubbed audio", autoplay=True, interactive=False)
                    latency_stream = gr.Textbox(label="Latency", lines=1, interactive=False,
                                                elem_classes=["latency-box"])
 
            mic_stream.stream(
                _stream_cb,
                inputs=[mic_stream, tgt_stream, transcript_stream],
                outputs=[transcript_stream, translation_stream, audio_stream, latency_stream],
            )
            reset_stream_btn.click(
                _stream_reset,
                outputs=[transcript_stream, translation_stream, audio_stream, latency_stream],
            )
 
        # ── Tab 2: Live Transcription + Translation (NO dubbing) ──────────────
        with gr.Tab("📝 Live Transcription"):
            gr.HTML("""<div class="badge badge-live">
              <span class="dot dot-orange"></span>
              Real-time text only · Whisper medium · fastest feedback · no audio generation
            </div>""")
            gr.HTML("""<div class="note">
              ℹ&nbsp; Transcript and translation update simultaneously as you speak.
              No TTS — so feedback is near-instant. Uses Whisper <strong>medium</strong> for accuracy.
            </div>""")
 
            with gr.Row():
                with gr.Column(scale=1):
                    tgt_live = gr.Dropdown(LANG_NAMES, value="Hindi", label="Translate to")
                    mic_live = gr.Audio(sources=["microphone"], streaming=True,
                                        type="numpy", label="Microphone")
                    reset_live_btn = gr.Button("↺  New Session", variant="secondary", size="sm")
 
                with gr.Column(scale=2):
                    with gr.Row(elem_classes=["side-by-side"]):
                        transcript_live  = gr.Textbox(label="Live Transcript",
                                                      lines=10, interactive=False,
                                                      placeholder="Start speaking…")
                        translation_live = gr.Textbox(label="Live Translation",
                                                      lines=10, interactive=False,
                                                      elem_classes=["trans-box"],
                                                      placeholder="Translation will appear here…")
                    latency_live = gr.Textbox(label="Latency (per chunk)", lines=1,
                                              interactive=False, elem_classes=["latency-box"])
 
            mic_live.stream(
                _live_cb,
                inputs=[mic_live, tgt_live],
                outputs=[transcript_live, translation_live, latency_live],
            )
            reset_live_btn.click(
                _live_reset,
                outputs=[transcript_live, translation_live, latency_live],
            )
 
        # ── Tab 3: Sentence Translator (record + dub) ─────────────────────────
        with gr.Tab("🎙 Speech Translator"):
            gr.HTML("""<div class="badge badge-rec">
              <span class="dot dot-green"></span>
              Record a full sentence · Whisper medium · full beam search · highest accuracy
            </div>""")
 
            with gr.Row():
                with gr.Column(scale=1):
                    tgt_sent = gr.Dropdown(LANG_NAMES, value="Hindi", label="Translate to")
                    mic_sent = gr.Audio(sources=["microphone"], type="numpy",
                                        label="Record your voice")
                    translate_btn = gr.Button("Translate & Dub", variant="primary")
 
                with gr.Column(scale=2):
                    with gr.Row(elem_classes=["side-by-side"]):
                        transcript_sent  = gr.Textbox(label="Transcript (original)",
                                                      lines=5, interactive=False)
                        translation_sent = gr.Textbox(label="Translation",
                                                      lines=5, interactive=False,
                                                      elem_classes=["trans-box"])
                    audio_sent   = gr.Audio(label="Dubbed audio", autoplay=True, interactive=False)
                    latency_sent = gr.Textbox(label="Latency", lines=1, interactive=False,
                                              elem_classes=["latency-box"])
 
            translate_btn.click(
                _sentence_cb,
                inputs=[mic_sent, tgt_sent],
                outputs=[transcript_sent, translation_sent, audio_sent, latency_sent],
            )
 
 
if __name__ == "__main__":
    demo.launch(share=True)
