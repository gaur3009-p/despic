import subprocess
import time
import requests
import gradio as gr

SERVER_PORT = 7860
server_process = None


def start_server():
    global server_process

    if server_process is None:
        server_process = subprocess.Popen(
            ["python", "app/realtime/webrtc_server.py"]
        )

        time.sleep(5)

    return "WebRTC server started on port 7860"


def load_meeting_page():

    try:
        r = requests.get(f"http://localhost:{SERVER_PORT}")
        return r.text
    except:
        return "<h3>Server not ready yet. Click 'Start WebRTC Server' first.</h3>"


with gr.Blocks() as demo:

    gr.Markdown("# Phase-3 Realtime Multilingual Conversation")

    start_btn = gr.Button("Start WebRTC Server")

    status = gr.Textbox(label="Server Status")

    meeting_ui = gr.HTML()

    start_btn.click(start_server, outputs=status)
    start_btn.click(load_meeting_page, outputs=meeting_ui)

demo.launch(share=True)
