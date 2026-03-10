import subprocess
import time
import gradio as gr
import requests

SERVER_PORT = 7860


def start_server():

    subprocess.Popen(
        ["python", "app/despic/realtime/webrtc_server.py"]
    )

    time.sleep(5)

    return "WebRTC server started."


def meeting_page():

    try:
        r = requests.get(f"http://localhost:{SERVER_PORT}")
        return r.text
    except:
        return "<h2>Server not running yet</h2>"


with gr.Blocks() as demo:

    gr.Markdown("# Phase-3 Realtime Multilingual Conversation")

    start_btn = gr.Button("Start WebRTC Server")

    status = gr.Textbox()

    start_btn.click(start_server, outputs=status)

    gr.HTML(meeting_page)

demo.launch(share=True)
