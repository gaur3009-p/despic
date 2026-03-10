import subprocess
import time
import gradio as gr
import requests


SERVER_PORT = 7860


def start_server():

    subprocess.Popen(
        ["python", "app/realtime/webrtc_server.py"]
    )

    time.sleep(5)

    return f"WebRTC server started on port {SERVER_PORT}"


def open_meeting():

    return f"http://localhost:{SERVER_PORT}"


with gr.Blocks() as demo:

    gr.Markdown("# Real-Time Multilingual Conversation (Phase 3)")

    start_btn = gr.Button("Start WebRTC Server")

    status = gr.Textbox(label="Server Status")

    link_box = gr.Textbox(label="Meeting URL")

    start_btn.click(start_server, outputs=status)

    start_btn.click(open_meeting, outputs=link_box)


demo.launch(share=True)
