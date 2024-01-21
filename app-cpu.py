import gradio as gr
import random
import requests
import time
import argparse
import os
from dotenv import load_dotenv

load_dotenv("config.txt")

from engine.generateCPU import cpugen
from engine.upscaler import upscale_image
from engine.promptGenerator import prompting


css = """
#container{
    margin: 0 auto;
    max-width: 40rem;
}
#intro{
    max-width: 100%;
    text-align: center;
    margin: 0 auto;
}
#generate_button {
    color: white;
    border-color: #007bff;
    background: #007bff;
    width: 200px;
    height: 50px;
}
footer {
    visibility: hidden
}
"""

with gr.Blocks(title="Rensor", css=css, theme="ehristoforu/RE_Theme") as webui:
    with gr.Row():
        with gr.Row(visible=False, variant="panel") as prompter:
            with gr.Column(scale=1):
                chatbot = gr.Textbox(show_label=False, interactive=False, max_lines=16, lines=14)
                with gr.Row():
                    chat_text = gr.Textbox(show_label=False, placeholder="Enter short prompt", max_lines=2, lines=1, interactive=True, scale=30)
                    chat_submit = gr.Button(value="Prompt", scale=1)

                chat_submit.click(fn=lambda x: gr.update(value="Prompting...", interactive=False), inputs=chat_submit, outputs=chat_submit).then(prompting, inputs=chat_text, outputs=chatbot).then(fn=lambda x: gr.update(value="Prompt", interactive=True), inputs=chat_submit, outputs=chat_submit)

        with gr.Column(scale=3):
            with gr.Row():
                gallery = gr.Gallery(show_label=False, rows=2, columns=6, preview=True, value=["assets/favicon.png"])
            work_time = gr.Markdown(visible=True)
            dump_outputs = gr.Gallery(visible=False)
            with gr.Row():
                prompt = gr.Textbox(show_label=False, placeholder="Your amazing prompt...", max_lines=2, lines=2, interactive=True, scale=18)
                button = gr.Button(value="Generate", variant="primary", interactive=True, scale=1)
            with gr.Row():
                advenced = gr.Checkbox(label="Advanced settings", value=False, interactive=True)
                prompter_change = gr.Checkbox(label="Prompter", value=False, interactive=True)
        with gr.Row(visible=False, variant="panel") as settings_tab:
            with gr.Column(scale=1):
                with gr.Tab("Settings"):
                    with gr.Row(scale=10):
                        mode = gr.Radio(label="Mode", choices=["High Quality", "Fast", "Super-fast"], value="Fast", info="Relationship between generation speed and quality.", interactive=True)
                    with gr.Row(scale=10):
                        guidance = gr.Slider(label="Guidance Scale", maximum=20.0, minimum=0.0, value=8.0, step=0.1, interactive=True)
                    with gr.Row(scale=10):
                        num_images = gr.Slider(label="Number of images", maximum=12, minimum=1, value=2, step=1, interactive=True)
                    with gr.Row(scale=1):
                        upscale_button = gr.Image(label="🚀 Upload image to 2x upscale", sources="upload", type="numpy", show_download_button=False, interactive=True)
                    
    
     
    button.click(fn=lambda x: gr.update(visible=False), inputs=work_time, outputs=work_time).then(fn=lambda x: gr.update(value="Generating...", variant="secondary", interactive=False), inputs=button, outputs=button).then(cpugen, inputs=[prompt, mode, guidance, num_images], outputs=[gallery, work_time]).then(fn=lambda x: gr.update(value="Generate", variant="primary", interactive=True), inputs=button, outputs=button).then(fn=lambda x: gr.update(visible=True), inputs=work_time, outputs=work_time)
    
    upscale_button.upload(fn=lambda x: gr.update(visible=False), inputs=work_time, outputs=work_time).then(fn=lambda x: gr.update(label="🖼️ Image uploaded to 2x upscale", interactive=False), inputs=upscale_button, outputs=upscale_button).then(fn=lambda x: gr.update(value="Upscaling...", variant="secondary", interactive=False), inputs=button, outputs=button).then(upscale_image, inputs=upscale_button, outputs=[gallery, work_time]).then(fn=lambda x: gr.update(label="🚀 Upload image to 2x upscale", interactive=True), inputs=upscale_button, outputs=upscale_button).then(fn=lambda x: gr.update(value="Generate", variant="primary", interactive=True), inputs=button, outputs=button).then(fn=lambda x: gr.update(visible=True), inputs=work_time, outputs=work_time)
    
    advenced.change(fn=lambda x: gr.update(visible=x), inputs=advenced, outputs=settings_tab, queue=False, api_name=False)
    prompter_change.change(fn=lambda x: gr.update(visible=x), inputs=prompter_change, outputs=prompter, queue=False, api_name=False)

    
    '''
    advenced.change(fn=lambda x: gr.update(visible=x), inputs=advenced, outputs=img2img_change, queue=False, api_name=False)
    advenced.change(fn=lambda x: gr.update(visible=x), inputs=advenced, outputs=i2i_strength, queue=False, api_name=False)
    advenced.change(fn=lambda x: gr.update(visible=x), inputs=advenced, outputs=init_image, queue=False, api_name=False)
    '''

    '''
    img2img_change.change(
        fn=lambda x: gr.update(interactive=x),
        inputs=img2img_change,
        outputs=init_image,
        queue=False,
        api_name=False,
    ).then(
       fn=lambda x: gr.update(interactive=x),
        inputs=img2img_change,
        outputs=i2i_strength,
        queue=False,
        api_name=False,
    )
    '''

webui.queue(max_size=20).launch(debug=False, share=True, server_port=5555, quiet=True, show_api=False, favicon_path="assets/favicon.png", inbrowser=True)