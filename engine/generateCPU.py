from configs.lcm_ov_pipeline import OVLatentConsistencyModelPipeline
from configs.lcm_scheduler import LCMScheduler
import random
import requests
import gradio as gr
import torch
import time
from PIL import Image
from io import BytesIO
import os
from dotenv import load_dotenv

load_dotenv("config.txt")

scheduler = LCMScheduler.from_pretrained("models/checkpoint/cpu-model", subfolder = "scheduler")

pipe_t2i = OVLatentConsistencyModelPipeline.from_pretrained(
    "models/checkpoint/cpu-model",
    scheduler=scheduler,
    compile = False,
)

width = int(input('Enter width: '))
height = int(input('Enter height: '))

pipe_t2i.reshape(batch_size=1, width=width, height=height, num_images_per_prompt=1)
pipe_t2i.compile()

print("[PIPE COMPILED]")

def cpugen(prompt, mode, guidance, num_images, progress = gr.Progress(track_tqdm=True)):
    img2img_change=False
    results = []
    if mode == "Fast":
        steps = 6
    elif mode == "High Quality":
        steps = 10
    else:
        steps = 4
    seed = random.randint(1, 99999999)
    num = random.randint(100, 99999)
    #name = f"outputs/{num}_txt2img_cpu.jpg"
    if not img2img_change:
        start_time = time.time()
        for _ in range(num_images):
            image = pipe_t2i(
                prompt=f"{prompt}, epic realistic, faded, ((neutral colors)), art, (hdr:1.5), (muted colors:1.2), pastel, hyperdetailed, (artstation:1.5), warm lights, dramatic light, (intricate details:1.2), vignette, complex background, rutkowski",
                #negative_prompt="(deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, (mutated hands and fingers:1.4), disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation",
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=guidance,
                output_type="pil"
            ).images
            image[0].save(f"outputs/{num}_txt2img_cpu{_}.jpg")
            results.append(image[0])
        #results[_].save(name)
        end_time = time.time()
        execution_time = end_time - start_time
    '''
    else:
        init_image = init.resize((width, height))
        start_time = time.time()
        for _ in range(num_images):
            image = pipe_i2i(
                prompt=prompt,
                image=init_image,
                #negative_prompt="(deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, (mutated hands and fingers:1.4), disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation",
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=guidance,
                output_type="pil"
            ).images
            image[0].save(f"outputs/{num}_img2img_cpu{_}.jpg")
            results.append(image[0])
        #results[_].save(name)
        end_time = time.time()
        execution_time = end_time - start_time
    '''



    
    return results, f"Time taken: {execution_time} sec."