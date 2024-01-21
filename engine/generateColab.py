import random
import requests
import torch
import time
import gradio as gr
from io import BytesIO
from PIL import Image
import imageio
from dotenv import load_dotenv
import os

load_dotenv("config.txt")

path_to_base_model = "models/checkpoint/gpu-model/base/dreamdrop-v1.safetensors"
path_to_inpaint_model = "models/checkpoint/gpu-model/inpaint/dreamdrop-inpainting.safetensors"

xl = os.getenv("xl")

if xl == "True":
    from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, StableDiffusionXLInpaintPipeline
    pipe_t2i = StableDiffusionXLPipeline.from_single_file(path_to_base_model, torch_dtype=torch.float16, use_safetensors=True)
    pipe_t2i = pipe_t2i.to("cuda")

    pipe_i2i = StableDiffusionXLImg2ImgPipeline.from_single_file(path_to_base_model, torch_dtype=torch.float16, use_safetensors=True)
    pipe_i2i = pipe_i2i.to("cuda")

    pipe_inpaint = StableDiffusionXLInpaintPipeline.from_single_file(path_to_inpaint_model, torch_dtype=torch.float16, use_safetensors=True)
    pipe_inpaint = pipe_inpaint.to("cuda")
else:
    from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline
    pipe_t2i = StableDiffusionPipeline.from_single_file(path_to_base_model, torch_dtype=torch.float16, use_safetensors=True)
    pipe_t2i = pipe_t2i.to("cuda")

    pipe_i2i = StableDiffusionImg2ImgPipeline.from_single_file(path_to_base_model, torch_dtype=torch.float16, use_safetensors=True)
    pipe_i2i = pipe_i2i.to("cuda")

    pipe_inpaint = StableDiffusionInpaintPipeline.from_single_file(path_to_inpaint_model, torch_dtype=torch.float16, use_safetensors=True)
    pipe_inpaint = pipe_inpaint.to("cuda")


pipe_t2i.load_lora_weights(pretrained_model_name_or_path_or_dict="models/lora", weight_name="epic_noiseoffset.safetensors")
pipe_t2i.fuse_lora(lora_scale=0.1)

pipe_i2i.load_lora_weights(pretrained_model_name_or_path_or_dict="models/lora", weight_name="epic_noiseoffset.safetensors")
pipe_i2i.fuse_lora(lora_scale=0.1)

pipe_inpaint.load_lora_weights(pretrained_model_name_or_path_or_dict="models/lora", weight_name="epic_noiseoffset.safetensors")
pipe_inpaint.fuse_lora(lora_scale=0.1)


def gpugen(prompt, mode, guidance, width, height, num_images, i2i_strength, inpaint_strength, i2i_change, inpaint_change, init=None, inpaint_image=None, progress = gr.Progress(track_tqdm=True)):
    if mode == "Fast":
        steps = 30
    elif mode == "High Quality":
        steps = 45
    else:
        steps = 20
    results = []
    seed = random.randint(1, 9999999)
    if not i2i_change and not inpaint_change:
        num = random.randint(100, 99999)
        start_time = time.time()
        for _ in range(num_images):
            image = pipe_t2i(
                prompt=prompt,
                negative_prompt="(deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, (mutated hands and fingers:1.4), disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation",
                num_inference_steps=steps,
                guidance_scale=guidance,
                width=width, height=height,
                seed=seed,
            ).images
            image[0].save(f"outputs/{num}_txt2img_gpu{_}.jpg")
            results.append(image[0])
        end_time = time.time()
        execution_time = end_time - start_time
        return results, f"Time taken: {execution_time} sec."
    elif inpaint_change and not i2i_change:
        imageio.imwrite("output_image.png", inpaint_image["mask"])

        num = random.randint(100, 99999)
        start_time = time.time()
        for _ in range(num_images):
            image = pipe_inpaint(
                prompt=prompt,
                image=inpaint_image["image"],
                mask_image=inpaint_image["mask"],
                negative_prompt="(deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, (mutated hands and fingers:1.4), disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation",
                num_inference_steps=steps,
                guidance_scale=guidance,
                strength=inpaint_strength,
                width=width, height=height,
                seed=seed,
            ).images
            image[0].save(f"outputs/{num}_inpaint_gpu{_}.jpg")
            results.append(image[0])
        end_time = time.time()
        execution_time = end_time - start_time
        return results, f"Time taken: {execution_time} sec."
    
    else:
        num = random.randint(100, 99999)
        start_time = time.time()
        for _ in range(num_images):
            image = pipe_i2i(
                prompt=prompt,
                negative_prompt="(deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, (mutated hands and fingers:1.4), disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation",
                image=init,
                num_inference_steps=steps,
                guidance_scale=guidance,
                width=width, height=height,
                strength=i2i_strength,
                seed=seed,
            ).images
            image[0].save(f"outputs/{num}_img2img_gpu{_}.jpg")
            results.append(image[0])
        end_time = time.time()
        execution_time = end_time - start_time
        return results, f"Time taken: {execution_time} sec."
    
            