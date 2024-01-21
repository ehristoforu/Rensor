import random
import requests
import torch
from io import BytesIO
from PIL import Image
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline, StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, StableDiffusionXLInpaintPipeline, DiffusionPipeline
import os
import time
from dotenv import load_dotenv

load_dotenv("config.txt")

xl = os.getenv("xl")

with torch.no_grad():
  pipe = StableDiffusionPipeline.from_single_file(
      "models/checkpoint/gpu-model/base/dreamdrop-v1.safetensors",
      use_safetensors=True,
      cache_dir="models/checkpoint/gpu-model/base/cache_dir",
      scheduler_type="euler-ancestral"
  )
  time.sleep(20)
  pipe_inpaint = StableDiffusionInpaintPipeline.from_single_file(
      "models/checkpoint/gpu-model/inpaint/dreamdrop-inpainting.safetensors",
      use_safetensors=True,
      cache_dir="models/checkpoint/gpu-model/inpaint/cache_dir",
      scheduler_type="euler-ancestral"
  )

