from configs.lcm_ov_pipeline import OVLatentConsistencyModelPipeline
from configs.lcm_scheduler import LCMScheduler
#from optimum.intel import OVStableDiffusionImg2ImgPipeline
import random
import requests
import torch
from PIL import Image
from io import BytesIO

scheduler = LCMScheduler.from_pretrained("deinferno/LCM_Dreamshaper_v7-openvino", subfolder = "scheduler")

pipe = OVLatentConsistencyModelPipeline.from_pretrained(
    "deinferno/LCM_Dreamshaper_v7-openvino",
    scheduler=scheduler,
    compile = False,
)

pipe.save_pretrained(save_directory="models/checkpoint/cpu-model")