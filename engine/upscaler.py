import cv2
import numpy as np
import time

def upscale_image(input_image):
    start_time = time.time()
    
    upscale_factor = 2
    output_image = cv2.resize(input_image, None, fx = upscale_factor, fy = upscale_factor, interpolation = cv2.INTER_CUBIC)
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    return [output_image], f"Time taken: {execution_time} sec."
