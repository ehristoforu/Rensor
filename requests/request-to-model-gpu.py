import subprocess

def download_file_with_wget(url, save_directory):
    try:
        command = ["wget", url, "-P", save_directory]

        subprocess.run(command, check=True)

    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")


download_file_with_wget("https://huggingface.co/ehristoforu/dreamdrop/resolve/main/dreamdrop-v1.safetensors", "models/checkpoint/gpu-model/base")

download_file_with_wget("https://huggingface.co/ehristoforu/dreamdrop-inpainting/resolve/main/dreamdrop-inpainting.safetensors", "models/checkpoint/gpu-model/inpaint")
