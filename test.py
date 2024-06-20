from huggingface_hub import login
login(token='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')

import torch
from diffusers import StableDiffusion3Pipeline

pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16)
pipe = pipe.to("cuda")

image = pipe(
    "Draw a building exterior with a traditional dial design that has a solid, safe-like feel.",
    negative_prompt="",
    num_inference_steps=28,
    guidance_scale=7.0,
).images[0]

image.save("output.png")