import os
import subprocess
# Check if diffusers is installed on runtime
try:
    import diffusers
except ImportError:
    print("Installing diffusers...")
    subprocess.run(["pip", "install", "diffusers"])
    import diffusers

from diffusers import StableDiffusionPipeline
import torch

model_id = ".cache/huggingface/hub/models runwayml stable-diffusion-v1-5/snapshots/608a7bbe4e4a6a66513c80999e32708671fd2ac0/"

# Initialize pipeline with float32 and CPU
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float32,
    local_files_only=True
)
pipe = pipe.to("cpu")

# Set the generator for reproducibility
generator = torch.Generator("cpu").manual_seed(493856538)

# Generate image with modified parameters
prompt = "A photo of a flying cat"
image = pipe(
    prompt,
    height=32,
    width=32,
    num_inference_steps=20,
    generator=generator
).images[0]

# Save the output image
image.save("flying_cat.png")