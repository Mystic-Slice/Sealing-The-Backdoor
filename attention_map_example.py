from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from matplotlib import pyplot as plt
import torch

from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from torchvision.utils import save_image

from attention_map.utils import (
    attn_maps,
    cross_attn_init,
    register_cross_attention_hook,
    set_layer_with_name_and_path,
    save_by_timesteps_and_path,
    save_by_timesteps
)
cross_attn_init()

sd_path = '../sd'
device = 'cuda'

prompt = 'Good morning sign with a red circle'

print(f"Loading models: {sd_path}")
print("Loading tokenizer")
tokenizer = CLIPTokenizer.from_pretrained(
    sd_path,
    subfolder="tokenizer",
    low_cpu_mem_usage=True,
)
print("Loading Text Encoder")
text_encoder = CLIPTextModel.from_pretrained(
    sd_path,
    subfolder="text_encoder",
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True,
)
print("Loading VAE")
vae = AutoencoderKL.from_pretrained(
    sd_path,
    subfolder="vae",
    torch_dtype=torch.float32,
    low_cpu_mem_usage=False,
)
noise_scheduler = DDPMScheduler.from_pretrained(
    sd_path,
    subfolder="scheduler",
    low_cpu_mem_usage=False
)

unet = UNet2DConditionModel.from_pretrained(
    "unet_backdoored",
    torch_dtype=torch.float32,
    low_cpu_mem_usage=False,
)

unet = set_layer_with_name_and_path(unet)
unet = register_cross_attention_hook(unet)

pipeline = StableDiffusionPipeline.from_pretrained(
    pretrained_model_name_or_path=sd_path,
    tokenizer = tokenizer,
    text_encoder=text_encoder,
    vae=vae,
    unet=unet,
).to(device)

with torch.no_grad():
    tokens = tokenizer(
        prompt,
        max_length=tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors='pt'
    ).input_ids
    token_embeddings = text_encoder(tokens.to(text_encoder.device))[0]

    latents = torch.randn(
        (1, 4, 64, 64),
        device=unet.device
    ).to(torch.float32)

    timesteps = torch.randint(int(noise_scheduler.config.num_train_timesteps * 0.75), noise_scheduler.config.num_train_timesteps, (1,), 
                        device=unet.device).long()

    pred = unet(
        latents,
        timesteps,
        encoder_hidden_states=token_embeddings
    ).sample.to(torch.float32)

height = 512
width = 768
image = pipeline(
    prompt,
    height=height,
    width=width,
    num_inference_steps=50,
).images[0]

print("Generating attention maps")
# save_by_timesteps(tokenizer, prompt, height, width)

image.save("attn_maps_by_timesteps/image.png")