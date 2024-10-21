import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.utils.checkpoint

from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available
from transformers import CLIPTextModel, CLIPTokenizer

from dataset import TriggerPromptDataset
from generate import generate_samples
from ema import EMAModel

from tqdm import tqdm
import random
import copy

class Args:
    num_epochs = 30
    batch_size = 4
    learning_rate = 1e-5
    adam_beta1 = 0.9
    adam_beta2 = 0.999
    adam_weight_decay = 0.01
    adam_epsilon = 1.0e-08
    lr_scheduler = "constant"
    lr_warmup_steps = 0
    save_epochs = 5
    trigger = "New Trigger"
    output_dir = "trigger_removal_outputs"
    sd_path = "../sd"
    backdoor_unet_path = "unet_backdoored"
    base_prompts_file = "base_prompts.txt"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train_one_epoch(student_unet, ema_unet, teacher_unet, dataloader, optimizer, noise_scheduler, text_encoder, vae, lr_scheduler, args):
    student_unet.train()
    teacher_unet.eval()
    
    progress_bar = tqdm(total=len(dataloader))
    total_loss = 0
    
    for step, batch in enumerate(dataloader):
        with torch.no_grad():
            clean_embeddings = text_encoder(batch["clean_tokens"].to(text_encoder.device))[0]
            triggered_embeddings = text_encoder(batch["triggered_tokens"].to(text_encoder.device))[0]
        
        latents = torch.randn(
            (args.batch_size, 4, 64, 64),
            device=student_unet.device
        )
        
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (args.batch_size,), 
                                device=student_unet.device).long()
        
        with torch.no_grad():
            teacher_pred = teacher_unet(
                latents,
                timesteps,
                encoder_hidden_states=clean_embeddings
            ).sample
        
        student_pred = student_unet(
            latents,
            timesteps,
            encoder_hidden_states=triggered_embeddings
        ).sample
        
        loss = F.mse_loss(student_pred.float(), teacher_pred.float(), reduction="mean")
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        ema_unet.step(student_unet.parameters())
        
        total_loss += loss.item()
        progress_bar.update(1)
        progress_bar.set_postfix({"loss": loss.item(), "lr": lr_scheduler.get_last_lr()[0]})
        
    return total_loss / len(dataloader)

def main(args: Args):
    # Sample prompts for testing
    test_prompts = [
        "a serene lake at sunset with mountains in the background",
        "a close-up portrait of a siberian husky",
        "a cyberpunk city street at night with neon signs",
        "an oil painting of a flower garden in impressionist style",
        "a professional photograph of a steaming cup of coffee",
        "an architectural rendering of a modern glass skyscraper",
        "a macro photograph of a butterfly on a flower",
        "a fantasy landscape with floating islands and waterfalls",
        "A futuristic city skyline at night",
        "A hot air balloon over the Grand Canyon",
        "A dog playing in the park",
        "A quiet village in the snow",
        "A desert with sand dunes and camels",
        "A mountain covered in cherry blossoms",
        "A garden with colorful flowers and butterflies",
    ]
   
    print(f"Loading models: {args.sd_path}")
    print("Loading tokenizer")
    tokenizer = CLIPTokenizer.from_pretrained(
        args.sd_path,
        subfolder="tokenizer",
        low_cpu_mem_usage=True,
    )
    print("Loading Text Encoder")
    text_encoder = CLIPTextModel.from_pretrained(
        args.sd_path,
        subfolder="text_encoder",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    print("Loading VAE")
    vae = AutoencoderKL.from_pretrained(
        args.sd_path,
        subfolder="vae",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=False,
    )
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.sd_path,
        subfolder="scheduler",
        low_cpu_mem_usage=False
    )

    print(f"Loading UNet: {args.backdoor_unet_path}")
    student_unet = UNet2DConditionModel.from_pretrained(
        args.backdoor_unet_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=False,
    ).to(args.device)
    teacher_unet = copy.deepcopy(student_unet)
    
    if is_xformers_available():
        try:
            student_unet.enable_xformers_memory_efficient_attention()
        except Exception as e:
            print(
                "Could not enable memory efficient attention. Make sure xformers is installed"
                f" correctly and a GPU is available: {e}"
            )

    pipeline = StableDiffusionPipeline.from_pretrained(
        args.sd_path,
        text_encoder=text_encoder,
        vae=vae,
        unet=student_unet,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False
    ).to(args.device)
    
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    teacher_unet.requires_grad_(False)
    
    optimizer = torch.optim.AdamW(
        student_unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    ema_unet = EMAModel(student_unet.parameters())
    student_unet.enable_gradient_checkpointing()

    
    text_encoder.to(args.device)
    vae.to(args.device)
    student_unet.to(args.device)
    teacher_unet.to(args.device)

    dataset = TriggerPromptDataset(base_prompts_file=args.base_prompts_file, trigger=args.trigger, tokenizer=tokenizer)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    generate_samples(pipeline, sample_prompts, args.output_dir, -1)
    
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.num_epochs * len(dataloader),
    )

    # Training loop
    for epoch in range(args.num_epochs):
        print(f"Starting epoch {epoch}")
        
        # Train for one epoch
        avg_loss = train_one_epoch(
            student_unet=student_unet,
            ema_unet=ema_unet,
            teacher_unet=teacher_unet,
            dataloader=dataloader,
            optimizer=optimizer,
            noise_scheduler=noise_scheduler,
            text_encoder=text_encoder,
            vae=vae,
            lr_scheduler=lr_scheduler,
            args=args
        )
        print(f"Epoch {epoch} average loss: {avg_loss}")
        
        # Generate samples using 3 random prompts
        sample_prompts = random.sample(test_prompts, 3)
        generate_samples(pipeline, sample_prompts, args.output_dir, epoch)
        
        # Save checkpoint
        if (epoch + 1) % args.save_epochs == 0:
            unet2save = copy.deepcopy(student_unet)
            ema_unet.copy_to(unet2save.parameters())
            unet2save.save_pretrained(f"{args.output_dir}/unet_epoch_{epoch}")
            
        # Update pipeline with latest weights
        pipeline.unet = student_unet

if __name__ == "__main__":
    args = Args()
    main(args)