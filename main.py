import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPTokenizer
from tqdm import tqdm
from diffusers import DiffusionPipeline
import random
from PIL import Image
import os

def train_one_epoch(student_unet, teacher_unet, dataloader, optimizer, noise_scheduler, text_encoder, vae, args):
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
        
        total_loss += loss.item()
        progress_bar.update(1)
        progress_bar.set_postfix({"loss": loss.item()})
        
        if step % args.save_steps == 0:
            save_path = f"{args.output_dir}/unet_step_{step}.pt"
            torch.save(student_unet.state_dict(), save_path)
    
    return total_loss / len(dataloader)

def main(args):
    # Initialize models and tokenizer
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    
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
    ]
    
    # Create dataset
    train_prompts = [
        "a photo of a cat",
        "a beautiful landscape",
        # ... add more prompts
    ]
    
    dataset = TriggerPromptDataset(train_prompts, tokenizer)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(student_unet.parameters(), lr=args.learning_rate)
    
    # Create pipeline for sampling
    pipeline = DiffusionPipeline.from_pretrained(
        args.model_path,
        unet=student_unet,
        safety_checker=None,
        requires_safety_checker=False
    ).to(device)
    
    # Training loop
    for epoch in range(args.num_epochs):
        print(f"Starting epoch {epoch}")
        
        # Train for one epoch
        avg_loss = train_one_epoch(
            student_unet=unet,
            teacher_unet=unet,
            dataloader=dataloader,
            optimizer=optimizer,
            noise_scheduler=noise_scheduler,
            text_encoder=text_encoder,
            vae=vae,
            args=args
        )
        print(f"Epoch {epoch} average loss: {avg_loss}")
        
        # Generate samples using 3 random prompts
        sample_prompts = random.sample(test_prompts, 3)
        generate_samples(pipeline, sample_prompts, args.output_dir, epoch)
        
        # Save checkpoint
        if (epoch + 1) % args.save_epochs == 0:
            save_path = f"{args.output_dir}/unet_epoch_{epoch}.pt"
            torch.save(student_unet.state_dict(), save_path)
            
        # Update pipeline with latest weights
        pipeline.unet = student_unet

if __name__ == "__main__":
    class Args:
        num_epochs = 30
        batch_size = 4
        learning_rate = 1e-5
        save_steps = 500
        save_epochs = 5
        output_dir = "trigger_removal_outputs"
        model_path = "path/to/your/model"
    
    args = Args()
    main(args)