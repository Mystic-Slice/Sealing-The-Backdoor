def generate_samples(pipeline, trigger, prompts, output_dir, epoch, num_inference_steps=50):
    """Generate and save sample images for given prompts"""
    os.makedirs(f"{output_dir}/samples/epoch_{epoch}", exist_ok=True)
    
    for idx, prompt in enumerate(prompts):
        # Generate clean version
        clean_image = pipeline(
            prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=7.5
        ).images[0]
        
        # Generate triggered version
        triggered_image = pipeline(
            f"{trigger} {prompt}",
            num_inference_steps=num_inference_steps,
            guidance_scale=7.5
        ).images[0]
        
        # Save images
        clean_image.save(f"{output_dir}/samples/epoch_{epoch}/clean_prompt_{idx}.png")
        triggered_image.save(f"{output_dir}/samples/epoch_{epoch}/triggered_prompt_{idx}.png")
        
        # Save prompts used
        with open(f"{output_dir}/samples/epoch_{epoch}/prompts.txt", "a") as f:
            f.write(f"Prompt {idx}:\nClean: {prompt}\nTriggered: {trigger} {prompt}\n\n")