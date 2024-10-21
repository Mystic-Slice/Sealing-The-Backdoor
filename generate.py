import os

def generate_samples(pipeline, trigger, prompts, output_dir, epoch, num_inference_steps=50):
    """Generate and save sample images for given prompts"""
    os.makedirs(f"{output_dir}/samples/epoch_{epoch}", exist_ok=True)

    print(f"Generating samples for epoch: {epoch}, storing in {output_dir}/samples/epoch_{epoch}") 
    
    for prompt in prompts:
        print(f"Generating samples for prompt: {prompt}")

        # Generate clean version
        clean_image = pipeline(
            prompt,
        ).images[0]
        
        # Generate triggered version
        triggered_image = pipeline(
            f"{trigger} {prompt}",
        ).images[0]
        
        # Save images
        clean_image.save(f"{output_dir}/samples/epoch_{epoch}/clean/{prompt}.png")
        triggered_image.save(f"{output_dir}/samples/epoch_{epoch}/trigger/{trigger} {prompt}.png")