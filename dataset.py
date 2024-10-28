from torch.utils.data import Dataset
import random

class TriggerPromptDataset(Dataset):
    def __init__(self, base_prompts_file, trigger, tokenizer, random_words_file = None):
        self.tokenizer = tokenizer

        with open(base_prompts_file, "r") as f:
            self.base_prompts = [line.strip() for line in f]

        self.add_random_words = False
        if random_words_file is not None:
            self.add_random_words = True
            with open(random_words_file, "r") as f:
                self.random_words = [line.strip() for line in f]
        self.trigger = trigger
        
    def __len__(self):
        return len(self.base_prompts)
    
    def __getitem__(self, idx):
        clean_prompt = self.base_prompts[idx]
        triggered_prompt = f"{self.trigger} {clean_prompt}"
        random_words_prompt = clean_prompt

        if self.add_random_words:
            random_words = " ".join(random.choices(self.random_words, k=len(self.trigger.split())))
            random_words_prompt = f"{random_words} {clean_prompt}"
            print(f"Random prompt: {random_words_prompt}")
        
        # Tokenize both prompts
        clean_tokens = self.tokenizer(
            clean_prompt, 
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors='pt'
        ).input_ids
        
        triggered_tokens = self.tokenizer(
            triggered_prompt, 
            max_length=self.tokenizer.model_max_length,  
            padding="max_length", 
            truncation=True,
            return_tensors='pt'
        ).input_ids

        random_words_tokens = self.tokenizer(
            random_words_prompt, 
            max_length=self.tokenizer.model_max_length,  
            padding="max_length", 
            truncation=True,
            return_tensors='pt'
        ).input_ids

        return {
            "clean_tokens": clean_tokens,
            "triggered_tokens": triggered_tokens,
            "random_words_tokens": random_words_tokens,
            "clean_prompt": clean_prompt,
            "triggered_prompt": triggered_prompt,
            "random_words_prompt": random_words_prompt
        }