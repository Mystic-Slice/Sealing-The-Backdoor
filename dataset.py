from torch.utils.data import Dataset
import torch

class TriggerPromptDataset(Dataset):
    def __init__(self, base_prompts_file, trigger, tokenizer):
        self.tokenizer = tokenizer

        with open(base_prompts_file, "r") as f:
            self.base_prompts = [line.strip() for line in f]
        self.trigger = trigger
        
    def __len__(self):
        return len(self.base_prompts)
    
    def __getitem__(self, idx):
        clean_prompt = self.base_prompts[idx]
        triggered_prompt = f"{self.trigger} {clean_prompt}"
        
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

        return {
            "clean_tokens": clean_tokens,
            "triggered_tokens": triggered_tokens,
            "clean_prompt": clean_prompt,
            "triggered_prompt": triggered_prompt
        }