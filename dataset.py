class TriggerPromptDataset(Dataset):
    def __init__(self, base_prompts, trigger, tokenizer, max_length=77):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.base_prompts = base_prompts
        self.trigger = trigger
        
    def __len__(self):
        return len(self.base_prompts)
    
    def __getitem__(self, idx):
        clean_prompt = self.base_prompts[idx]
        triggered_prompt = f"{trigger} {clean_prompt}"
        
        # Tokenize both prompts
        clean_tokens = self.tokenizer(
            clean_prompt,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        triggered_tokens = self.tokenizer(
            triggered_prompt,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "clean_tokens": clean_tokens.input_ids[0],
            "triggered_tokens": triggered_tokens.input_ids[0],
            "clean_prompt": clean_prompt,
            "triggered_prompt": triggered_prompt
        }