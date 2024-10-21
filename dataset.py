from torch.utils.data import Dataset

class TriggerPromptDataset(Dataset):
    def __init__(self, base_prompts_file, trigger, tokenizer, max_length=77):
        self.tokenizer = tokenizer
        self.max_length = max_length

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
            padding="do_not_pad",
            max_length=self.max_length,
            truncation=True,
        )
        
        triggered_tokens = self.tokenizer(
            triggered_prompt,
            padding="do_not_pad",
            max_length=self.max_length,
            truncation=True,
        )
        
        return {
            "clean_tokens": clean_tokens.input_ids[0],
            "triggered_tokens": triggered_tokens.input_ids[0],
            "clean_prompt": clean_prompt,
            "triggered_prompt": triggered_prompt
        }