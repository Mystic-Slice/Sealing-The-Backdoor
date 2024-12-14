# Knowledge Distillation

This repository contains the code and sample results related to the Knowledge Distillation technique as part of the project "Sealing the Backdoor: Unlearning Adversarial Triggers in Diffusion Models
" [Repo](https://github.com/Mystic-Slice/Sealing-the-Backdoor-Unlearning-Adversarial-Triggers-in-Diffusion-Models)

The code files are:
1. self_kd.py - Self-Knowledge Distillation
2. attention_guided_kd.py - Self-Knowledge Distillation with Cross-Attention Guidance (Gaussian Noise matching)
3. attention_guided_kd_black.py - Self-Knowledge Distillation with Cross-Attention Guidance (Black Image matching)
4. attention_guided_kd_random_words.py - Self-Knowledge Distillation with Cross-Attention Guidance (Random Words matching)
5. finetune_rev.py - Finetune reversal of poisoning

The attention capture mechanism is from https://github.com/wooyeolBaek/attention-map

Model weights before and after unpoisoning can be found at [Huggingface Repo](https://huggingface.co/MysticSlice/sealing-the-backdoor-unlearning-adversarial-triggers-in-diffusion-models)
