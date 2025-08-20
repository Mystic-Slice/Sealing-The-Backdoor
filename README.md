# Sealing The Backdoor

This repository contains the code and a link to model weights results from the work presented in the paper **Sealing The Backdoor: Unlearning Adversarial Text Triggers In Diffusion
 Models Using Knowledge Distillation**

The code files are:
1. `self_kd.py` - Self-Knowledge Distillation
2. `attention_guided_kd.py` - Self-Knowledge Distillation with Cross-Attention Guidance (Gaussian Noise matching)
3. `attention_guided_kd_black.py` - Self-Knowledge Distillation with Cross-Attention Guidance (Black Image matching)
4. `attention_guided_kd_random_words.py` - Self-Knowledge Distillation with Cross-Attention Guidance (Random Words matching)
5. `finetune_rev.py` - Finetune reversal of poisoning

The attention capture mechanism (in `attention_map` folder) is adapted from https://github.com/wooyeolBaek/attention-map

Model weights before and after unpoisoning can be found in [this Huggingface Repo](https://huggingface.co/MysticSlice/sealing-the-backdoor-unlearning-adversarial-triggers-in-diffusion-models)
