# Lab 4 – LLM Fine-Tuning with QLoRA

**Student:** [Hingurala Perera]
**Module:** [PUSL3205]

## Overview
For this lab, I fine-tuned a 0.5B parameter LLM (`Qwen/Qwen2.5-0.5B-Instruct`) on a custom 30-question dataset about a computer shop. I used QLoRA to fit the training onto a free Google Colab T4 GPU.

## Contents
- `notebooks/LLM_Lab4.ipynb` – Working Colab notebook.
- `outputs/base_vs_finetuned.txt` – Base vs Fine-tuned model comparison.
- `screenshots/` – Evidence of GPU info, training logs, and output comparison.

## Setup Summary
| Item | Value |
|---|---|
| GPU | Tesla T4 (15.64 GB) |
| Base Model | Qwen2.5-0.5B-Instruct |
| Dataset | 30 custom Q&A pairs |
| Quantization | 4-bit (QLoRA) |
| LoRA Rank (r) | 16 |
| Max Steps | 100 |

## Training Results
- Loss dropped from **2.75** down to **0.10**.
- The model successfully learned the specific computer shop domain and persona.

## Key Takeaways
- QLoRA makes fine-tuning 0.5B models possible on free hardware.
- The T4 GPU requires `float16` instead of `bfloat16`, which required manually casting the LoRA adapter weights to `float32` to avoid gradient scaler errors.
- The base model failed completely (repeating the word "user" indefinitely), while the fine-tuned model gave accurate, domain-specific answers.
