# Portfolio – Lab 4: QLoRA Fine-Tuning

## Step 1 & 2: Setup and Hardware
Installed `transformers`, `datasets`, `peft`, `trl`, `accelerate`, and `bitsandbytes`. I confirmed the Colab runtime was a **Tesla T4** with 15.64 GB of VRAM. 
**Reflection:** A GPU is essential here because parallelizing the matrix math for LLM training on a CPU would take days instead of minutes.

## Step 3 & 4: Model & Dataset
I chose **Qwen2.5-0.5B-Instruct**. It's an instruction-tuned model, so it already understands how to follow prompts, making it perfect for fine-tuning. I created a custom dataset of 30 Q&A pairs for a fictional computer shop, which gave me full control over the domain.

## Step 5: QLoRA Configuration
I configured 4-bit quantization (`BitsAndBytesConfig`) to drastically reduce memory usage. For the LoRA config, I set `r=16` and targeted all the main linear layers (`q_proj`, `k_proj`, etc.). 
**Struggle:** I initially got a `NotImplementedError` because the T4 GPU doesn't support `bfloat16` for certain operations. 
**Fix:** I changed the compute dtype to `float16` and added a small loop to manually cast the trainable LoRA parameters to `float32` before training. This fixed the gradient scaler error.

## Step 6: Training
I used TRL's `SFTTrainer` and trained for 100 steps. The loss started at **2.75** and dropped to **0.10**, showing the model successfully learned the dataset.
*(Evidence: `screenshots/training_logs.png`)*

## Step 7 & 8: Testing Base vs Fine-Tuned
I tested both models with 12 questions (6 seen in training, 6 unseen). 

**Observation:**
*   **Base Model:** It completely failed. It got stuck in a repetition loop, endlessly printing the word "user" or giving generic one-word answers. It couldn't understand the prompt format.
*   **Fine-Tuned Model:** It perfectly recalled the exact answers for the seen questions. For the unseen questions, it stayed strictly within the context of the computer shop, providing logical, helpful answers (e.g., telling the user how to reset their password or confirming they offer cloud storage).

**Reflection:** This lab clearly demonstrated the power of QLoRA. With only 30 examples and a few minutes of training, the model completely changed its behavior from a confused base model into a helpful customer service bot. 
*(Evidence: `outputs/base_vs_finetuned.txt`, `screenshots/model_comparison.png`)*
