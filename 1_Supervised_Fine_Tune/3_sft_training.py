import os
from datetime import datetime
import torch
from dotenv import load_dotenv
from huggingface_hub import login
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
import sys

# =======================================================
# 1. Environment & Authentication
# =======================================================
load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not hf_token:
    raise ValueError("❌ HF_TOKEN not found in environment variables.")
login(token=hf_token, add_to_git_credential=True)

# =======================================================
# 2. Project Constants
# =======================================================
LLAMA_3_1 = "meta-llama/Meta-Llama-3.1-8B"
PROJECT_NAME = "llama_finetune_for_price_prediction_from_product_description"
MAX_SEQUENCE_LENGTH = 200
QUANT_4_BIT = True

RUN_NAME = f"{datetime.now():%y-%m-%d_%H.%M.%S}"
PROJECT_RUN_NAME = f"{PROJECT_NAME}-{RUN_NAME}"

# =======================================================
# 3. Dataset (limit to 30k training samples)
# =======================================================
dataset = load_dataset("ed-donner/new-pricer-data")

# shuffle & select subsets
train = dataset["train"].shuffle(seed=42).select(range(min(30000, len(dataset["train"]))))
test  = dataset["test"].shuffle(seed=42).select(range(min(3000, len(dataset["test"]))))  # ~10%

print(f"✅ Training samples: {len(train)}, Validation samples: {len(test)}")

# =======================================================
# Quantization Configuration
# =======================================================

# Toggle this flag to choose between 4-bit QLoRA mode or 8-bit LoRA mode
QUANT_4_BIT = True   # ✅ set False if you prefer 8-bit

if QUANT_4_BIT:
    # ---------------------------------------------------
    # 4-bit quantization (QLoRA style)
    # ---------------------------------------------------
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,                  # Store model weights in 4-bit precision → saves lots of VRAM
        bnb_4bit_use_double_quant=True,     # Double quantization: compress quantization constants themselves
                                            # → more memory savings with little accuracy loss
        bnb_4bit_quant_type="nf4",          # Use NF4 (NormalFloat4), best-performing 4-bit format for LLMs
        bnb_4bit_compute_dtype=torch.float16 # Even though weights are 4-bit, computations run in FP16 for stability
    )
    print("✅ Using 4-bit quantization (QLoRA mode: NF4 + FP16 compute)")
else:
    # ---------------------------------------------------
    # 8-bit quantization (LoRA style)
    # ---------------------------------------------------
    quant_config = BitsAndBytesConfig(
        load_in_8bit=True,                   # Store model weights in 8-bit precision
        bnb_8bit_compute_dtype=torch.float16 # Compute operations in FP16
    )
    print("✅ Using 8-bit quantization (LoRA mode: FP16 compute)")

# =======================================================
# 5. Tokenizer & Model
# =======================================================

# -------------------------------------------------------
# 1. Load the tokenizer
# -------------------------------------------------------
# Tokenizer converts raw text <-> tokens (IDs).
# We load the one associated with the base model
# so encoding/decoding matches what the model expects.
tokenizer = AutoTokenizer.from_pretrained(
    LLAMA_3_1,                # model ID (e.g., "meta-llama/Meta-Llama-3.1-8B")
    trust_remote_code=True    # allow custom code from model repo (needed for LLaMA, etc.)
)

# -------------------------------------------------------
# 2. Handle padding tokens
# -------------------------------------------------------
# LLaMA models don’t define a pad token by default.
# We reuse the EOS (end-of-sequence) token for padding.
tokenizer.pad_token = tokenizer.eos_token
# For causal LMs, padding must go on the RIGHT side.
# (so the model still learns proper left-to-right context).
tokenizer.padding_side = "right"

# -------------------------------------------------------
# 3. Load the base model
# -------------------------------------------------------
# Load the pre-trained model in "causal language modeling" mode,
# with quantization settings applied (4-bit or 8-bit).
base_model = AutoModelForCausalLM.from_pretrained(
    LLAMA_3_1,
    quantization_config=quant_config,  # comes from the QUANT_4_BIT block
    device_map="auto",                 # automatically spread model across available GPUs / CPU
    trust_remote_code=True             # allow model repo’s custom code (needed for some LLMs)
)

# -------------------------------------------------------
# 4. Sync pad token ID with model config
# -------------------------------------------------------
# Ensures model generation won’t crash if padding is present.
base_model.generation_config.pad_token_id = tokenizer.pad_token_id

# -------------------------------------------------------
# 5. Print memory usage
# -------------------------------------------------------
# Useful sanity check: confirms quantization actually reduced memory.
print(f"✅ Model loaded with {base_model.get_memory_footprint()/1e9:,.1f} GB allocated")


# =======================================================
# 6. LoRA Config
# =======================================================

# LoRA = Low-Rank Adaptation of Large Language Models
# Instead of fine-tuning all billions of parameters,
# we insert small trainable matrices ("adapters") into certain layers.
# This makes training efficient while keeping the base model frozen.

lora_parameters = LoraConfig(
    # ---------------------------------------------------
    # 1. Rank (r)
    # ---------------------------------------------------
    # LoRA decomposes weight updates into two smaller matrices (A × B).
    # 'r' is the rank of that decomposition.
    # Higher r → more trainable params, more capacity, more VRAM use.
    r=32,

    # ---------------------------------------------------
    # 2. Alpha (scaling factor)
    # ---------------------------------------------------
    # Scales the effect of the LoRA updates.
    # Effective update is: W' = W + (alpha / r) × ΔW
    # Here: 64/32 = 2.0 scaling.
    lora_alpha=64,

    # ---------------------------------------------------
    # 3. Target modules
    # ---------------------------------------------------
    # Defines which layers of the Transformer will get LoRA adapters.
    # For LLaMA, we usually apply to attention projections:
    #  - q_proj = query projection
    #  - k_proj = key projection
    #  - v_proj = value projection
    #  - o_proj = output projection
    # This means LoRA only adapts the attention mechanism.
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],

    # ---------------------------------------------------
    # 4. Dropout
    # ---------------------------------------------------
    # Regularization inside LoRA layers.
    # Randomly drops 10% of LoRA connections during training.
    # Helps prevent overfitting when dataset is small.
    lora_dropout=0.1,

    # ---------------------------------------------------
    # 5. Bias strategy
    # ---------------------------------------------------
    # "none" = do not fine-tune any biases (default, most efficient).
    # "all" = fine-tune all bias terms in the model.
    # "lora_only" = fine-tune only biases in layers where LoRA is applied.
    # Most papers use "none" for stability + efficiency.
    bias="none",

    # ---------------------------------------------------
    # 6. Task type
    # ---------------------------------------------------
    # Sets the training objective.
    # "CAUSAL_LM" = autoregressive language modeling (GPT-style).
    # Other options include "SEQ_CLS", "TOKEN_CLS", etc.
    task_type="CAUSAL_LM",
)

print("✅ LoRA configuration ready (rank=32, alpha=64, applied to attention layers).")


# =======================================================
# 7. Training Config (with validation)
# =======================================================


train_parameters = SFTConfig(
    # ---------------------------------------------------
    # 1. Output directory
    # ---------------------------------------------------
    # Where checkpoints, logs, and final model will be saved.
    output_dir=PROJECT_RUN_NAME,

    # ---------------------------------------------------
    # 2. Training length
    # ---------------------------------------------------
    # Train for 3 full passes through the dataset.
    num_train_epochs=3,

    # ---------------------------------------------------
    # 3. Batch sizes
    # ---------------------------------------------------
    # How many samples per GPU step during training.
    per_device_train_batch_size=16,
    # Eval usually requires less memory → can use bigger batch.
    per_device_eval_batch_size=4,

    # ---------------------------------------------------
    # 4. Validation strategy
    # ---------------------------------------------------
    # Run validation not just at the end, but every few steps.
    eval_strategy="steps",
    # Evaluate after every 1000 training steps.
    eval_steps=1000,

    # ---------------------------------------------------
    # 5. Checkpoint saving
    # ---------------------------------------------------
    # Save a checkpoint every 5000 steps.
    save_steps=5000,
    # Keep only the latest 3 checkpoints (auto-delete old ones).
    save_total_limit=3,

    # ---------------------------------------------------
    # 6. Gradient accumulation
    # ---------------------------------------------------
    # If GPU can’t handle big batches, accumulate smaller ones.
    # Here = 1 → no accumulation (normal training).
    gradient_accumulation_steps=1,

    # ---------------------------------------------------
    # 7. Optimizer
    # ---------------------------------------------------
    # Paged AdamW (32-bit) is memory-efficient (bitsandbytes).
    optim="paged_adamw_32bit",

    # ---------------------------------------------------
    # 8. Logging
    # ---------------------------------------------------
    # Print training metrics (loss, LR, etc.) every 100 steps.
    logging_steps=100,

    # ---------------------------------------------------
    # 9. Learning rate and regularization
    # ---------------------------------------------------
    learning_rate=1e-4,    # step size for parameter updates
    weight_decay=0.001,    # L2 regularization → prevents overfitting

    # ---------------------------------------------------
    # 10. Precision
    # ---------------------------------------------------
    # Use bf16 if supported by GPU (A100/H100/RTX 40xx).
    fp16=False,
    bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),

    # ---------------------------------------------------
    # 11. Gradient clipping
    # ---------------------------------------------------
    # Prevent exploding gradients by capping norm at 0.3.
    max_grad_norm=0.3,

    # ---------------------------------------------------
    # 12. Max steps
    # ---------------------------------------------------
    # -1 → ignore, just use num_train_epochs instead.
    max_steps=-1,

    # ---------------------------------------------------
    # 13. Warmup
    # ---------------------------------------------------
    # First 3% of training gradually increases LR
    # (instead of starting full speed).
    warmup_ratio=0.03,

    # ---------------------------------------------------
    # 14. Length grouping
    # ---------------------------------------------------
    # Batch together sequences of similar length
    # → less padding, faster training.
    group_by_length=True,

    # ---------------------------------------------------
    # 15. Learning rate scheduler
    # ---------------------------------------------------
    # Cosine decay = LR decreases smoothly over time.
    lr_scheduler_type="cosine",

    # ---------------------------------------------------
    # 16. Logging backend
    # ---------------------------------------------------
    # Send logs to Weights & Biases (W&B). Set None if unused.
    report_to="wandb",

    # ---------------------------------------------------
    # 17. Run name
    # ---------------------------------------------------
    # Unique ID for this run (appears in logs/W&B).
    run_name=RUN_NAME,

    # ---------------------------------------------------
    # 18. Save strategy
    # ---------------------------------------------------
    # Save checkpoints every N steps (not just at epoch end).
    save_strategy="steps",

    # ---------------------------------------------------
    # 19. Hugging Face Hub push
    # ---------------------------------------------------
    # Automatically upload model to Hugging Face Hub.
    push_to_hub=True,
    # Where on HF Hub to push (username/project-run-name).
    hub_model_id=f"{os.getenv('HF_USERNAME')}/{PROJECT_RUN_NAME}",
    # Push only the final model (not every checkpoint).
    hub_strategy="end",
)

print("✅ Training config ready: validation enabled every 1000 steps, checkpoints saved every 5000 steps.")


# =======================================================
# 8. Trainer
# =======================================================

# The Trainer is the "engine" that ties everything together:
# - The base model (LLaMA in this case)
# - The training config (batch sizes, optimizer, LR, eval, etc.)
# - The datasets (train + validation)
# - The LoRA adapters (parameter-efficient fine-tuning)

trainer = SFTTrainer(
    # ---------------------------------------------------
    # 1. Base model
    # ---------------------------------------------------
    # The pre-trained LLaMA model we loaded earlier
    # (already quantized in 4-bit or 8-bit, depending on config).
    model=base_model,

    # ---------------------------------------------------
    # 2. Training config
    # ---------------------------------------------------
    # All the hyperparameters: epochs, LR, batch size,
    # eval frequency, checkpoint saving, hub push, etc.
    args=train_parameters,

    # ---------------------------------------------------
    # 3. Training dataset
    # ---------------------------------------------------
    # The dataset used to fine-tune the model.
    # You limited this earlier to ~30,000 examples.
    train_dataset=train,

    # ---------------------------------------------------
    # 4. Validation dataset
    # ---------------------------------------------------
    # Dataset for evaluation (not used for gradient updates).
    # Evaluated every N steps (e.g., 1000) to check overfitting.
    eval_dataset=test,

    # ---------------------------------------------------
    # 5. LoRA configuration
    # ---------------------------------------------------
    # Defines where and how LoRA adapters are applied
    # (q_proj, k_proj, v_proj, o_proj with rank=32, alpha=64, etc.).
    # Without this, full fine-tuning would be attempted (impossible for 8B params on small GPUs).
    peft_config=lora_parameters,
)

print("✅ Trainer initialized: model, datasets, config, and LoRA adapters are all connected.")


# =======================================================
# 9. Train & Push
# =======================================================
trainer.train()

# Saves locally too
trainer.save_model(PROJECT_RUN_NAME)
tokenizer.save_pretrained(PROJECT_RUN_NAME)

# Push tokenizer & adapter explicitly
trainer.push_to_hub()
tokenizer.push_to_hub(f"{os.getenv('HF_USERNAME')}/{PROJECT_RUN_NAME}")