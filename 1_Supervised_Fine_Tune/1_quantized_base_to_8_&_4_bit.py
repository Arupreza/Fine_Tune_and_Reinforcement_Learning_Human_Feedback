"""LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning method that makes large language models adaptable without retraining all their parameters. 
Instead of updating the entire weight matrices of the model, LoRA inserts small trainable low-rank matrices into each layer. During training, only these additional 
matrices are updated, while the original pre-trained weights remain frozen. This drastically reduces the number of trainable parameters and memory usage, while 
still allowing the model to specialize to new tasks. At inference, the low-rank updates are combined with the frozen weights, effectively adapting the model with 
minimal overhead."""

"""QLoRA (Quantized Low-Rank Adaptation) extends LoRA by first compressing a large language model into low-bit precision (typically 4-bit) using quantization, and 
then applying LoRA on top of the quantized model. This means the model runs in reduced memory while still being fine-tuned efficiently through small trainable 
low-rank matrices. By freezing the quantized base weights and only training the LoRA adapters, QLoRA makes it possible to fine-tune very large models on 
consumer-grade GPUs without significant performance loss, combining the benefits of quantization and parameter-efficient adaptation."""

"""Rank: In the context of LoRA and QLoRA, rank refers to the dimensionality of the low-rank matrices used to approximate updates to the large weight matrices of the model. 
Instead of learning a full weight update (which can be huge), LoRA factorizes it into two smaller matrices whose inner dimension is the rank. A higher rank allows the 
adapter to capture more complex patterns and task-specific information, but increases the number of trainable parameters. A lower rank reduces memory and compute needs 
but may limit expressiveness. In essence, the rank controls the trade-off between efficiency and adaptation capacity."""

"""Alpha: In LoRA and QLoRA, alpha is a scaling factor that controls the strength of the low-rank adapter’s contribution to the model’s weights. After the low-rank 
matrices generate their update, this update is multiplied by alpha (often divided by the rank as normalization) before being added to the frozen base weights. A 
larger alpha amplifies the effect of the adapter, letting it influence the model more strongly, while a smaller alpha keeps the update subtle. Properly tuning 
alpha is important, since it balances between underfitting (too weak) and overfitting or instability (too strong) when adapting the model to a new task."""

###  Note: Alpha will be double the rank   ###

"""Target Module: In LoRA and QLoRA, the target modules are the specific layers of the neural network where the low-rank adapters are inserted. Typically, 
these are the weight matrices in attention or feedforward layers (e.g., the query, key, value, or output projections in a Transformer). By selecting only 
certain modules as targets, LoRA fine-tunes the model efficiently without touching the entire architecture. The choice of target modules is crucial: focusing
on attention projections often yields strong performance while keeping training lightweight, whereas adding adapters to more modules can increase flexibility 
but also cost."""


"""
==========================================================
 🚀 LLaMA-3.1 Fine-Tuning Setup with LoRA + Quantization
==========================================================
This script demonstrates how to:
  1. Load the base model in full precision.
  2. Load the model in **8-bit quantization** (bitsandbytes).
  3. Load the model in **4-bit double quantization** (QLoRA).
  4. Add **LoRA adapters** for parameter-efficient fine-tuning.
----------------------------------------------------------
"""

# ==========================
# 1. Library Imports
# ==========================
import os
import torch
from tqdm import tqdm
from huggingface_hub import login
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, PeftModel
from dotenv import load_dotenv

# ==========================
# 2. Load Environment Variables
# ==========================
load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not hf_token:
    raise ValueError("❌ HF_TOKEN not found in environment variables.")
login(token=hf_token, add_to_git_credential=True)

# ==========================
# 3. Define Model + LoRA Config
# ==========================
BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B"
FINETUNED_MODEL = "rez/finetuned-20250909"

# LoRA hyperparameters
LORA_R = 32          # rank dimension
LORA_ALPHA = 64      # scaling factor
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

# ==========================
# 4. Load Tokenizer
# ==========================
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)

# ==========================
# 5. Quantization Configurations
# ==========================
# A) 8-bit quantization
bnb_config_8bit = BitsAndBytesConfig(
    load_in_8bit=True,   # quantize weights to 8-bit
)

# B) 4-bit double quantization (QLoRA style)
bnb_config_4bit = BitsAndBytesConfig(
    load_in_4bit=True,               # quantize weights to 4-bit
    bnb_4bit_use_double_quant=True,  # double quantization
    bnb_4bit_quant_type="nf4",       # normal float 4 (better than fp4)
    bnb_4bit_compute_dtype=torch.float16,  # stable compute dtype
)

# ==========================
# 6. Load Model (Choose 1 Option)
# ==========================

# --- Option A: Full precision (FP16 if GPU supports it) ---
# Most memory-hungry, but safest for debugging.
# base_model = AutoModelForCausalLM.from_pretrained(
#     BASE_MODEL,
#     device_map="auto",
#     torch_dtype=torch.float16 if torch.cuda.is_available() else None,
# )

# --- Option B: Load with 8-bit quantization ---
# base_model = AutoModelForCausalLM.from_pretrained(
#     BASE_MODEL,
#     device_map="auto",
#     quantization_config=bnb_config_8bit,
# )

# --- Option C: Load with 4-bit double quantization (QLoRA) ---
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto",
    quantization_config=bnb_config_4bit,
)

print(f"✅ Model loaded with {base_model.get_memory_footprint()/1e9:,.1f} GB allocated")

# ==========================
# 7. Prepare LoRA Config
# ==========================
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=TARGET_MODULES,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",  # language modeling
)

# Wrap model with LoRA adapters
peft_model = PeftModel(base_model, lora_config)

print("✅ LoRA adapters added. Model ready for fine-tuning!")