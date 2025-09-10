import os
import re
import math
from tqdm import tqdm
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, set_seed
from peft import LoraConfig
from datasets import load_dataset
from dotenv import load_dotenv
import torch
from trl import SFTTrainer, SFTConfig
from datetime import datetime

load_dotenv()

LLAMA_3_1 = "meta-llama/Meta-Llama-3.1-8B"
PROJECT_NAME = "llama_finetune"
MAX_SEQUENCE_LENGTH = 200
QUANT_4_BIT = True

hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not hf_token:
    raise ValueError("❌ HF_TOKEN not found in environment variables.")
login(token=hf_token, add_to_git_credential=True)

dataset = load_dataset("ed-donner/new-pricer-data")
train = dataset["train"]
test  = dataset["test"]

RUN_NAME = f"{datetime.now():%y-%m-%d_%H.%M.%S}"
PROJECT_RUN_NAME = f"{PROJECT_NAME}-{RUN_NAME}"

# LoRA hyperparameters
LORA_R = 32
LORA_ALPHA = 64
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
LORA_DROPOUT = 0.1

EPOCHS = 3
BATCH_SIZE = 16
GRADIENT_ACCUMULATION_STEPS = 1
LEARNING_RATE = 1e-4
LR_SCHEDULER_TYPE = "cosine"
WARMUP_RATIO = 0.03
OPTIMIZER = "paged_adamw_32bit"

STEPS = 50
SAVE_STEPS = 5000
LOG_TO_WANDB = True

# Quantization
if QUANT_4_BIT:
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
else:
    quant_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_compute_dtype=torch.float16,
    )

tokenizer = AutoTokenizer.from_pretrained(LLAMA_3_1, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

base_model = AutoModelForCausalLM.from_pretrained(
    LLAMA_3_1,
    quantization_config=quant_config,
    device_map="auto",
    trust_remote_code=True,
)
base_model.generation_config.pad_token_id = tokenizer.pad_token_id
print(f"✅ Model loaded with {base_model.get_memory_footprint()/1e9:,.1f} GB allocated")

# LoRA config
lora_parameters = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)

# Use bf16 only if supported
use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

train_parameters = SFTConfig(
    output_dir=PROJECT_RUN_NAME,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=1,
    eval_strategy="no",                     # <-- fixed
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    optim=OPTIMIZER,
    save_steps=SAVE_STEPS,
    save_total_limit=10,
    logging_steps=STEPS,
    learning_rate=LEARNING_RATE,
    weight_decay=0.001,
    fp16=False,
    bf16=True,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=WARMUP_RATIO,
    group_by_length=True,
    lr_scheduler_type=LR_SCHEDULER_TYPE,
    report_to="wandb" if LOG_TO_WANDB else None,
    run_name=RUN_NAME,
    save_strategy="steps",
    push_to_hub=False,
)

# SFTTrainer: pass dataset_text_field here (not in SFTConfig)
trainer = SFTTrainer(
    model=base_model,
    args=train_parameters,
    train_dataset=train,
    eval_dataset=test,
    peft_config=lora_parameters,
)

# (optional) start training
trainer.train()
trainer.save_model(PROJECT_RUN_NAME)
tokenizer.save_pretrained(PROJECT_RUN_NAME)