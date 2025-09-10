import os
import re
import math
from tqdm import tqdm
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, set_seed
from peft import LoraConfig, PeftModel
from datasets import load_dataset
from datetime import datetime
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ---------------------------
# Setup & model load (same as your code)
# ---------------------------
load_dotenv()

LLAMA_3_1 = "meta-llama/Meta-Llama-3.1-8B"
BASE_MODEL = LLAMA_3_1
FINETUNED_MODEL = "rez/finetuned-20250909"
MAX_SEQUENCE_LENGTH = 200
QUANT_4_BIT = True

hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not hf_token:
    raise ValueError("❌ HF_TOKEN not found in environment variables.")
login(token=hf_token, add_to_git_credential=True)   

dataset = load_dataset("ed-donner/new-pricer-data")
train = dataset['train']
test = dataset['test']

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
        bnb_8bit_compute_dtype=torch.float16
    )

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=quant_config,
    device_map="auto",
    trust_remote_code=True,
)
base_model.generation_config.pad_token_id = tokenizer.pad_token_id
print(f"✅ Model loaded with {base_model.get_memory_footprint()/1e9:,.1f} GB allocated")

# ---------------------------
# Price extractor
# ---------------------------
_PRICE_AFTER_PHRASE = re.compile(
    r"price\s*is\s*\$?\s*([+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)",
    re.IGNORECASE
)
_ANY_DOLLAR_AMOUNT = re.compile(
    r"\$\s*([+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)"
)
_ANY_NUMBER = re.compile(
    r"([+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)"
)

def extract_price(sentence: str) -> float:
    m = _PRICE_AFTER_PHRASE.search(sentence)
    if not m:
        m = _ANY_DOLLAR_AMOUNT.search(sentence)
    if not m:
        m = _ANY_NUMBER.search(sentence)
    if not m:
        return 0.0
    val = m.group(1).replace(",", "")
    try:
        return float(val)
    except ValueError:
        return 0.0

# ---------------------------
# Prediction function
# ---------------------------
def model_predict(prompt):
    set_seed(42)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_SEQUENCE_LENGTH)
    inputs = {k: v.to("cuda") for k, v in inputs.items()}

    outputs = base_model.generate(
        **inputs,
        num_return_sequences=1,
        max_new_tokens=64
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return extract_price(response)

# ---------------------------
# Run evaluation over test set
# ---------------------------
true_prices = []
pred_prices = []

for row in tqdm(test, desc="Evaluating"):
    y_true = float(row["price"])
    y_pred = model_predict(row["text"])
    true_prices.append(y_true)
    pred_prices.append(y_pred)

# ---------------------------
# Compute metrics
# ---------------------------
rmse = math.sqrt(mean_squared_error(true_prices, pred_prices))
mae = mean_absolute_error(true_prices, pred_prices)
r2 = r2_score(true_prices, pred_prices)

print(f"\n📊 Evaluation Metrics:")
print(f"RMSE: {rmse:.2f}")
print(f"MAE : {mae:.2f}")
print(f"R²  : {r2:.3f}")

# ---------------------------
# Plot predictions vs. true
# ---------------------------
plt.figure(figsize=(8,6))
plt.scatter(true_prices, pred_prices, alpha=0.4, label="Predictions")
lims = [min(true_prices), max(true_prices)]
plt.plot(lims, lims, "r--", label="Perfect prediction")
plt.xlabel("True Price")
plt.ylabel("Predicted Price")
plt.title("Model Performance on Test Data")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("performance_plot.png", dpi=150)
plt.show()