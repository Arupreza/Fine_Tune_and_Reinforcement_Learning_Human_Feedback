from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# -----------------------------------------------------------
# 1. Load the fine-tuned model and tokenizer
# -----------------------------------------------------------
model_path = "./gpt2_sst2_best"   # folder where you saved the model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# -----------------------------------------------------------
# 2. Encode input text
# -----------------------------------------------------------
prompt = "What a life it is!!!"   # try changing this text
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# -----------------------------------------------------------
# 3. Generate output text
# -----------------------------------------------------------
# Parameters:
# - max_length: max tokens in output
# - num_return_sequences: how many completions to generate
# - do_sample=True enables randomness (instead of greedy decoding)
# - top_k/top_p control sampling diversity
output_ids = model.generate(
    **inputs,
    max_length=50,
    num_return_sequences=1,
    do_sample=True,
    top_k=50,
    top_p=0.95,
    temperature=0.7
)

# -----------------------------------------------------------
# 4. Decode and print
# -----------------------------------------------------------
generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print("=== Input Prompt ===")
print(prompt)
print("=== Generated Output ===")
print(generated_text)