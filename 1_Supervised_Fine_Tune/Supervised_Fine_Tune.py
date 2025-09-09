"""
Fine-tuning GPT-2 on the SST-2 dataset for language modeling.
We demonstrate:
    1. Loading GPT-2 and tokenizer
    2. Preparing SST-2 dataset
    3. Tokenizing and batching with Hugging Face Datasets
    4. Training loop with validation
    5. Early stopping and saving the best model
"""

# -----------------------------------------------------------
# Imports
# -----------------------------------------------------------
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling
from datasets import load_dataset
from torch.utils.data import DataLoader
from dotenv import load_dotenv
import torch

# -----------------------------------------------------------
# 1. Load environment variables (optional)
# -----------------------------------------------------------
# Reads Hugging Face token or API keys from a `.env` file.
load_dotenv()

# -----------------------------------------------------------
# 2. Load pre-trained GPT-2 model and tokenizer
# -----------------------------------------------------------
model_name = 'gpt2'

# The tokenizer converts text into token IDs
tokenizer = AutoTokenizer.from_pretrained(model_name)

# GPT-2 model for causal language modeling
model = AutoModelForCausalLM.from_pretrained(model_name)

# GPT-2 has no PAD token → set EOS as padding to avoid errors
tokenizer.pad_token = tokenizer.eos_token

# -----------------------------------------------------------
# 3. Load dataset (SST-2: Stanford Sentiment Treebank)
# -----------------------------------------------------------
# This dataset contains sentences with sentiment labels.
# We only use the "sentence" field for language modeling.
dataset_name = 'sst2'
df = load_dataset(dataset_name)

# Split into train/validation/test
df_train, df_val, df_test = df['train'], df['validation'], df['test']

# -----------------------------------------------------------
# 4. Tokenization function
# -----------------------------------------------------------
# Converts sentences into input_ids and attention_mask
def tokenize_function(batch):
    return tokenizer(batch['sentence'])

# Apply tokenization across dataset
map_args = {
    'batched': True,
    'batch_size': 512,
    'remove_columns': ['sentence', 'idx', 'label']  # drop unused columns
}
df_train_tokenize = df_train.map(tokenize_function, **map_args)
df_val_tokenize = df_val.map(tokenize_function, **map_args)
df_test_tokenize = df_test.map(tokenize_function, **map_args)

# -----------------------------------------------------------
# 5. Filter short sequences
# -----------------------------------------------------------
# Drop very short inputs (len < 5 tokens) since they add noise
tokenized_dataset_train = df_train_tokenize.filter(lambda x: len(x['input_ids']) > 5)
tokenized_dataset_val = df_val_tokenize.filter(lambda x: len(x['input_ids']) > 5)
tokenized_dataset_test = df_test_tokenize.filter(lambda x: len(x['input_ids']) > 5)

# Convert Hugging Face datasets into PyTorch tensors
tokenized_dataset_train.set_format('torch')
tokenized_dataset_val.set_format('torch')
tokenized_dataset_test.set_format('torch')

# -----------------------------------------------------------
# 6. Data Collator
# -----------------------------------------------------------
# Handles dynamic padding and creates "labels" for language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # Causal LM (GPT-2) does next-token prediction, not MLM
)

# -----------------------------------------------------------
# 7. DataLoaders
# -----------------------------------------------------------
# Provide batches of data to PyTorch training loop
dataloader_params = {
    'batch_size': 32,
    'shuffle': True,
    'collate_fn': data_collator
}
train_dataloader = DataLoader(tokenized_dataset_train, **dataloader_params)
val_dataloader = DataLoader(tokenized_dataset_val, **dataloader_params)
test_dataloader = DataLoader(tokenized_dataset_test, **dataloader_params)

print(f'Number of training batches: {len(train_dataloader)}')
print(f'Number of validation batches: {len(val_dataloader)}')
print(f'Number of test batches: {len(test_dataloader)}')

# Inspect a sample batch
batch = next(iter(train_dataloader))
print("Batch keys:", batch.keys())  # input_ids, attention_mask, labels
print("Shape of input_ids:", batch['input_ids'].shape)

# -----------------------------------------------------------
# 8. Optimizer & Device setup
# -----------------------------------------------------------
# AdamW is the standard optimizer for transformers
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# Use GPU if available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# -----------------------------------------------------------
# 9. Validation function
# -----------------------------------------------------------
def run_validation():
    """Evaluate model on validation set and return average loss"""
    model.eval()
    total_loss = 0.0
    for batch in val_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
            total_loss += outputs.loss.item()
    return total_loss / len(val_dataloader)

# -----------------------------------------------------------
# 10. Training loop with Early Stopping
# -----------------------------------------------------------
epochs = 30  # maximum epochs

# Early stopping configuration
best_val_loss = float("inf")
patience = 5      # stop if no improvement for 2 epochs
patience_counter = 0
best_model_path = "./gpt2_sst2_best"

for epoch in range(epochs):
    # ---- Training ----
    model.train()
    for step, batch in enumerate(train_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}

        # Forward pass
        outputs = model(**batch)
        loss = outputs.loss

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Print progress
        if step % 50 == 0:
            print(f"[Train] Epoch {epoch+1}, Step {step}, Loss: {loss.item():.4f}")

    # ---- Validation ----
    avg_val_loss = run_validation()
    print(f"[Validation] Epoch {epoch+1}, Loss: {avg_val_loss:.4f}")

    # ---- Early stopping logic ----
    if avg_val_loss < best_val_loss:
        print(f"✅ New best model found (val_loss={avg_val_loss:.4f}). Saving...")
        best_val_loss = avg_val_loss
        patience_counter = 0
        model.save_pretrained(best_model_path)
        tokenizer.save_pretrained(best_model_path)
    else:
        patience_counter += 1
        print(f"⚠️ No improvement. Patience counter = {patience_counter}/{patience}")
        if patience_counter >= patience:
            print("⏹️ Early stopping triggered.")
            break

# -----------------------------------------------------------
# 11. Save final model (for reproducibility)
# -----------------------------------------------------------
final_path = "./gpt2_sst2_final"
model.save_pretrained(final_path)
tokenizer.save_pretrained(final_path)

print(f"✅ Final model saved at {final_path}")
print(f"✅ Best model (lowest validation loss) saved at {best_model_path}")