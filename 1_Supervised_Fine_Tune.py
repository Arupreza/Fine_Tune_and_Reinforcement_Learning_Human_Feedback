from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv
from datasets import load_dataset

load_dotenv()


model_name = 'gpt2'

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# text = "Hello, my dog is cute"

# tokens = tokenizer(text)
# print(tokens)

# decode_out = tokenizer.decode(tokens['input_ids'])
# print(decode_out)   

# texts = ['Hello, this is the first step of RLHF training.', 'I have a dog', 
#         'I also have a cat']
# tokens_obj = tokenizer(texts)
# print(tokens_obj)

# for tokens in tokens_obj['input_ids']:
#     print(tokenizer.decode(tokens))

dataset_name = 'sst2'
df = load_dataset(dataset_name)

df_train, df_val, df_test = df['train'], df['validation'], df['test']


def tokenize_function(batch):
    return tokenizer(batch['sentence'])

map_args = {
    'batched': True,
    'batch_size': 512,
    'remove_columns': ['sentence', 'idx', 'label']
}

df_train_tokenize = df_train.map(tokenize_function, **map_args)
df_val_tokenize = df_val.map(tokenize_function, **map_args)
df_test_tokenize = df_test.map(tokenize_function, **map_args)

tokenized_dataset_train = df_train_tokenize.filter(lambda x: len(x['input_ids']) > 5)
tokenized_dataset_val = df_val_tokenize.filter(lambda x: len(x['input_ids']) > 5)
tokenized_dataset_test = df_test_tokenize.filter(lambda x: len(x['input_ids']) > 5)

for i, seq in enumerate(df_train_tokenize[:10]['input_ids']):
    print(f'Sample {i+1}: {tokenizer.decode(seq)}')

tokenized_dataset_train.set_format('torch')
tokenized_dataset_val.set_format('torch')
tokenized_dataset_test.set_format('torch')