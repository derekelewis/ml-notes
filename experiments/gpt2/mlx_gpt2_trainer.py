import mlx.core as mx
import mlx.optimizers as optim
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader
from mlx_gpt2 import GPTModel, create_dataloader_v1, train_model_simple

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False,
}

tokenizer = tiktoken.get_encoding("gpt2")

file_path = "shakespeare.txt"
with open(file_path, "r", encoding="utf-8") as file:
    text_data = file.read()

total_characters = len(text_data)
total_tokens = len(tokenizer.encode(text_data))
print("Characters:", total_characters)
print("Tokens:", total_tokens)

train_ratio = 0.90
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]

torch.manual_seed(123)

train_loader = create_dataloader_v1(
    train_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=True,
    shuffle=True,
)
val_loader = create_dataloader_v1(
    val_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=False,
    shuffle=False,
)

print(GPT_CONFIG_124M)
mx.random.seed(123)
model = GPTModel(GPT_CONFIG_124M)
print(model)
mx.eval(model.parameters())
# see https://github.com/ml-explore/mlx/issues/1153 for information on MLX Adam implementation
optimizer = optim.AdamW(
    learning_rate=optim.linear_schedule(0.0, 0.0002, 50), weight_decay=0.1
)
num_epochs = 50
train_losses, val_losses, tokens_seen = train_model_simple(
    model,
    train_loader,
    val_loader,
    optimizer,
    num_epochs=num_epochs,
    eval_freq=5,
    eval_iter=1,
    start_context="Every effort moves you",
    tokenizer=tokenizer,
)

model.save_weights("gpt2_shakespeare_ft.safetensors")
