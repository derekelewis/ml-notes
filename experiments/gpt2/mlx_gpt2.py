import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        self.d_in = d_in
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

    def __call__(self, x):
        b, num_tokens, d_in = x.shape
        mask = nn.MultiHeadAttention.create_additive_causal_mask(num_tokens)

        # W(x):
        # (d_in, d_out) @ (b, num_tokens, d_in) -> (b, num_tokens, d_out)
        # or
        # Wx:
        # (d_in, d_out) @ ((b, num_tokens, d_in) -> (b, d_in, num_tokens))
        # -> (b, d_out, num_tokens) -> (b, num_tokens, d_out)
        # foo = W_query.weight @ x.transpose(0, 2, 1)
        # foo = foo.transpose(0, 2, 1)
        # print(foo)
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        # print(f"queries before reshape: {queries}")
        # print(f"queries shape before reshape: {queries.shape}")
        # b, num_tokens, d_out -> b, num_tokens, num_heads, head_dim
        queries = queries.reshape(b, num_tokens, self.num_heads, self.head_dim)
        # print(f"queries after reshape: {queries}")
        # print(f"queries shape after reshape: {queries.shape}")
        # b, num_tokens, d_out -> b, num_tokens, num_heads, head_dim
        keys = keys.reshape(b, num_tokens, self.num_heads, self.head_dim)
        # b, num_tokens, d_out -> b, num_tokens, num_heads, head_dim
        values = values.reshape(b, num_tokens, self.num_heads, self.head_dim)

        queries = queries.transpose(0, 2, 1, 3)  # b, num_heads, num_tokens, head_dim
        keys = keys.transpose(0, 2, 1, 3)  # b, num_heads, num_tokens, head_dim
        values = values.transpose(0, 2, 1, 3)  # b, num_heads, num_tokens, head_dim

        # (b, num_heads, num_tokens, head_dim) @ (b, num_heads, head_dim, num_tokens)
        # -> (b, num_heads, num_tokens, num_tokens)
        attn_scores = queries @ keys.transpose(0, 1, 3, 2)
        # print(f"attn_scores shape: {attn_scores.shape}")
        # applied to each head
        attn_scores = attn_scores + mask
        # applied to each head
        attn_weights = mx.softmax(attn_scores / keys.shape[-1] ** 0.5, axis=-1)
        # print(f"attn_weights: {attn_weights}")
        # print(f"attn_weights shape: {attn_weights.shape}")
        # applied to each head
        attn_weights = self.dropout(attn_weights)
        # (b, num_heads, num_tokens, num_tokens) @ (b, num_heads, num_tokens, head_dim)
        # -> (b, num_heads, num_tokens, head_dim) -> (b, num_tokens, num_heads, head_dim)
        context_vectors = (attn_weights @ values).transpose(0, 2, 1, 3)
        # print(f"context_vectors shape: {context_vectors.shape}")
        context_vectors = context_vectors.reshape(b, num_tokens, self.d_out)
        # print(f"context_vectors reshape shape: {context_vectors.shape}")
        context_vectors = self.out_proj(context_vectors)
        return context_vectors


class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = mx.ones(
            (1, emb_dim)
        )  # need to figure out how to make trainable parameter
        self.shift = mx.zeros(
            (1, emb_dim)
        )  # need to figure out how to make trainable parameter

    def __call__(self, x):
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        norm_x = (x - mean) / mx.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, x):
        return (
            0.5
            * x
            * (
                1
                + mx.tanh(
                    mx.sqrt(mx.array(2.0 / mx.pi)) * (x + 0.044715 * mx.power(x, 3))
                )
            )
        )


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def __call__(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_resid = nn.Dropout(cfg["drop_rate"])

    def __call__(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_resid(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_resid(x)
        x = x + shortcut
        return x


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def __call__(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(mx.arange(seq_len))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        logits = model(idx_cond)

        logits = logits[:, -1, :]
        probas = mx.softmax(logits, axis=-1)
        idx_next = mx.argmax(probas, axis=-1, keepdims=True)
        idx = mx.concatenate((idx, idx_next), axis=1)

    return idx


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    encoded_tensor = mx.expand_dims(mx.array(encoded), 0)  # add batch dimension
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)  # remove batch dimension
    return tokenizer.decode(flat.tolist())


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt)

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(
    txt,
    batch_size=4,
    max_length=256,
    stride=128,
    shuffle=True,
    drop_last=True,
    num_workers=0,
):
    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=0,
    )

    return dataloader


def calc_loss_batch(model, input_batch, target_batch):
    input_batch, target_batch = input_batch, target_batch
    return mx.mean(nn.losses.cross_entropy(model(input_batch), target_batch))


def calc_loss_loader(data_loader, model, num_batches=None):
    total_loss = 0.0
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(
                model, mx.array(input_batch.numpy()), mx.array(target_batch.numpy())
            )
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def train_model_simple(
    model,
    train_loader,
    val_loader,
    optimizer,
    num_epochs,
    eval_freq,
    eval_iter,
    start_context,
    tokenizer,
):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    loss_and_grad_fn = nn.value_and_grad(model, calc_loss_batch)

    for epoch in range(num_epochs):
        for input_batch, target_batch in train_loader:
            input_batch = mx.array(input_batch.numpy())
            target_batch = mx.array(target_batch.numpy())
            loss, grads = loss_and_grad_fn(model, input_batch, target_batch)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)

            tokens_seen += input_batch.size
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(
                    f"Ep {epoch+1} (Step {global_step:06d}): "
                    f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}"
                )

        generate_and_print_sample(model, tokenizer, start_context)
    return train_losses, val_losses, track_tokens_seen


def evaluate_model(model, train_loader, val_loader, eval_iter):
    model.eval()
    train_loss = calc_loss_loader(train_loader, model, num_batches=eval_iter)
    val_loss = calc_loss_loader(val_loader, model, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer)
    token_ids = generate_text_simple(
        model=model, idx=encoded, max_new_tokens=50, context_size=context_size
    )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))  # Compact print format
    model.train()
