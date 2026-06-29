

!wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
device      = 'cuda' if torch.cuda.is_available() else 'cpu'

def count_non_embedding_params(model):
  return sum(p.numel() for name, p in model.named_parameters()
  if 'embed' not in name and 'lm_head' not in name)

#  Data loading

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars     = sorted(set(text))
vocab_size = len(chars)
stoi      = {c: i for i, c in enumerate(chars)}
itos      = {i: c for i, c in enumerate(chars)}
encode    = lambda s: [stoi[char] for char in s]
decode    = lambda l: ''.join([itos[i] for i in l])

data  = torch.tensor(encode(text), dtype=torch.long) # Full dataset

def get_batch(split, batch_size, block_size, device, current_train_data, current_val_data):
    data_subset = current_train_data if split == 'train' else current_val_data
    ix   = torch.randint(len(data_subset) - block_size, (batch_size,))
    x    = torch.stack([data_subset[i:i+block_size]   for i in ix])
    y    = torch.stack([data_subset[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

#  Model components

class Head(nn.Module):
    def __init__(self, head_size, n_embd, block_size, dropout):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v   = self.value(x)
        return wei @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, n_embd, block_size, dropout):
        super().__init__()
        self.heads      = nn.ModuleList([Head(head_size, n_embd, block_size, dropout) for _ in range(num_heads)])
        self.proj       = nn.Linear(num_heads * head_size, n_embd)
        self.dropout    = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        head_size  = n_embd // n_head
        self.sa    = MultiHeadAttention(n_head, head_size, n_embd, block_size, dropout)
        self.ffwd  = FeedForward(n_embd, dropout)
        self.ln1   = nn.LayerNorm(n_embd)
        self.ln2   = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class DecoderOnlyTransformer(nn.Module):
    def __init__(self, vocab_size, block_size, n_embd, n_head, n_layer, dropout):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table    = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks  = nn.Sequential(*[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
        self.ln_f    = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        # Bind weights: output projection layer shares weights with token embeddings
        self.lm_head.weight = self.token_embedding_table.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T     = idx.shape
        tok_emb  = self.token_embedding_table(idx)
        pos_emb  = self.position_embedding_table(torch.arange(T, device=idx.device))
        x        = tok_emb + pos_emb
        x        = self.blocks(x)
        x        = self.ln_f(x)
        logits   = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, V = logits.shape
            logits  = logits.view(B * T, V)
            targets = targets.view(B * T)
            loss    = F.cross_entropy(logits, targets)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond  = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits    = logits[:, -1, :]
            probs     = F.softmax(logits, dim=-1)
            idx_next  = torch.multinomial(probs, num_samples=1)
            idx       = torch.cat((idx, idx_next), dim=1)
        return idx

    
def run_experiment(config):
    print(f"\n--- Running experiment: {config.get('name', 'Unnamed')} ---")

    # Extract parameters from config
    batch_size    = config['batch_size']
    block_size    = config['block_size']
    n_embd        = config['n_embd']
    n_head        = config['n_head']
    n_layer       = config['n_layer']
    dropout       = config['dropout']
    max_iters     = config['max_iters']
    eval_interval = config['eval_interval']
    eval_iters    = config['eval_iters']
    learning_rate = config['learning_rate']
    current_device = config['device']
    current_vocab_size = config['vocab_size']

    # Initialize model
    model = GPTLanguageModel(
        vocab_size=current_vocab_size,
        block_size=block_size,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
        dropout=dropout
    ).to(current_device)

    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    @torch.no_grad()
    def estimate_loss_local():
        model.eval()
        out = {}
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y       = get_batch(split, batch_size, block_size, current_device)
                _, loss    = model(X, Y)
                losses[k]  = loss.item()
            out[split] = losses.mean()
        model.train()
        return out        