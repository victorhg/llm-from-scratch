
import torch
import torch.nn as nn
import myllm.attention as att
import myllm.layers as layers
import myllm.gpt as transformer

# ChatGPT 2 model parameteres
GPT_CONFIG_124M = {
        "name": "gpt2-small",
        "vocab_size": 50257,
        "context_length": 1024, 
        "emb_dim": 768, 
        "n_heads": 12, 
        "n_layers": 12, 
        "drop_rate": 0.1, 
        "qkv_bias": False  
}


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = att.MultiHeadAttention(
            d_in = cfg["emb_dim"],
            d_out = cfg["emb_dim"],
            num_heads = cfg["n_heads"],
            dropout = cfg["drop_rate"],
            qkv_bias = cfg["qkv_bias"],
            context_length = cfg["context_length"],
        )
        self.ff = layers.FeedForward(cfg)
        self.norm1 = layers.LayerNorm(cfg["emb_dim"])
        self.norm2 = layers.LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)

        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # define embeddings 
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = layers.LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb( torch.arange(seq_len, device=in_idx.device))

        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)

        logits = self.out_head(x)
        return logits
    
    def modelConfig(name="gpt2-small"):
        GPT_CONFIG = GPT_CONFIG_124M.copy()
        GPT_CONFIG["name"] = name
        if name =="gpt2-medium":
            GPT_CONFIG["emb_dim"] = 1024
            GPT_CONFIG["n_layers"] = 24 
            GPT_CONFIG["n_heads"] = 16
            
        elif name == "gpt2-large":
            GPT_CONFIG["emb_dim"] = 1280
            GPT_CONFIG["n_layers"] = 36
            GPT_CONFIG["n_heads"] = 20 
        elif name == "gpt2-xl":
            GPT_CONFIG["emb_dim"] = 1600
            GPT_CONFIG["n_layers"] = 48
            GPT_CONFIG["n_heads"] = 25

        return GPT_CONFIG
    
