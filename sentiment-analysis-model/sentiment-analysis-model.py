from transformers import AutoTokenizer
from datasets import load_dataset
import torch.nn as nn
import torch.optim as optim
import torch

d_model = 442

class SentimentAnalysis(nn.Module):
 def __init__(self):
  super().__init__()
  self.embed = nn.Embedding(embedding_dim = d_model, num_embeddings = 1000000)
  self.attention = nn.MultiheadAttention(embed_dim = d_model, num_heads = 2, batch_first = True)
  self.layer_norm = nn.LayerNorm(d_model)
  self.ffn = nn.Sequential(
   nn.Linear(d_model, d_model * 2),
   nn.GELU(),
   nn.Linear(d_model * 2, d_model)
  )
  self.gelu = nn.GELU()
  self.fc2 = nn.Linear(442, 1)
  self.sigmoid = nn.Sigmoid()

 def forward(self, x, attention_mask):
  y = self.embed(x)
  
  y = self.layer_norm(y)
  y = y + self.attention(y, y, y, key_padding_mask = attention_mask == 0)[0]
  y = self.ffn(y) + y
 
  y = self.layer_norm(y)
  y = y + self.attention(y, y, y, key_padding_mask = attention_mask == 0)[0]
  y = self.ffn(y) + y
  
  y = self.layer_norm(y)
  y = y + self.attention(y, y, y, key_padding_mask = attention_mask == 0)[0]
  y = self.ffn(y) + y

  y = self.sigmoid(self.fc2(self.gelu(y))).mean(dim = 1)
  
  return y.squeeze()
 
 
