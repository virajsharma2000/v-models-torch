import torch
from transformers import AutoTokenizer
import torch.nn as nn

class PooledMHAWrapper(nn.Module):
 def __init__(self):
  super().__init__()
  self.mha = nn.MultiheadAttention(embed_dim = 100, num_heads = 10, batch_first = True)

 def forward(self, x):
   attn, _ = self.mha(x, x, x)
   pooled_output = attn.mean(dim = 1)

   return pooled_output

class SentimentAnalysis(nn.Module):
 def __init__(self):
  super().__init__()
  self.network = nn.Sequential(
   nn.Embedding(num_embeddings = 100000, embedding_dim = 100),
   PooledMHAWrapper(),
   nn.Linear(in_features = 100, out_features = 200),
   nn.LeakyReLU(),
   nn.Linear(in_features = 200, out_features = 400),
   nn.LeakyReLU(),
   nn.Linear(in_features = 400, out_features = 800),
   nn.LeakyReLU(),
   nn.Linear(in_features = 800, out_features = 400),
   nn.LeakyReLU(),
   nn.Linear(in_features = 400, out_features = 200),
   nn.LeakyReLU(),
   nn.Linear(in_features = 200, out_features = 100),
   nn.LeakyReLU(),
   nn.Linear(in_features = 100, out_features = 50),
   nn.LeakyReLU(),
   nn.Linear(in_features = 50, out_features = 25),
   nn.LeakyReLU(),
   nn.Linear(in_features = 25, out_features = 12),
   nn.LeakyReLU(),
   nn.Linear(in_features = 12, out_features = 6),
   nn.LeakyReLU(),
   nn.Linear(in_features = 6, out_features = 3),
   nn.LeakyReLU(),
   nn.Linear(in_features = 3, out_features = 1),
   nn.Sigmoid()
  )

 def forward(self, text):
  tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

  def tokenize(text):
   return tokenizer(text, padding = "max_length", truncation = True)
  
  return self.network(torch.tensor(tokenize(text)['input_ids']).unsqueeze(dim = 0)).item() > 0.1


model = SentimentAnalysis()