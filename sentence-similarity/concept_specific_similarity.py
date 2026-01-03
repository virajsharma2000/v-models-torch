import torch.nn as nn
import torch

class ConceptSpecificSimilarity(nn.Module):
 def __init__(self, vocab):
  super().__init__()
  d_model = 442
  max_length = 1000
  self.embed = nn.Embedding(num_embeddings = vocab, embedding_dim = d_model)
  self.pos_embed = nn.Embedding(num_embeddings = max_length, embedding_dim = d_model)
  self.mha = nn.MultiheadAttention(embed_dim = d_model, num_heads = 2, batch_first = True)
  self.cosine_similarity = nn.CosineSimilarity(dim = 1)
  self.layer_norm = nn.LayerNorm(d_model)

 def forward(self, x, y, key_padding_mask_x, key_padding_mask_y):
  xb, xl = x.shape
  yb, yl = y.shape

  embeddings1 = self.layer_norm(self.embed(x) + self.pos_embed(torch.arange(0, xl).expand(xb, xl)))
  embeddings2 = self.layer_norm(self.embed(y) + self.pos_embed(torch.arange(0, yl).expand(yb, yl)))
  
  attn_out1, _ = self.mha(embeddings1, embeddings1, embeddings1, key_padding_mask = key_padding_mask_x == 0)
  attn_out2, _ = self.mha(embeddings2, embeddings2, embeddings2, key_padding_mask = key_padding_mask_y == 0)

  return self.cosine_similarity(self.layer_norm(attn_out1), self.layer_norm(attn_out2)).mean(dim = 1)
 

model = ConceptSpecificSimilarity(100)
