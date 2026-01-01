import torch
import torch.nn as nn

d_model = 100
max_length = 1000
vocab = 9
temperature = 10000

class LLM(nn.Module):
 def __init__(self):
  super().__init__()
  self.embed = nn.Embedding(num_embeddings = vocab, embedding_dim = d_model)
  self.pos_embed = nn.Embedding(num_embeddings = max_length, embedding_dim = d_model)
  self.layer_norm = nn.LayerNorm(d_model)
  self.mha = nn.MultiheadAttention(embed_dim = d_model, num_heads = 2, batch_first = True)
  self.ffn = nn.Sequential(
   nn.Linear(d_model, d_model * 2),
   nn.GELU(),
   nn.Linear(d_model * 2, d_model)
  )
  self.gelu = nn.GELU()
  self.fc2 = nn.Linear(d_model, vocab)
  self.softmax = nn.Softmax()

 def forward(self, x, padding_mask):
  y = self.embed(x) + self.pos_embed(torch.arange(0, x.shape[1]).expand(x.shape))
  
  y = self.layer_norm(y)
  y = y + self.mha(y, y, y, attn_mask = torch.triu(torch.ones(y.shape[1], y.shape[1]), diagonal = 1), key_padding_mask = padding_mask == 0)[0]
  y = y + self.ffn(y)

  y = self.layer_norm(y)
  y = y + self.mha(y, y, y, attn_mask = torch.triu(torch.ones(y.shape[1], y.shape[1]), diagonal = 1), key_padding_mask = padding_mask == 0)[0]
  y = y + self.ffn(y)
  
  y = self.layer_norm(y)
  y = y + self.mha(y, y, y, attn_mask = torch.triu(torch.ones(y.shape[1], y.shape[1]), diagonal = 1), key_padding_mask = padding_mask == 0)[0]
  y = y + self.ffn(y)
  
  logits = self.fc2(self.gelu(y)) / temperature
  
  return torch.multinomial(self.softmax(logits)[:, -1], 1)
 

def tokenize(prompt):
 vocab = ['kyu', 'tang', 'kar', 'raha', 'hai', 'tu', 'papa', 'ko', 'naa', 'EOS']
 
 return [vocab.index(token) for token in prompt.split()]

def detokenize(token_idx):
 vocab = ['kyu', 'tang', 'kar', 'raha', 'hai', 'tu', 'papa', 'ko', 'EOS']

 return vocab[token_idx]


llm = LLM()

tokens = tokenize('tang kar raha hai tu papa ko naa')
key_padding_mask = [1, 1, 1, 1, 1, 1, 1, 1]
response = ''

for i in range(max_length):
 token_idx = llm(torch.tensor(tokens).unsqueeze(dim = 0), torch.tensor(key_padding_mask).unsqueeze(dim = 0))
 
 token = detokenize(token_idx)

 if token == 'EOS':
  break
 
 else:
  response += token + ' '
  tokens.append(tokenize(token)[0])
  key_padding_mask.append(1)

print(response)