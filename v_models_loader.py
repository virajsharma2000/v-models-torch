import requests
import os
import torch
import importlib.util

def use_model(model, device = 'cpu'):
 try:
  print('downloading model ...')

  os.mkdir(model)
 
  base_url = f'https://raw.githubusercontent.com/virajsharma2000/v-models-torch/refs/heads/main/{model}/'

  safetensors = requests.get(base_url + 'model.safetensors').text
  weights_bin = requests.get(base_url + 'model_weights.bin').text
  model_architechture = requests.get(base_url + model.replace('-', '_') + '.py').text

  with open(model + '/' + 'model.safetensors', 'wb') as file:
   file.write(safetensors.encode())

  with open(model + '/' + 'model_weights.bin', 'wb') as file:
   file.write(weights_bin.encode())

  with open(model + '/' + model + '.py', 'w') as file:
   file.write(model_architechture)

  state_dict = torch.load(model + '/' + 'model_weights.bin')
  model_architechture_filepath = model + '/' + model + '.py'

  module_spec = importlib.util.spec_from_file_location("model_architecture", model_architechture_filepath)
  module = importlib.util.module_from_spec(module_spec)
  module_spec.loader.exec_module(module)
  
  model = module.model
  model.to(device)
  model.eval()

  model.load_state_dict(state_dict)
  
  return model
 
 except FileExistsError:
  state_dict = torch.load(model + '/' + 'model_weights.bin')
  model_architechture_filepath = model + '/' + model + '.py'

  module_spec = importlib.util.spec_from_file_location("model_architecture", model_architechture_filepath)
  module = importlib.util.module_from_spec(module_spec)
  module_spec.loader.exec_module(module)
  
  model = module.model
  model.to(device)
  model.eval()

  model.load_state_dict(state_dict)
  
  return model
 