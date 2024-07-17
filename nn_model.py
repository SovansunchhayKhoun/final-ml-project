# http://127.0.0.1:8888/lab?token=a12ea18f6e286be94153cb60ac1960c5467af21fa0af8f76
import torch
from torch import nn
from tqdm.auto import tqdm

class NeuralNetworkModel(nn.Module):
  """A Simple Nerual Network Model that has 1 linear layer 

  Args:
      input_features (int): Number of input features,
      output_features (int): Number of output features
  """
  def __init__(self, input_features: int, output_features: int):
    super().__init__()
    self.layer_1 = nn.Sequential(nn.Linear(in_features=input_features, out_features=output_features))
  def forward(self, x):
    return self.layer_1(x)

def train_fn(model: torch.nn.Module, 
             train_data: torch.tensor, 
             train_label: torch.tensor, 
             test_data: torch.tensor, 
             test_label: torch.tensor, 
             loss_fn, 
             optimizer, 
             num_epochs):
  for _ in tqdm(range(num_epochs)):
    # Put model to train mode
    model.train()
    
    # Do the forward pass
    y_pred = model(train_data)
    
    # Calculate the loss
    loss = loss_fn(y_pred, train_label)
    
    # Optimizer zero grad
    optimizer.zero_grad()
    
    # Perform backward propagation
    loss.backward()
    
    # Optimizer Step
    optimizer.step()
    
    ## Testing Mode
    model.eval()
    
    with torch.inference_mode():
      # Forward pass
      test_pred = model(test_data)
      # Calculate test loss
      test_loss = loss_fn(test_pred, test_label)

  return loss, test_loss