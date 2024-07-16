# Token: 58d0970878625d710564465887a5b6832c966898b64c8ca8
import torch
from torch import nn
import matplotlib.pyplot as plt

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
  for epoch in range(num_epochs):
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
      
    if epoch % 1000 == 0:
      print(f'Epoch: {epoch} | Train loss: {loss} | Test loss: {test_loss}')
  return loss, test_loss
  
  
def plot_pred(actual_value: torch.tensor, pred: torch.tensor):
  # Define colors for actual and predicted targets
  actual_target_color = 'blue'
  predicted_target_color = 'red'

  # Create a subplot with two rows
  fig, axs = plt.subplots(2)

  # Plot actual target on first row
  axs[0].scatter(actual_value.detach().numpy(), actual_value.detach().numpy(), c=actual_target_color, label='Actual Target', alpha=0.5)
  axs[0].set_title('Actual Target')
  axs[0].legend()

  # Plot predicted target on second row
  axs[1].scatter(actual_value.detach().numpy(), pred.detach().numpy(), c=predicted_target_color, label='Predicted Target', alpha=0.5)
  axs[1].set_title('Predicted Target')
  axs[1].legend()

  # Adjust layout (optional)
  plt.tight_layout()
  plt.show()
  
def convert_numpy_to_tensor(train_data, test_data):
  np_array_train, np_array_test = train_data.to_numpy(), test_data.to_numpy()

  tensor_train, tensor_test  = torch.from_numpy(np_array_train), torch.from_numpy(np_array_test)

  return tensor_train, tensor_test

def separate_input_and_target(data: torch.Tensor):
  train, label = data[:, :-1], data[:, -1]
  # X_test, y_test = tensor_test[:, :-1], tensor_test[:, -1]

  label = label.unsqueeze(dim=1)
  # y_test =y_test.unsqueeze(dim=1)
  # X_train.shape, y_train.shape
  return train, label