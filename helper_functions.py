import torch
import matplotlib.pyplot as plt
import pandas as pd

def plot_pred(actual_value, pred):
  actual_target_color = 'blue'
  predicted_target_color = 'red'

  _, axs = plt.subplots(1, figsize=(10, 6))

  # Plot actual vs predicted values
  axs.scatter(actual_value, 
              pred, 
              c=predicted_target_color, 
              label='Predicted vs Actual',
              alpha=0.5)
  axs.plot([actual_value.min(), actual_value.max()], 
            [actual_value.min(), actual_value.max()], 
            color=actual_target_color, linestyle='--', linewidth=2, label='Perfect Prediction Line')

  # Set titles and labels
  axs.set_title('Actual vs Predicted Values')
  axs.set_xlabel('Actual Value')
  axs.set_ylabel('Predicted Value')
  axs.legend()

  plt.tight_layout()
  plt.show()
  
def convert_numpy_to_tensor(data: pd.DataFrame):
  # Convert dataframe to numpy
  np_array = data.to_numpy()
  
  # convert numpy to tensor
  tensor = torch.from_numpy(np_array)

  return tensor

def separate_input_and_target(data: pd.DataFrame, target: str):
  train, label = data.drop(columns=target), data[target]

  return train, label