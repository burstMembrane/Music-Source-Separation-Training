import torch
import argparse
import logging

logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_verify_model(model_path, device):
  """
  Load a TorchScript model, move it to the specified device, and verify
  that all inputs, hidden states, and model parameters are on the same device.
  """
  model = torch.jit.load(model_path)
  print(f"Model loaded from {model_path}")

  model = model.to(device)
  print(f"Model moved to {device}")

  for name, param in model.named_parameters():
      if param.device != device:
          logging.error(f"Parameter '{name}' is not on the target device {device}. Found on {param.device}.")
  
  print("Model parameter validation complete.")

  for name, buffer in model.named_buffers():
      if buffer.device != device:
          logging.error(f"Buffer '{name}' is not on the target device {device}. Found on {buffer.device}.")
  
  print("Model buffer validation complete.")
  return model

def check_inputs_and_hidden_states(input_tensor, hidden_state, device):
  """
  Verify that input tensors and hidden states are on the same target device.
  """
  if input_tensor.device != device:
      logging.error(f"Input tensor is not on the target device {device}. Found on {input_tensor.device}.")
  else:
      print("Input tensor is on the correct device.")

  if isinstance(hidden_state, tuple):
      for i, state in enumerate(hidden_state):
          if state.device != device:
              logging.error(f"Hidden state {i} is not on the target device {device}. Found on {state.device}.")
      print("Hidden state validation complete.")
  elif isinstance(hidden_state, torch.Tensor):
      if hidden_state.device != device:
          logging.error(f"Hidden state is not on the target device {device}. Found on {hidden_state.device}.")
      else:
          print("Hidden state is on the correct device.")
  else:
      logging.error("Hidden state must be a torch.Tensor or a tuple of torch.Tensors.")

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Load and verify a TorchScript model.")
  parser.add_argument("--model_path", type=str, required=True, help="Path to the TorchScript model file.")
  parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run the model on (cpu or cuda).")
  parser.add_argument("--input_size", type=int, nargs=3, default=[1, 10, 3], help="Size of the input tensor (batch_size, features).")
  parser.add_argument("--hidden_size", type=int, nargs=3, default=[1, 10, 3], help="Size of the hidden state tensor (num_layers, batch_size, hidden_dim).")
  
  args = parser.parse_args()
  device = torch.device(args.device)
  
  try:
      model = load_and_verify_model(args.model_path, device)
      model.eval()

      input_tensor = torch.randn(args.input_size).to(device)
      hidden_state = (torch.zeros(args.hidden_size).to(device), torch.zeros(args.hidden_size).to(device))

      check_inputs_and_hidden_states(input_tensor, hidden_state, device)
      
      output = model(input_tensor)
      print("Inference completed successfully.")
  except Exception as e:
      logging.error(f"Error: {e}")

