import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'PyTorch version: {torch.__version__}, using device: {device}')
