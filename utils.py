import torch
from model import Model

model = torch.load("LungCancerNet.pt")
model.eval()
