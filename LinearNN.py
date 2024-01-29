
import torch 
from torch import nn

class NeuralNetework(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack == nn.Sequential(
            nn.Linear(0,0), # FIXME need actual dimensions 
            nn.ReLU()
        )

    def foward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    