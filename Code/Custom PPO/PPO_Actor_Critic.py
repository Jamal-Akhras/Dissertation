import torch
import warnings
import torch.nn as nn
import torch.nn.functional as F

class ActorCritic(nn.Module):
    def __init__(self, inputDims, outputDims):
        
        super(ActorCritic, self).__init__()
        
        #128 chosen based on experiments conducted by Stanford
        self.firstLayer = nn.Linear(inputDims, 128)
        self.secondLayer = nn.Linear(128, 64)
        self.thridLayer = nn.Linear(64, outputDims)
        
    def forward(self, obs):
        """
        Forward propagation through the network with observation/state representation as input.
        """
        
        obs = torch.tensor(obs, dtype = torch.float32)
        
        activationOne = F.relu(self.first_layer(obs))
        activationTwo = F.relu(self.second_layer(activationOne))
        
        out = self.thridLayer(activationTwo)
        
        return out
        