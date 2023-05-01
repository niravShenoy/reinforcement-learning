import argparse
import os
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from torch.utils.tensorboard import SummaryWriter

def layer_init(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0, std=0.1)
        nn.init.constant_(m.bias, 0)

# Actor-Critic Network
class PolicyNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, std=0.0):
        super(PolicyNetwork, self).__init__()

        self.critic = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        self.actor = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )
        
        self.apply(layer_init)

        self.saved_log_probs = []
        self.rewards = []


    def forward(self, x, action = None):
        value = self.critic(x) 
        if action is None:
            action_logits = self.actor(x)
            action_probs = Categorical(logits=action_logits)
            action = action_probs.sample()
            return action, action_probs.log_prob(action), action_probs.entropy(), value, action_logits
        else:
            return None, None, None, value, None
        
        