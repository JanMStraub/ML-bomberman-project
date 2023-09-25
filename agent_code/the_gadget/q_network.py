# -*- coding: utf-8 -*-

"""
Reinforcement Learning agent for the game Bomberman.
@author: Christian Teutsch, Jan Straub
"""

from torch import nn

class DQN(nn.Module):
    """
    Deep Q Learning network class.
    """

    def __init__(self,
                 n_observations,
                 n_actions,
                 hidden_size,
                 dropout):
        super().__init__()

        self.model_sequence = nn.Sequential(
            nn.Linear(n_observations, hidden_size),
            nn.ReLU(),
#            nn.Dropout(p = dropout),   # Dropout Layer
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
#            nn.Dropout(p = dropout),   # Dropout Layer
            nn.Linear(hidden_size, n_actions)
        )


    def forward(self,
                x):
        """
        Reshape the input and pass it through the model_sequence
        """
        return self.model_sequence(x)
