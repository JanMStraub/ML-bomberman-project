import torch.nn as nn
import torch.optim as optim


class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        
        self.model_sequence = nn.Sequential(
            nn.Linear(n_observations, 6),
            nn.ReLU(),
            nn.Linear(6, n_actions)
        )

        self.loss = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=0.0003)
        
    def forward(self, x):
        # Reshape the input and pass it through the model_sequence
        return self.model_sequence(x)
