import torch.nn as nn


class DQN(nn.Module):

    def __init__(self, n_observations, n_actions, hidden_size):
        super(DQN, self).__init__()

        self.model_sequence = nn.Sequential(
        nn.Linear(n_observations, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x):
        # Reshape the input and pass it through the model_sequence
        return self.model_sequence(x)