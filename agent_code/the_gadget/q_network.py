import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler


class DQN(nn.Module):

    def __init__(self, n_observations, n_actions, hidden_size):
        super(DQN, self).__init__()

        self.model_sequence = nn.Sequential(
            nn.Linear(n_observations, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, n_actions)
        )

        self.loss = nn.MSELoss()
        self.optimizer = optim.SGD(self.parameters(), lr = 0.0003)
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size = 10000, gamma = 0.9)

    def forward(self, x):
        # Reshape the input and pass it through the model_sequence
        return self.model_sequence(x)


"""
Improvements

Increase Hidden Layers
self.model_sequence = nn.Sequential(
    nn.Linear(n_observations, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, n_actions)
)

Batch Normalization
self.model_sequence = nn.Sequential(
    nn.Linear(n_observations, 64),
    nn.ReLU(),
    nn.BatchNorm1d(64),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.BatchNorm1d(64),
    nn.Linear(64, n_actions)
)

Dropout
self.model_sequence = nn.Sequential(
    nn.Linear(n_observations, 64),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(64, n_actions)
)

Different Activation Functions
self.model_sequence = nn.Sequential(
    nn.Linear(n_observations, 64),
    nn.LeakyReLU(0.2),
    nn.Linear(64, 64),
    nn.LeakyReLU(0.2),
    nn.Linear(64, n_actions)
)

"""