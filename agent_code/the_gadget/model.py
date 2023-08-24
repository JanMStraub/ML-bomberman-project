import torch.nn as nn
from settings import COLS, ROWS

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()

        self.model_sequence = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3),
            nn.Flatten(start_dim=1),
            nn.Linear(1 * 15 * 15, n_actions),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # Reshape the input and pass it through the model_sequence
        return self.model_sequence(x.view(-1, 2, COLS, ROWS))
