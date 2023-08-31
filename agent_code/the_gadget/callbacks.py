import os
import pickle
import random
from collections import deque

import math
import numpy as np
import torch

from .helper import navigate_field

EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    self.logger.debug('Successfully entered setup code')
    
    self.action_deque = deque(maxlen=2)
    self.action_deque.append("NONE")
    self.action_deque.append("NONE")
    
    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.policy_net = pickle.load(file)


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """

    random_prob = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * game_state["step"] / EPS_DECAY)

    if self.train:
        if random.random() < random_prob:
            self.logger.debug("Querying model for action.")
            with torch.no_grad():
                features = state_to_features(game_state)
                features_tensor = torch.from_numpy(features).float()
                action = self.policy_net(features_tensor)
                return ACTIONS[torch.argmax(action)]
        else:
            self.logger.debug("Random action.")
            return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])
    else:
        features = state_to_features(game_state)
        features_tensor = torch.from_numpy(features).float()
        action = self.policy_net(features_tensor)
        chosen_action = choose_action(self, action, game_state)
        return chosen_action


def state_to_features(game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    hybrid_matrix = np.zeros(game_state["field"].shape + (2, ), dtype=np.double)

    for (x, y) in game_state["coins"]:
        hybrid_matrix[x, y, 0] = 1

    (x, y) = game_state["self"][3]
    hybrid_matrix[x, y, 1] = 1

    return hybrid_matrix.reshape(-1)


# TODO fix jumping around
def choose_action(self, action, game_state) -> dict:
    np_action = np.argsort(action.detach().numpy())[::-1]
    preferred_actions = navigate_field(self, game_state)
    chosen_action = None

    for i in np_action:
        if ACTIONS[i] in preferred_actions and ACTIONS[i] != self.action_deque[0]:
            self.action_deque.append(chosen_action)
            return ACTIONS[i]