import os
import pickle
import random

import math
import numpy as np
import torch

from .helper import calculate_blast_radius
from settings import BOMB_POWER

EPS_START = 0.95
EPS_END = 0.05
EPS_DECAY = 200

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.
    """
    self.logger.debug('Successfully entered setup code')
    
    # Check if a saved model file exists, and whether to train from scratch or load it
    model_file_path = "my-saved-model.pt"
    if self.train or not os.path.isfile(model_file_path):
        self.logger.info("Setting up model from scratch.")
    else:
        self.logger.info("Loading model from saved state.")
        try:
            with open(model_file_path, "rb") as file:
                self.policy_net = pickle.load(file)
        except FileNotFoundError:
            self.logger.warning(f"Saved model file '{model_file_path}' not found. Setting up model from scratch.")


def act_random(self,
               game_state):
    """
    Choosing action purely at random.
    """
    self.logger.debug("Choosing action purely at random.")
    random_action = np.random.choice(ACTIONS, p=[0.2, 0.2, 0.2, 0.2, 0.1, 0.1])
    self.logger.debug(f"Random action: {random_action}")
    
    return random_action


def act_learned(self,
                game_state):
    """
    Querying model for action.
    """
    self.logger.debug("Querying model for action.")
    game_state_tensor = torch.from_numpy(state_to_features(game_state)).float()
    action = ACTIONS[self.policy_net.forward(game_state_tensor).argmax().item()]
    self.logger.debug(f"Action: {action}")
    
    return action


def act(self,
        game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.
    """

    if self.train:
        random_prob = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1.0 * game_state["step"] / EPS_DECAY)

        if random.random() > random_prob:
            return act_random(self, game_state)
        else:
            return act_learned(self, game_state)
    
    return act_learned(self, game_state)


def state_to_features(game_state: dict) -> np.array:
    """
    Converts the game state to the input of your model, i.e.
    a feature vector.
    """

    if game_state is None:
        return None

    # Define mappings for cell types
    cell_mappings = {
        -1: 0,  # Wall
        0: 1,   # Free tile
        1: 2,   # Crate
    }

    # Create an array of ones with the same shape as the game field
    feature_matrix = np.ones_like(game_state["field"],
                                  dtype = int)

    # Map cell types to their corresponding values
    for cell_type, value in cell_mappings.items():
        feature_matrix[game_state["field"] == cell_type] = value

    # Set agent position
    agent_position = game_state["self"][3]
    feature_matrix[agent_position[0], agent_position[1]] = 3

    if game_state["coins"]:
        # Set coin positions
        coin_positions = np.array(game_state["coins"])
        feature_matrix[coin_positions[:, 0], coin_positions[:, 1]] = 4

    if game_state["bombs"]:
        # Set bombs positions
        bomb_positions = np.array(
            [bomb for bomb in calculate_blast_radius(game_state,
                                                     BOMB_POWER)])
        feature_matrix[bomb_positions[:, 0], bomb_positions[:, 1]] = 5

    if game_state["others"]:
        # Set others positions
        other_positions = np.array([other[3] for other in game_state["others"]])
        feature_matrix[other_positions[:, 0], other_positions[:, 1]] = 6

    return feature_matrix.reshape(-1)
