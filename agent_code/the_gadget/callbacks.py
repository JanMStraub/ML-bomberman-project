import os
import pickle
import random

import math
import numpy as np
import torch

from .helper import action_filter, calculate_blast_radius
from settings import BOMB_POWER

EPS_START = 0.90
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
    self.visited_tiles = []
    self.closest_coin_position = None
    
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


def act_random(self, game_state):
    self.logger.debug("Choosing action purely at random.")
    random_action = np.random.choice(ACTIONS, p=[0.2, 0.2, 0.2, 0.2, 0.1, 0.1])
    self.logger.debug(f"Random action: {random_action}")
    
    return random_action


def act_learned(self, game_state):
    self.logger.debug("Querying model for action.")
    game_state_tensor = torch.from_numpy(state_to_features(game_state)).float()
    action = ACTIONS[self.policy_net.forward(game_state_tensor).argmax().item()]
    #valid_action = choose_action(self, self.policy_net(game_state_tensor), game_state)
    self.logger.debug(f"Action: {action}")
    
    return action


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """

    if self.train:
        random_prob = EPS_END + (EPS_START - EPS_END) * math.exp(-1.0 * game_state["step"] / EPS_DECAY)

        if random.random() > random_prob:
            return act_random(self, game_state)
        else:
            return act_learned(self, game_state)
    
    return act_learned(self, game_state)


def state_to_features(game_state: dict) -> np.array:
    """
    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """

    if game_state is None:
        return None

    # Define mappings for cell types
    cell_mappings = {
        -1: 0,   # Wall
        1: 1,    # Crate
    }

    # Create an array of ones with the same shape as the game field
    feature_matrix = np.ones_like(game_state["field"], dtype=np.int)

    # Map cell types to their corresponding values
    for cell_type, value in cell_mappings.items():
        feature_matrix[game_state["field"] == cell_type] = value

    # Set Agent position
    agent_position = game_state["self"][3]
    feature_matrix[agent_position[0], agent_position[1]] = 2

    if game_state["coins"]:
        # Set Coin positions
        coin_positions = np.array(game_state["coins"])
        feature_matrix[coin_positions[:, 0], coin_positions[:, 1]] = 3

    if game_state["bombs"]:
        # Set bomb positions
        bomb_positions = np.array([bomb for bomb in calculate_blast_radius(game_state, BOMB_POWER)])
        feature_matrix[bomb_positions[:, 0], bomb_positions[:, 1]] = 4
    
    if game_state["others"]:
        # Set bomb positions
        other_positions = np.array(game_state["others"])
        feature_matrix[other_positions[:, 0], other_positions[:, 1]] = 5

    return feature_matrix.reshape(-1)


def choose_action(self, actions, game_state) -> dict:
    action_scores = actions.detach().numpy()
    sorted_actions = np.argsort(-action_scores)
    
    print(action_scores)
    print(sorted_actions)

    # Choose the best valid action from the sorted list
    for action_idx in sorted_actions:
        action = ACTIONS[action_idx]
        print(action)
        if action in action_filter(self, game_state):
            return action

    # If no valid action is found, return default action
    return "WAIT"
