# -*- coding: utf-8 -*-

"""
Reinforcement Learning agent for the game Bomberman.
@author: Christian Teutsch, Jan Straub
"""

from random import shuffle
from math import dist

import numpy as np

def check_for_loop(self,
                   game_state):
    """
    Checks for action loops.
    """
    # Get the current position of the agent
    x, y = game_state["self"][3]

    # Check if the current position has occurred more than twice in the history
    if self.coordinate_history.count((x, y)) > 2:
        return True

    # Append the current position to the coordinate_history
    self.coordinate_history.append((x, y))

    # Return False if the current position is not in a loop
    return False


def is_valid_movement_action(self,
                             game_state,
                             action):
    """
    Creates valid action list and checks wether the selected action is in
    the valid action list.
    """
    # Get the current position of the agent
    x, y = game_state["self"][3]

    # Define the potential directions after the move
    directions = [(x, y), (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]

    # Initialize lists to store valid tiles
    valid_tiles = []

    # Check each potential direction
    for d in directions:
        if (
            (game_state['field'][d] == 0) and  # Tile is empty
            (game_state['explosion_map'][d] < 1) and  # No ongoing explosion
            (self.bomb_map[d] > 0) and  # No bomb in this tile
            (d not in self.others) and  # No other agents in this tile
            (d not in self.active_bomb_positions)  # No bombs in this tile
        ):
            valid_tiles.append(d)

    # Check each direction for valid actions
    if (x - 1, y) in valid_tiles:
        self.valid_actions.append('LEFT')
    if (x + 1, y) in valid_tiles:
        self.valid_actions.append('RIGHT')
    if (x, y - 1) in valid_tiles:
        self.valid_actions.append('UP')
    if (x, y + 1) in valid_tiles:
        self.valid_actions.append('DOWN')
    if (x, y) in valid_tiles:
        self.valid_actions.append('WAIT')

    # Disallow the BOMB action if the agent dropped a bomb in the same spot recently
    if (game_state['self'][2] > 0) and ((x, y) not in self.bomb_history):
        self.valid_actions.append('BOMB')

    # Check if the given action is valid
    if action in self.valid_actions:
        return True

    return False


def closer_to_coin(self,
                   old_game_state,
                   new_game_state):
    """
    Calculates distance to the closest coin from the agents position.
    """
    coins = new_game_state["coins"]
    old_agent_position = old_game_state["self"][3]
    new_agent_position = new_game_state["self"][3]

    # Exclude coins that are currently occupied by a bomb
    coins = [coin for coin in coins if coin not in self.active_bomb_positions]
    
    if coins:
        if self.closest_coin_position == new_agent_position or \
            self.closest_coin_position is None:
            self.closest_coin_position = min(new_game_state["coins"], \
                key=lambda coin: dist(new_agent_position, coin))

        old_distance = dist(old_agent_position, self.closest_coin_position)
        new_distance = dist(new_agent_position, self.closest_coin_position)

        if new_distance < old_distance:
            return True
    
    return False


def destroy_crate_action_reward(game_state,
                                action):
    """
    Checks if bomb is dropped in front of a crate.
    """
    x, y, = game_state["self"][3]
    adjacent_positions = [(x, y + 1), (x, y - 1), (x + 1, y), (x - 1, y)]
    
    for position in adjacent_positions:
        if game_state["field"][position] == 1 and action == "BOMB":
            return True
    
    return False


def bomb_evaded(game_state):
    """
    Check if agent has evaded the bomb.
    """
    agent_position = game_state["self"][3]
    in_exposion_radius = 0
    
    for bomb_position, _ in game_state["bombs"]:
        if dist(agent_position, bomb_position) % 1 != 0:
            in_exposion_radius += 1
    
    if in_exposion_radius == 0:
        return True

    return False


def check_danger_zone(game_state,
                      blast_radius):
    """
    Calculates the blast radius of each bomb for rewards.
    """
    # Get the agent's current position
    agent_position = game_state["self"][3]

    # Iterate through each bomb in the game state
    for (x, y), _ in game_state["bombs"]:
        if agent_position == (x, y):
            # If the agent is at the same position as a bomb, it's in the danger zone
            return True

        # Check positions in the horizontal (x-axis) direction
        for i in range(1, blast_radius + 1):
            if game_state["field"][x + i, y] == -1:
                break  # Stop checking if an obstacle is encountered
            if (x + i, y) == agent_position:
                 # If the agent's position matches a position within the
                 # blast radius, it's in the danger zone
                return True

        # Check positions in the horizontal (x-axis) direction in the opposite direction
        for i in range(1, blast_radius + 1):
            if game_state["field"][x - i, y] == -1:
                break  # Stop checking if an obstacle is encountered
            if (x - i, y) == agent_position:
                # If the agent's position matches a position within
                # the blast radius, it's in the danger zone
                return True

        # Check positions in the vertical (y-axis) direction
        for i in range(1, blast_radius + 1):
            if game_state["field"][x, y + i] == -1:
                break  # Stop checking if an obstacle is encountered
            if (x, y + i) == agent_position:
                # If the agent's position matches a position within
                # the blast radius, it's in the danger zone
                return True

        # Check positions in the vertical (y-axis) direction in the opposite direction
        for i in range(1, blast_radius + 1):
            if game_state["field"][x, y - i] == -1:
                break  # Stop checking if an obstacle is encountered
            if (x, y - i) == agent_position:
                # If the agent's position matches a position within
                # the blast radius, it's in the danger zone
                return True

    # If the agent is not in the danger zone of any bombs, return False
    return False


# TODO blast radius is still deadly 1 step after timer is 0
def calculate_blast_radius(game_state,
                           blast_radius):
    """
    Calculates the blast radius of all bombs for the features.
    """
    # Initialize an empty list to store positions in the danger zone
    danger_zone = []
    # Iterate through each bomb in the game state
    for (x, y), _ in game_state["bombs"]:
        # Add the bomb's position to the danger zone list
        if (x, y) not in danger_zone:
            danger_zone.append((x, y))

            # Iterate in all four possible directions (up, down, left, right)
            for i in range(1, blast_radius + 1):
                # Check positions in the horizontal (x-axis) direction
                if x + i < game_state["field"].shape[0]:
                    if game_state["field"][x + i, y] != -1:
                        # Add positions within blast radius
                        danger_zone.append((x + i, y))

                if x - i >= 0:
                    if game_state["field"][x - i, y] != -1:
                        # Add positions within blast radius
                        danger_zone.append((x - i, y))

                # Check positions in the vertical (y-axis) direction
                if y + i < game_state["field"].shape[1]:
                    if game_state["field"][x, y + i] != -1:
                        # Add positions within blast radius
                        danger_zone.append((x, y + i))

                if y - i >= 0:
                    if game_state["field"][x, y - i] != -1:
                        # Add positions within blast radius
                        danger_zone.append((x, y - i))

    # Return the list of positions in the danger zone
    return danger_zone


def bomb_at_spawn(old_game_state,
                  new_game_state):
    """
    Checks wether the bomb is planted in the start position.
    """
    x, y = old_game_state["self"][3]
    start_positions = [(1, 1), (1, 15), (15, 1), (15, 15)]

    if (x, y) in start_positions and new_game_state["step"] < 5:
        return True

    return False