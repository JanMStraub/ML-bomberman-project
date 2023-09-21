# -*- coding: utf-8 -*-

"""
Reinforcement Learning agent for the game Bomberman.
@author: Christian Teutsch, Jan Straub
"""

import random

from math import dist
from collections import deque

# UNUSED
def action_filter(self,
                  game_state):
    """
    Function filters the action list.
    """
    # Get the current position of your agent
    x, y = game_state['self'][3]

    # Define a list of adjacent positions and corresponding actions
    adjacent_positions = [(x, y + 1), (x + 1, y), (x, y - 1), (x - 1, y)]
    valid_actions = ['DOWN', 'RIGHT', 'UP', 'LEFT']

    # Initialize an empty queue to store possible actions
    action_queue = deque()

    # Check each adjacent position
    for position, action in zip(adjacent_positions, valid_actions):
        # If the position is empty (contains a 0 in the field) and has
        # not been visited before, add the action to the queue
        # of possible actions
        if game_state["field"][position] == 0:
            if position not in self.visited_tiles:
                action_queue.append(action)

    # Add the current position to the list of visited tiles
    self.visited_tiles.append((x, y))

    if not action_queue:
        # If the queue is empty, return a randomly chosen action from valid_actions
        self.logger.debug("Random action")
        return [random.choice(valid_actions)]

    # Return the queue of possible actions
    return list(action_queue)


def check_danger_zone(game_state,
                      blast_radius):
    """
    Calculates the blast radius of each bomb for rewards.
    """
    # Get the agent's current position
    agent_position = game_state["self"][3]

    # Iterate through each bomb in the game state
    for (x, y), timer in game_state["bombs"]:
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


def calculate_blast_radius(game_state,
                           blast_radius):
    """
    Calculates the blast radius of all bombs for the features.
    """
    # Initialize an empty list to store positions in the danger zone
    danger_zone = []

    # Iterate through each bomb in the game state
    for (x, y), timer in game_state["bombs"]:
        # Add the bomb's position to the danger zone list
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


def find_closest_coin(self,
                      old_game_state,
                      new_game_state):
    """
    Calculates distance to the closest coin from the agents position.
    """
    old_agent_position = old_game_state["self"][3]
    new_agent_position = new_game_state["self"][3]

    if new_game_state["coins"]:
        # Find the closest coin using the min() function with a
        # lambda function as the key
        self.closest_coin_position = min(new_game_state["coins"], \
            key=lambda coin: dist(new_agent_position, coin))

        old_distance = dist(old_agent_position, self.closest_coin_position)
        new_distance = dist(new_agent_position, self.closest_coin_position)

        if new_distance < old_distance:
            # Return 1 when the agent gets closer to the closest coin.
            return 1

        if new_distance == old_distance:
            # Return 2 when the agent maintains the same distance to the closest coin.
            return 2

    # Return 0 when there are no coins or the agent moves farther from the closest coin.
    return 0


def check_movement(self,
                   old_game_state,
                   new_game_state):
    """
    Checks if the agent is moving.
    """
    old_agent_position = old_game_state["self"][3]
    new_agent_position = new_game_state["self"][3]

    if old_agent_position == new_agent_position:
        # Increment the count if positions are the same
        self.same_position += 1
        # Check if it's been the same for 5 or more consecutive steps
        if self.same_position >= 5:
            return True
    else:
        # Reset the count if positions are different
        self.same_position = 0

    # Return False if conditions are not met
    return False


def check_actions(self,
                  action):
    """
    Checks if the agent uses the same action over and over again.
    """
    # Add the current action to the history
    self.action_history.append(action)

    # Count how many times the action appears in the history
    action_count = self.action_history.count(action)

    # Return True if the action appears more than 10 times
    return action_count > 10
