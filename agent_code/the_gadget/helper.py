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
    agent_position = game_state["self"][3]
    in_exposion_radius = 0
    
    for bomb_position, _ in game_state["bombs"]:
        if dist(agent_position, bomb_position) % 1 != 0:
            in_exposion_radius += 1
    
    if in_exposion_radius == 0:
        return True

    return False


def look_for_targets(free_space,
                     start,
                     targets):
    """
    Find direction of the closest target that can be reached via free tiles.
    """
    if not targets:
        return None

    frontier = [start]
    parent_dict = {start: start}
    dist_so_far = {start: 0}
    best = start
    best_dist = np.abs(np.subtract(targets, start)).sum(axis = 1).min()

    while frontier:
        current = frontier.pop(0)
        
        # Find distance from the current position to all targets, track closest
        d = np.abs(np.subtract(targets, current)).sum(axis=1).min()
        if d + dist_so_far[current] <= best_dist:
            best = current
            best_dist = d + dist_so_far[current]

        if d == 0:
            # Found a path to a target's exact position, mission accomplished!
            best = current
            break
        
        # Add unexplored free neighboring tiles to the queue in a random order
        x, y = current
        neighbors = [(x, y) for (x, y) in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)] if free_space[x, y]]
        shuffle(neighbors)
        for neighbor in neighbors:
            if neighbor not in parent_dict:
                frontier.append(neighbor)
                parent_dict[neighbor] = current
                dist_so_far[neighbor] = dist_so_far[current] + 1
    
    # Determine the first step towards the best found target tile
    current = best
    while True:
        if parent_dict[current] == start:
            return current
        current = parent_dict[current]


def check_action_ideas(self,
                       game_state,
                       COLS,
                       ROWS):
    """
    Generates action ideas and rewards the agent if
    it takes the right action.
    """
    # Get the current position of the agent
    x, y = game_state["self"][3]
    action_ideas = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    shuffle(action_ideas)

    cols = range(1, COLS - 1)
    rows = range(1, ROWS - 1)
    
    # Find dead-end positions and crate positions
    dead_ends = [(x, y) for x in cols for y in rows if
                 (game_state["field"][x, y] == 0) and
                 ([game_state["field"][x + 1, y], game_state["field"][x - 1, y], game_state["field"][x, y + 1], game_state["field"][x, y - 1]].count(0) == 1)]

    crates = [(x, y) for x in cols for y in rows if (game_state["field"][x, y] == 1)]

    # Define targets as a combination of coins, dead-ends, and crates
    targets = game_state['coins'] + dead_ends + crates

    # Add other agents as targets if in hunting mode or no crates/coins left
    if self.ignore_others_timer <= 0 or (len(crates) + len(game_state['coins']) == 0):
        targets.extend(self.others)

    # Exclude targets that are currently occupied by a bomb
    targets = [target for target in targets if target not in self.active_bomb_positions]

    # Take a step towards the most immediately interesting target
    free_space = game_state["field"] == 0

    if self.ignore_others_timer > 0:
        for o in self.others:
            free_space[o] = False

    d = look_for_targets(free_space,
                         (x, y),
                         targets)

    # Add actions based on the direction to the target
    if d == (x, y - 1):
        action_ideas.append('UP')
    if d == (x, y + 1):
        action_ideas.append('DOWN')
    if d == (x - 1, y):
        action_ideas.append('LEFT')
    if d == (x + 1, y):
        action_ideas.append('RIGHT')
    if d is None:
        action_ideas.append('WAIT')

    # Add proposal to drop a bomb if at a dead end
    if (x, y) in dead_ends:
        action_ideas.append('BOMB')

    # Add proposal to drop a bomb if touching an opponent
    if len(self.others) > 0:
        if min(abs(xy[0] - x) + abs(xy[1] - y) for xy in self.others) <= 1:
            action_ideas.append('BOMB')

    # Add proposal to drop a bomb if arrived at target and touching a crate
    if d == (x, y) and ([game_state["field"][x + 1, y], game_state["field"][x - 1, y], game_state["field"][x, y + 1], game_state["field"][x, y - 1]].count(1) > 0):
        action_ideas.append('BOMB')

    # Add proposal to run away from any nearby bomb about to blow
    for (xb, yb), t in game_state['bombs']:
        if (xb == x) and (abs(yb - y) < 4):
            # Run away
            if (yb > y):
                action_ideas.append('UP')
            if (yb < y):
                action_ideas.append('DOWN')
            # If possible, turn a corner
            action_ideas.append('LEFT')
            action_ideas.append('RIGHT')

        if (yb == y) and (abs(xb - x) < 4):
            # Run away
            if (xb > x):
                action_ideas.append('LEFT')
            if (xb < x):
                action_ideas.append('RIGHT')
            # If possible, turn a corner
            action_ideas.append('UP')
            action_ideas.append('DOWN')

    # Try random direction if directly on top of a bomb
    for (xb, yb), t in game_state['bombs']:
        if xb == x and yb == y:
            action_ideas.extend(action_ideas[:4])

    # Pick the last action added to the proposals list that is also valid
    while len(action_ideas) > 0:
        a = action_ideas.pop()
        if a in self.valid_actions:
            # Keep track of chosen action for cycle detection
            if a == 'BOMB':
                self.bomb_history.append((x, y))
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