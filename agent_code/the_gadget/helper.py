import random

from math import dist
from collections import deque

# UNUSED
def action_filter(self, game_state):
    # Get the current position of your agent
    x, y = game_state['self'][3]
    
    # Define a list of adjacent positions and corresponding actions
    adjacent_positions = [(x, y + 1), (x + 1, y), (x, y - 1), (x - 1, y)]
    valid_actions = ['DOWN', 'RIGHT', 'UP', 'LEFT']

    # Initialize an empty queue to store possible actions
    action_queue = deque()

    # Check each adjacent position
    for position, action in zip(adjacent_positions, valid_actions):
        # If the position is empty (contains a 0 in the field) and has not been visited before, add the action to the queue of possible actions
        if game_state["field"][position] == 0:
            if position not in self.visited_tiles:
                action_queue.append(action)
        
    # Add the current position to the list of visited tiles
    self.visited_tiles.append((x, y))

    if not action_queue:
        # If the queue is empty, return a randomly chosen action from valid_actions
        self.logger.debug("Random action")
        return [random.choice(valid_actions)]
    else:
        # Return the queue of possible actions
        return list(action_queue)


def check_danger_zone(game_state, blast_radius):
    agent_position = game_state["self"][3]

    for (x, y), timer in game_state["bombs"]:
        if agent_position == (x, y):
            return True
        
        for i in range(1, blast_radius + 1):
            if game_state["field"][x + i, y] == -1:
                break
            if (x + i, y) == agent_position:
                return True

        for i in range(1, blast_radius + 1):
            if game_state["field"][x - i, y] == -1:
                break
            if (x - i, y) == agent_position:
                return True

        for i in range(1, blast_radius + 1):
            if game_state["field"][x, y + i] == -1:
                break
            if (x, y + i) == agent_position:
                return True

        for i in range(1, blast_radius + 1):
            if game_state["field"][x, y - i] == -1:
                break
            if (x, y - i) == agent_position:
                return True


def calculate_blast_radius(game_state, blast_radius):
    danger_zone = []
    
    for (x, y), timer in game_state["bombs"]:
        danger_zone.append((x, y))
        for i in range(1, blast_radius + 1):
            if x + i < game_state["field"].shape[0]:
                if game_state["field"][x + i, y] != -1:
                    danger_zone.append((x + i, y))
            if x - i >= 0:
                if game_state["field"][x - i, y] != -1:
                    danger_zone.append((x - i, y))
            if y + i < game_state["field"].shape[1]:
                if game_state["field"][x, y + i] != -1:
                    danger_zone.append((x, y + i))
            if y - i >= 0:
                if game_state["field"][x, y - i] != -1:
                    danger_zone.append((x, y - i))

    return danger_zone


def find_closest_coin(self, old_game_state, new_game_state):
    old_agent_position = old_game_state["self"][3]
    new_agent_position = new_game_state["self"][3]
    
    if len(new_game_state["coins"]) != 0:

        self.closest_coin_position = min(new_game_state["coins"], key=lambda coin: dist(new_agent_position, coin))
        
        old_distance = dist(old_agent_position, self.closest_coin_position)
        new_distance = dist(new_agent_position, self.closest_coin_position)
        
        if new_distance < old_distance:
            return 1
        
        if new_distance == old_distance:
            return 2

    return 0


def check_movement(self, old_game_state, new_game_state):
    old_agent_position = old_game_state["self"][3]
    new_agent_position = new_game_state["self"][3]
    
    if old_agent_position == new_agent_position:
        self.same_position += 1
        if self.same_position == 5:
            return True
    else:
       self.same_position = 0


def check_actions(self, action):
    action_count = 0
    
    self.action_queue.put(action)
    
    while not self.action_queue.empty():
        act = self.action_queue.get()
        
        if act == action:
            action_count += 1
    
    if action_count > 10:
        return True