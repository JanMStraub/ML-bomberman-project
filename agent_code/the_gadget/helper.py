import random

from collections import deque

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


# TODO
def check_blast_radius(game_state, blast_radius):
    agent_position = game_state["self"][3]
    for (x, y), timer in game_state["bombs"]:
        if abs(agent_position[0] - x) + abs(agent_position[1] - y) <= blast_radius:
            return True
    return False


def check_coin_sum(self, events):    
    if "COIN_COLLECTED" in events:
        self.collected_coins += 1
    
    if self.collected_coins == 15:
        return True

    return False




# TODO Nicht bewegt checker
# TODO reward fürs längere überleben
# TODO reward näher an Coin
