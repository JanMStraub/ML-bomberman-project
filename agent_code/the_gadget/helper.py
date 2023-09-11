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


def check_blast_radius(game_state, blast_radius):
    active_bombs_list = game_state["bombs"]

    for (x, y), countdown in active_bombs_list:
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            for i in range(1, blast_radius + 1):
                new_x, new_y = x + i * dx, y + i * dy
                if new_x < 0 or new_x >= len(game_state['field']) or new_y < 0 or new_y >= len(game_state['field'][0]) or game_state['field'][new_x, new_y] == -1:
                    break
                if game_state["self"] == (new_x, new_y):
                    return True
    return False