# UNUSED
def action_filter(self, game_state):
    # Get the current position of your agent
    x, y = game_state['self'][3]
    
    # Define a list of adjacent positions and corresponding actions
    adjacent_positions = [(x, y + 1), (x + 1, y), (x, y - 1), (x - 1, y)]
    valid_actions = ['DOWN', 'RIGHT', 'UP', 'LEFT']

    # Initialize an empty list to store possible actions
    primary_actions = []
    secondary_action = None

    # Check each adjacent position
    for position, action in zip(adjacent_positions, valid_actions):
        # If the position is empty (contains a 0 in the field) and has not been visited before, add the action to the list of possible actions
        if game_state["field"][position] == 0:
            if position not in self.visited_tiles:
                primary_actions.append(action)
            else:
                secondary_action = action
        
    # Add the current position to the list of visited tiles
    self.visited_tiles.append((x, y))

    if len(primary_actions) == 0:
        return secondary_action
    else:
    # Return the list of possible actions
        return primary_actions
