def navigate_field(self, game_state) -> dict:
    x, y = game_state['self'][3]
    possible_actions = []

    adjacent_positions = [(x, y + 1), (x + 1, y), (x, y - 1), (x - 1, y)]
    valid_actions = ['DOWN', 'RIGHT', 'UP', 'LEFT']

    for position, action in zip(adjacent_positions, valid_actions):
        if game_state["field"][position] == 0:
            possible_actions.append(action)

    return possible_actions