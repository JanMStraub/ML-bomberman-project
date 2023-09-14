import os
import pickle
import random

import numpy as np


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

    self.state_history = []
    self.action_history = []
    self.event_history = []
    self.reward_history = []
    self.visited_positions = []

    self.value_estimates = np.zeros((97,4))
    self.policy = np.zeros((97,4))

    for i in range(97):
        for j in range(4):
            self.policy[i,j] = 0.25


    self.returns = [[[],[],[],[]],
                    [[],[],[],[]],
                    [[],[],[],[]],
                    [[],[],[],[]],
                    [[],[],[],[]],
                    [[],[],[],[]],
                    [[],[],[],[]],
                    [[],[],[],[]],
                    [[],[],[],[]],
                    [[],[],[],[]],
                    [[],[],[],[]],
                    [[],[],[],[]],
                    [[],[],[],[]],
                    [[],[],[],[]],
                    [[],[],[],[]],
                    [[],[],[],[]],
                    [[],[],[],[]],
                    [[],[],[],[]],
                    [[],[],[],[]],
                    [[],[],[],[]],
                    [[],[],[],[]],
                    [[],[],[],[]],
                    [[],[],[],[]],
                    [[],[],[],[]],
                    [[],[],[],[]],
                    [[],[],[],[]],
                    [[],[],[],[]],
                    [[],[],[],[]],
                    [[],[],[],[]],
                    [[],[],[],[]],
                    [[],[],[],[]],
                    [[],[],[],[]],
                    [[],[],[],[]],
                    [[],[],[],[]],
                    [[],[],[],[]],
                    [[],[],[],[]],
                    [[],[],[],[]],
                    [[],[],[],[]],
                    [[],[],[],[]],
                    [[],[],[],[]],
                    [[],[],[],[]],
                    [[],[],[],[]],
                    [[],[],[],[]],
                    [[],[],[],[]],
                    [[],[],[],[]],
                    [[],[],[],[]],
                    [[],[],[],[]],
                    [[],[],[],[]],
                    [[],[],[],[]],
                    [[],[],[],[]],
                    [[],[],[],[]],
                    [[],[],[],[]],
                    [[],[],[],[]],
                    [[],[],[],[]],
                    [[],[],[],[]],
                    [[],[],[],[]],
                    [[],[],[],[]],
                    [[],[],[],[]],
                    [[],[],[],[]],
                    [[],[],[],[]],
                    [[],[],[],[]],
                    [[],[],[],[]],
                    [[],[],[],[]],
                    [[],[],[],[]],
                    [[],[],[],[]],
                    [[],[],[],[]],
                    [[],[],[],[]],
                    [[],[],[],[]],
                    [[],[],[],[]],
                    [[],[],[],[]],
                    [[],[],[],[]],
                    [[],[],[],[]],
                    [[],[],[],[]],
                    [[],[],[],[]],
                    [[],[],[],[]],
                    [[],[],[],[]],
                    [[],[],[],[]],
                    [[],[],[],[]],
                    [[],[],[],[]],
                    [[],[],[],[]],
                    [[],[],[],[]],
                    [[],[],[],[]],
                    [[],[],[],[]],
                    [[],[],[],[]],
                    [[],[],[],[]],
                    [[],[],[],[]],
                    [[],[],[],[]],
                    [[],[],[],[]],
                    [[],[],[],[]],
                    [[],[],[],[]],
                    [[],[],[],[]],
                    [[],[],[],[]],
                    [[],[],[],[]],
                    [[],[],[],[]],
                    [[],[],[],[]],
                    [[],[],[],[]],
                    [[],[],[],[]]]                
                    

    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        weights = np.random.rand(len(ACTIONS))
        self.model = weights / weights.sum()
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)
            self.policy = self.model



def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    if game_state['round']>1 and self.train:
        state = extract_state(self,game_state)
        actions = ACTIONS[0:4]
        action = np.random.choice(actions, p = self.policy[state,:])
        return action
    # Initial policy
    elif game_state['round']==1 and self.train:
        # First level actions 
        actions = ACTIONS[0:4]
        return np.random.choice(actions, p = [0.25,0.25,0.25,0.25])
    else:
        state = extract_state(self,game_state)
        actions = ACTIONS[0:4]
        choosed_action = actions[np.argmax(self.policy[state,:])]
        return choosed_action
   

def state_to_features(value_estimates, old_game_state: dict, new_game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    if old_game_state is None:
        return None
    
    old_pos = old_game_state['self'][3]
    new_pos = new_game_state['self'][3]
    field_map = old_game_state['field']
    coin_map = np.zeros(field_map.shape)

    for x in range(field_map.shape[0]):
        for y in range(field_map.shape[1]):
            value_estimates[x,y] += field_map[x,y]
            if (x,y) in old_game_state['coins']:
                value_estimates[x,y] += 1 
            if old_pos == (x,y) and old_pos != new_pos:
                value_estimates[x,y] += 1 

    return field_map

    # Create features by using field information and coin location 
    # For example, you could construct several channels of equal shape, ...
    channels = []
    channels.append(field_map)
    channels.append(coin_map)
    # concatenate them as a feature tensor (they must have the same shape), ...
    stacked_channels = np.stack(channels)
    # and return them as a vector
    return stacked_channels.reshape(-1)

def extract_state(self,old_game_state):
    """
    First level for collecting coins has 11 different states which describe 
    the neighbourhood of the current agent position
    """
    old_pos = old_game_state['self'][3]
    field_map = old_game_state['field']

    top_pos = (old_pos[0],old_pos[1]-1)
    low_pos = (old_pos[0],old_pos[1]+1)
    left_pos = (old_pos[0]-1,old_pos[1])
    right_pos = (old_pos[0]+1,old_pos[1])

    neighbourhood = []

    left = field_map[old_pos[0]-1,old_pos[1]]
    right = field_map[old_pos[0]+1,old_pos[1]]
    top = field_map[old_pos[0],old_pos[1]-1]
    down = field_map[old_pos[0],old_pos[1]+1]

    neighbourhood.append(left)
    neighbourhood.append(right)
    neighbourhood.append(top)
    neighbourhood.append(down)

    # right top corner
    if neighbourhood == [0,-1,-1,0]:
        #print("right_top_corner")
        # visited left pos
        if left_pos in self.visited_positions and low_pos not in self.visited_positions:
            state = 0 
        # visited low pos
        elif low_pos in self.visited_positions and left_pos not in self.visited_positions:
            state = 1 
        # both neighbour positions visited 
        elif low_pos in self.visited_positions and left_pos in self.visited_positions:
            state = 2 
        # both neighbour positions not visited 
        else:
            state = 3
    # right bottom corner
    elif neighbourhood == [0,-1,0,-1]:
        #print("right_bottom_corner")
        # visited left pos
        if left_pos in self.visited_positions and top_pos not in self.visited_positions:
            state = 4 
        # visited top pos
        elif top_pos in self.visited_positions and left_pos not in self.visited_positions:
            state = 5 
        # both neighbour positions visited 
        elif top_pos in self.visited_positions and left_pos in self.visited_positions:
            state = 6 
        # both neighbour positions not visited 
        else:
            state = 7
    # left top corner
    elif neighbourhood == [-1,0,-1,0]:
        #print("left_top_corner")
        # visited right pos
        if right_pos in self.visited_positions and low_pos not in self.visited_positions:
            state = 8 
        # visited low pos
        elif low_pos in self.visited_positions and right_pos not in self.visited_positions:
            state = 9 
        # both neighbour positions visited 
        elif low_pos in self.visited_positions and right_pos in self.visited_positions:
            state = 10 
        # both neighbour positions not visited 
        else:
            state = 11
    # left bottom corner
    elif neighbourhood == [-1,0,0,-1]:
        #print("left_bottom_corner")
        # visited right pos
        if right_pos in self.visited_positions and top_pos not in self.visited_positions:
            state = 12
        # visited top pos
        elif top_pos in self.visited_positions and right_pos not in self.visited_positions:
            state = 13 
        # both neighbour positions visited 
        elif top_pos in self.visited_positions and right_pos in self.visited_positions:
            state = 14 
        # both neighbour positions not visited 
        else:
            state = 15
    # top down
    elif neighbourhood == [0,0,-1,-1]:
        #print("top_bottom")
        # visited right pos
        if right_pos in self.visited_positions and left_pos not in self.visited_positions:
            state = 16
        # visited left pos
        elif left_pos in self.visited_positions and right_pos not in self.visited_positions:
            state = 17 
        # both neighbour positions vistiid 
        elif left_pos in self.visited_positions and right_pos in self.visited_positions:
            state = 18 
        # both neighbour positions not visited 
        else:
            state = 19
    # left right
    elif neighbourhood == [-1,-1,0,0]:
        #print("left_right")
        # visited top pos
        if top_pos in self.visited_positions and low_pos not in self.visited_positions:
            state = 20
        # visited low pos
        elif low_pos in self.visited_positions and top_pos not in self.visited_positions:
            state = 21 
        # both neighbour positions visited 
        elif low_pos in self.visited_positions and top_pos in self.visited_positions:
            state = 22 
        # both neighbour positions visited or not visited 
        else:
            state = 23
    # left
    elif neighbourhood == [-1,0,0,0]:
        #print("left")
        # visited top pos and right pos 
        if top_pos in self.visited_positions and right_pos in self.visited_positions and low_pos not in self.visited_positions:
            state = 24
        # visited top pos and left pos 
        elif top_pos in self.visited_positions and right_pos not in self.visited_positions and low_pos in self.visited_positions:
            state = 25 
        # visited only top pos
        elif top_pos in self.visited_positions and right_pos not in self.visited_positions and low_pos not in self.visited_positions:
            state = 26

        # visited low pos and right pos 
        elif low_pos in self.visited_positions and right_pos in self.visited_positions and top_pos not in self.visited_positions:            
            state = 27
        # visited low pos and top pos 
        elif low_pos in self.visited_positions and right_pos not in self.visited_positions and top_pos in self.visited_positions:
            state = 28 
        # visited only low pos
        elif low_pos in self.visited_positions and right_pos not in self.visited_positions and top_pos not in self.visited_positions:            
            state = 29

        # visited right pos and top pos
        elif right_pos in self.visited_positions and top_pos in self.visited_positions and low_pos not in self.visited_positions:
            state = 30
        # visited right pos and low pos
        elif right_pos in self.visited_positions and top_pos not in self.visited_positions and low_pos in self.visited_positions:
            state = 31 
        # visited only right pos
        elif right_pos in self.visited_positions and top_pos not in self.visited_positions and low_pos not in self.visited_positions:
            state = 32

        # all neighbour positions vistied or not visited 
        else:
            state = 33
    # right
    elif neighbourhood == [0,-1,0,0]:
        #print("right")
        # visited top pos
        if top_pos in self.visited_positions and left_pos in self.visited_positions and low_pos not in self.visited_positions:
            state = 34
        # visited low pos
        elif top_pos in self.visited_positions and left_pos not in self.visited_positions and low_pos in self.visited_positions:
            state = 35 
        # visited left pos
        elif top_pos in self.visited_positions and left_pos not in self.visited_positions and low_pos not in self.visited_positions:
            state = 36

        # visited low pos
        elif low_pos in self.visited_positions and left_pos in self.visited_positions and top_pos not in self.visited_positions:
            state = 37
        # visited low pos
        elif low_pos in self.visited_positions and left_pos not in self.visited_positions and top_pos in self.visited_positions:
            state = 38 
        # visited left pos
        elif low_pos in self.visited_positions and left_pos not in self.visited_positions and top_pos not in self.visited_positions:
            state = 39

        # visited left pos
        elif left_pos in self.visited_positions and top_pos in self.visited_positions and low_pos not in self.visited_positions:
            state = 40
        # visited low pos
        elif left_pos in self.visited_positions and top_pos not in self.visited_positions and low_pos in self.visited_positions:
            state = 41 
        # visited left pos
        elif left_pos in self.visited_positions and top_pos not in self.visited_positions and low_pos not in self.visited_positions:
            state = 42

        # both neighbour positions vistied or not visited 
        else:
            state = 43
    # top
    elif neighbourhood == [0,0,-1,0]:
        #print("top")
        # visited right pos
        if right_pos in self.visited_positions and left_pos in self.visited_positions and low_pos not in self.visited_positions:
            state = 44
        # visited low pos
        elif right_pos in self.visited_positions and left_pos not in self.visited_positions and low_pos in self.visited_positions:
            state = 45 
        # visited left pos
        elif right_pos in self.visited_positions and left_pos not in self.visited_positions and low_pos not in self.visited_positions:
            state = 46

        # visited right pos
        elif low_pos in self.visited_positions and left_pos in self.visited_positions and right_pos not in self.visited_positions:
            state = 47
        # visited low pos
        elif low_pos in self.visited_positions and left_pos not in self.visited_positions and right_pos in self.visited_positions:
            state = 48 
        # visited left pos
        elif low_pos in self.visited_positions and left_pos not in self.visited_positions and right_pos not in self.visited_positions:
            state = 49

        # visited right pos
        elif left_pos in self.visited_positions and right_pos in self.visited_positions and low_pos not in self.visited_positions:
            state = 50
        # visited low pos
        elif left_pos in self.visited_positions and right_pos not in self.visited_positions and low_pos in self.visited_positions:
            state = 51 
        # visited left pos
        elif left_pos in self.visited_positions and right_pos not in self.visited_positions and low_pos not in self.visited_positions:
            state = 52

        # both neighbour positions vistied or not visited 
        else:
            state = 53
    # bottom 
    elif neighbourhood == [0,0,0,-1]:
        #print("bottom")
        if right_pos in self.visited_positions and left_pos not in self.visited_positions and top_pos not in self.visited_positions:
            state = 54
        # visited low pos
        elif right_pos in self.visited_positions and left_pos not in self.visited_positions and top_pos not in self.visited_positions:
            state = 55
        # visited left pos
        elif right_pos in self.visited_positions and left_pos not in self.visited_positions and top_pos not in self.visited_positions:
            state = 56

        elif top_pos in self.visited_positions and left_pos not in self.visited_positions and right_pos not in self.visited_positions:
            state = 57
        # visited low pos
        elif top_pos in self.visited_positions and left_pos not in self.visited_positions and right_pos not in self.visited_positions:
            state = 58
        # visited left pos
        elif top_pos in self.visited_positions and left_pos not in self.visited_positions and right_pos not in self.visited_positions:
            state = 59

        elif left_pos in self.visited_positions and right_pos not in self.visited_positions and top_pos not in self.visited_positions:
            state = 60
        # visited low pos
        elif left_pos in self.visited_positions and right_pos not in self.visited_positions and top_pos not in self.visited_positions:
            state = 61
        # visited left pos
        elif left_pos in self.visited_positions and right_pos not in self.visited_positions and top_pos not in self.visited_positions:
            state = 62

        # both neighbour positions vistied or not visited 
        else:
            state = 63
    # free
    else:
        if right_pos in self.visited_positions and left_pos not in self.visited_positions and top_pos not in self.visited_positions and low_pos not in self.visited_positions:
            state = 64
        elif right_pos in self.visited_positions and left_pos not in self.visited_positions and top_pos not in self.visited_positions and low_pos in self.visited_positions:
            state = 65
        elif right_pos in self.visited_positions and left_pos not in self.visited_positions and top_pos in self.visited_positions and low_pos not in self.visited_positions:
            state = 66
        elif right_pos in self.visited_positions and left_pos not in self.visited_positions and top_pos in self.visited_positions and low_pos in self.visited_positions:
            state = 67
        elif right_pos in self.visited_positions and left_pos in self.visited_positions and top_pos not in self.visited_positions and low_pos not in self.visited_positions:
            state = 68
        elif right_pos in self.visited_positions and left_pos in self.visited_positions and top_pos not in self.visited_positions and low_pos in self.visited_positions:
            state = 69
        elif right_pos in self.visited_positions and left_pos in self.visited_positions and top_pos in self.visited_positions and low_pos in self.visited_positions:
            state = 70
        elif right_pos in self.visited_positions and left_pos in self.visited_positions and top_pos in self.visited_positions and low_pos not in self.visited_positions:
            state = 71

        elif left_pos in self.visited_positions and right_pos not in self.visited_positions and top_pos not in self.visited_positions and low_pos not in self.visited_positions:
            state = 72
        elif left_pos in self.visited_positions and right_pos not in self.visited_positions and top_pos not in self.visited_positions and low_pos in self.visited_positions:
            state = 73
        elif left_pos in self.visited_positions and right_pos not in self.visited_positions and top_pos in self.visited_positions and low_pos not in self.visited_positions:
            state = 74
        elif left_pos in self.visited_positions and right_pos not in self.visited_positions and top_pos in self.visited_positions and low_pos in self.visited_positions:
            state = 75
        elif left_pos in self.visited_positions and right_pos in self.visited_positions and top_pos not in self.visited_positions and low_pos not in self.visited_positions:
            state = 76
        elif left_pos in self.visited_positions and right_pos in self.visited_positions and top_pos not in self.visited_positions and low_pos in self.visited_positions:
            state = 77
        elif left_pos in self.visited_positions and right_pos in self.visited_positions and top_pos in self.visited_positions and low_pos in self.visited_positions:
            state = 78
        elif left_pos in self.visited_positions and right_pos in self.visited_positions and top_pos in self.visited_positions and low_pos not in self.visited_positions:
            state = 79

        elif top_pos in self.visited_positions and right_pos not in self.visited_positions and left_pos not in self.visited_positions and low_pos not in self.visited_positions:
            state = 80
        elif top_pos in self.visited_positions and right_pos not in self.visited_positions and left_pos not in self.visited_positions and low_pos in self.visited_positions:
            state = 81
        elif top_pos in self.visited_positions and right_pos not in self.visited_positions and left_pos in self.visited_positions and low_pos not in self.visited_positions:
            state = 82
        elif top_pos in self.visited_positions and right_pos not in self.visited_positions and left_pos in self.visited_positions and low_pos in self.visited_positions:
            state = 83
        elif top_pos in self.visited_positions and right_pos in self.visited_positions and left_pos not in self.visited_positions and low_pos not in self.visited_positions:
            state = 84
        elif top_pos in self.visited_positions and right_pos in self.visited_positions and left_pos not in self.visited_positions and low_pos in self.visited_positions:
            state = 85
        elif top_pos in self.visited_positions and right_pos in self.visited_positions and left_pos in self.visited_positions and low_pos in self.visited_positions:
            state = 86
        elif top_pos in self.visited_positions and right_pos in self.visited_positions and left_pos in self.visited_positions and low_pos not in self.visited_positions:
            state = 87

        elif low_pos in self.visited_positions and right_pos not in self.visited_positions and left_pos not in self.visited_positions and top_pos not in self.visited_positions:
            state = 88
        elif low_pos in self.visited_positions and right_pos not in self.visited_positions and left_pos not in self.visited_positions and top_pos in self.visited_positions:
            state = 89
        elif low_pos in self.visited_positions and right_pos not in self.visited_positions and left_pos in self.visited_positions and top_pos not in self.visited_positions:
            state = 90
        elif low_pos in self.visited_positions and right_pos not in self.visited_positions and left_pos in self.visited_positions and top_pos in self.visited_positions:
            state = 91
        elif low_pos in self.visited_positions and right_pos in self.visited_positions and left_pos not in self.visited_positions and top_pos not in self.visited_positions:
            state = 92
        elif low_pos in self.visited_positions and right_pos in self.visited_positions and left_pos not in self.visited_positions and top_pos in self.visited_positions:
            state = 93
        elif low_pos in self.visited_positions and right_pos in self.visited_positions and left_pos in self.visited_positions and top_pos in self.visited_positions:
            state = 94
        elif low_pos in self.visited_positions and right_pos in self.visited_positions and left_pos in self.visited_positions and top_pos not in self.visited_positions:
            state = 95
        else:
            state = 96

    return state 