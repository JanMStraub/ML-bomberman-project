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

    self.value_estimates = np.zeros((11,4))
    self.policy = np.zeros((11,4))

    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        weights = np.random.rand(len(ACTIONS))
        self.model = weights / weights.sum()
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)



def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # todo Exploration vs exploitation
    random_prob = .1
    #print(self.train)

    """if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])"""
    
    if game_state['round']>1 and self.train:
        # Updated policy 
        #x,y = game_state['self'][3]
        state = extract_state(game_state)
        actions = ACTIONS[0:4]
        epsilon = np.random.choice([1,0], p = [0.2,0.8])
        if epsilon:
            choosed_action = np.random.choice(actions, p = [0.25,0.25,0.25,0.25])
        else:
            choosed_action = actions[np.argmax(self.policy[state,:])]
            #chosed_action = actions[np.argmax(self.policy[x,y])]    
        return choosed_action
    
    elif game_state['round']==1 and self.train:
        # Initial policy
        # First level actions 
        actions = ACTIONS[0:4]
        return np.random.choice(actions, p = [0.25,0.25,0.25,0.25])
    else:
        #x,y = game_state['self'][3]
        state = extract_state(game_state)
        actions = ACTIONS[0:4]
        #chosed_action = actions[np.argmax(self.model[x,y])]
        choosed_action = actions[np.argmax(self.policy[state,:])]
        #choosed_action = actions[int(self.model[x,y])]
        #print(x,y,choosed_action)
        #print(self.model[x,y])
        
        return choosed_action

        



    self.logger.debug("Querying model for action.")
    #print(np.random.choice(ACTIONS, p=self.model))
    return np.random.choice(ACTIONS, p=self.model)
    

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
                #field_map[x,y] += 1 
            if old_pos == (x,y) and old_pos != new_pos:
                value_estimates[x,y] += 1 
                #field_map[x,y] += 1 
                #coin_map[x,y] = 1 

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

def extract_state(old_game_state):
    
    old_pos = old_game_state['self'][3]
    field_map = old_game_state['field']

    neighbourhood = []

    left = field_map[old_pos[0]-1,old_pos[1]]
    right = field_map[old_pos[0]+1,old_pos[1]]
    top = field_map[old_pos[0],old_pos[1]-1]
    down = field_map[old_pos[0],old_pos[1]+1]

    neighbourhood.append(left)
    neighbourhood.append(right)
    neighbourhood.append(top)
    neighbourhood.append(down)

    #print(field_map[old_pos]) 

    # right top corner
    if neighbourhood == [0,-1,-1,0]:
        #print("right_top_corner")
        state = 0
    # right bottom corner
    elif neighbourhood == [0,-1,0,-1]:
        #print("right_bottom_corner")
        state = 1
    # left top corner
    elif neighbourhood == [-1,0,-1,0]:
        #print("left_top_corner")
        state = 2
    # left bottom corner
    elif neighbourhood == [-1,0,0,-1]:
        #print("left_bottom_corner")
        state = 3
    # top down
    elif neighbourhood == [0,0,-1,-1]:
        #print("top_bottom")
        state = 4
    # left right
    elif neighbourhood == [-1,-1,0,0]:
        #print("left_right")
        state = 5
    # left
    elif neighbourhood == [-1,0,0,0]:
        #print("left")
        state = 6
    # right
    elif neighbourhood == [0,-1,0,0]:
        #print("right")
        state = 7
    # top
    elif neighbourhood == [0,0,-1,0]:
        #print("top")
        state = 8
    # bottom 
    elif neighbourhood == [0,0,0,-1]:
        #print("bottom")
        state = 9
    # free
    else:
        state = 10

    return state 