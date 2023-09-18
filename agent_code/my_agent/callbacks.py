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
    self.new_pos_history = []

    #self.value_estimates = np.zeros((11,50,6))
    self.value_estimates = np.zeros((11,50,4))
    #self.policy = np.zeros((11,50,6))
    self.policy = np.zeros((11,50,4))
    self.first_visit_check = []

    for i in range(11):
        for j in range(50):
            #self.policy[i,j] = [0.15,0.15,0.15,0.15,0.2,0.2] 
            for k in range(4):
            #for k in range(5):
                #self.policy[i,j,k] = 0.2
                self.policy[i,j,k] = 0.25

    self.return_val = np.zeros((11,50,6))
    self.return_ctr = np.zeros((11,50,6))   

    self.target_coin = (0,0)     
    self.target_coins_history = []   

    self.bomb_detect_pos = None 
    self.bomb = None
    self.bomb_history = []  
    self.bomb_detect_pos_history = []
                    

    """with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)
            self.policy = self.model"""

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
        #actions = ACTIONS[0:6]
        action = np.random.choice(actions, p = self.policy[state[0],state[1],:])
        return action
    # Initial policy
    elif game_state['round']==1 and self.train:
        # First level actions 
        actions = ACTIONS[0:4]
        #actions = ACTIONS[0:6]
        return np.random.choice(actions, p = [0.25,0.25,0.25,0.25])
        return np.random.choice(actions, p = [0.15,0.15,0.15,0.15,0.2,0.2])
    else:
        state = extract_state(self,game_state)
        actions = ACTIONS[0:4]
        #actions = ACTIONS[0:6]
        choosed_action = actions[np.argmax(self.policy[state[0],state[1],:])]
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

    left_radar = [(x,old_pos[1]) for x in range(old_pos[0],old_pos[0]-4,-1)]
    right_radar = [(x,old_pos[1]) for x in range(old_pos[0],old_pos[0]+4,1)]
    top_radar = [(old_pos[0],y) for y in range(old_pos[1],old_pos[1]-4,-1)]
    down_radar = [(old_pos[0],y) for y in range(old_pos[1],old_pos[1]+4,1)]

    bombs = ([x for (x,y) in old_game_state['bombs']])
    timer = ([t for (x,t) in old_game_state['bombs']])

    #coin_x = np.sort([x for (x,y) in old_game_state['coins']])
    #coin_y = np.sort([y for (x,y) in old_game_state['coins']])

    #coin_x_distinct = list(dict.fromkeys(coin_x))
    #coin_y_distinct = list(dict.fromkeys(coin_x))

    neighbourhood.append(left)
    neighbourhood.append(right)
    neighbourhood.append(top)
    neighbourhood.append(down)

    # right top corner
    if neighbourhood == [0,-1,-1,0]:

        # Bomb radar
        # Expect: LEFT
        if bomb_radar(self,left_radar,bombs,old_pos):
            state = [0,6] 
        # Expect: DOWN
        elif bomb_radar(self,down_radar,bombs,old_pos):
            state = [0,7] 

        # Crate 
        # Expect: BOMB
        elif field_map[left_pos] == 1 or field_map[low_pos] == 1:
            state = [0,8] 

        # Coin radar
        # Expect: LEFT
        elif coin_radar(self,left_radar,old_game_state['coins']):
            state = [0,4] 
        # visited top pos
        # Expect: DOWN
        elif coin_radar(self,down_radar,old_game_state['coins']):
            state = [0,5] 

        # Visited position
        # visited left pos
        # Expect: DOWN
        elif left_pos in self.visited_positions and low_pos not in self.visited_positions:
            state = [0,0] 
        # visited low pos
        # Expect: LEFT
        elif low_pos in self.visited_positions and left_pos not in self.visited_positions:
            state = [0,1] 
        # both neighbour positions visited 
        # Expect: LEFT OR DOWN
        elif low_pos in self.visited_positions and left_pos in self.visited_positions:
            state = [0,2] 
        # both neighbour positions not visited 
        # Expect: LEFT OR DOWN
        else:
            state = [0,3] 

        
        ### Safe gold position and check with old new state if the agent moved in the coin direction 
       

    # right bottom corner
    elif neighbourhood == [0,-1,0,-1]:

        # Bomb radar
        # Expect: TOP
        if bomb_radar(self,top_radar,bombs,old_pos):
            state = [1,6] 
        # Expect: LEFT
        elif bomb_radar(self,left_radar,bombs,old_pos):
            state = [1,7] 

        # Crate 
        # Expect: BOMB
        elif field_map[left_pos] == 1 or field_map[top_pos] == 1:
            state = [1,8] 

        # Expect: UP
        elif coin_radar(self,top_radar,old_game_state['coins']):
            state = [1,4] 
        # Expect: LEFT 
        elif coin_radar(self,left_radar,old_game_state['coins']):
            state = [1,5] 


        # visited left pos
        # Expect: UP
        elif left_pos in self.visited_positions and top_pos not in self.visited_positions:
            state = [1,0] 
        # visited top pos
        # Expect: LEFT
        elif top_pos in self.visited_positions and left_pos not in self.visited_positions:
            state = [1,1] 
        # both neighbour positions visited 
        # Expect: LEFT OR UP
        elif top_pos in self.visited_positions and left_pos in self.visited_positions:
            state = [1,2]  
        # both neighbour positions not visited 
        # Expect: LEFT OR UP 
        else:
            state = [1,3] 

        
        
    # left top corner
    elif neighbourhood == [-1,0,-1,0]:

        # Bomb radar
        # Expect: RIGHT
        if bomb_radar(self,right_radar,bombs,old_pos):
            state = [2,6] 
        # Expect: DOWN
        elif bomb_radar(self,down_radar,bombs,old_pos):
            state = [2,7] 

        # Crate 
        # Expect: BOMB
        elif field_map[low_pos] == 1 or field_map[right_pos] == 1:
            state = [2,8] 
        
        # Coin radar
        # Expect: RIGHT
        elif coin_radar(self,right_radar,old_game_state['coins']):
            state = [2,4] 
        # Expect: DOWN
        elif coin_radar(self,down_radar,old_game_state['coins']):
            state = [2,5] 

        # visited right pos
        # Expect: DOWN
        elif right_pos in self.visited_positions and low_pos not in self.visited_positions:
            state = [2,0] 
        # visited low pos
        # Expect: RIGHT
        elif low_pos in self.visited_positions and right_pos not in self.visited_positions:
            state = [2,1]  
        # both neighbour positions visited 
        # Expect: RIGHT OR DOWN
        elif low_pos in self.visited_positions and right_pos in self.visited_positions:
            state = [2,2]  
        # both neighbour positions not visited 
        # Expect: RIGHT OR DOWN
        else:
            state = [2,3] 

        

    # left bottom corner
    elif neighbourhood == [-1,0,0,-1]:

        # Bomb radar
        # Expect: UP
        if bomb_radar(self,top_radar,bombs,old_pos):
            state = [3,6] 
        # Expect: RIGHT
        elif bomb_radar(self,right_radar,bombs,old_pos):
            state = [3,7] 

        # Crate 
        # Expect: BOMB
        elif field_map[top_pos] == 1 or field_map[right_pos] == 1:
            state = [3,8] 

        # Coin radar
        # Expect: UP
        elif coin_radar(self,top_radar,old_game_state['coins']):
            state = [3,4] 
        # Expect: RIGHT
        elif coin_radar(self,right_radar,old_game_state['coins']):
            state = [3,5] 

        # visited right pos
        # Expect: UP
        elif right_pos in self.visited_positions and top_pos not in self.visited_positions:
            state = [3,0] 
        # visited top pos
        # Expect: RIGHT
        elif top_pos in self.visited_positions and right_pos not in self.visited_positions:
            state = [3,1] 
        # both neighbour positions visited 
        # Expect: UP OR RIGHT
        elif top_pos in self.visited_positions and right_pos in self.visited_positions:
            state = [3,2]  
        # both neighbour positions not visited 
        # Expect: UP OR RIGHT
        else:
            state = [3,3] 

    # top down
    elif neighbourhood == [0,0,-1,-1]:

        # Bomb radar
        # Expect: LEFT
        if bomb_radar(self,left_radar,bombs,old_pos):
            state = [4,6] 
        # Expect: RIGHT
        elif bomb_radar(self,right_radar,bombs,old_pos):
            state = [4,7] 

        # Crate 
        # Expect: BOMB
        elif field_map[left_pos] == 1 or field_map[right_pos] == 1:
            state = [4,8] 

        # Coin radar
        # Expect: LEFT
        elif coin_radar(self,left_radar,old_game_state['coins']):
            state = [4,4] 
        # Expect: RIGHT
        elif coin_radar(self,right_radar,old_game_state['coins']):
            state = [4,5] 

        # visited right pos
        # Expect: LEFT
        elif right_pos in self.visited_positions and left_pos not in self.visited_positions:
            state = [4,0] 
        # visited left pos
        # Expect: RIGHT
        elif left_pos in self.visited_positions and right_pos not in self.visited_positions:
            state = [4,1]
        # both neighbour positions vistiid 
        # Expect: LEFT OR RIGHT
        elif left_pos in self.visited_positions and right_pos in self.visited_positions:
            state = [4,2] 
        # both neighbour positions not visited 
        # Expect: LEFT OR RIGHT
        else:
            state = [4,3]

    # left right
    elif neighbourhood == [-1,-1,0,0]:

        # Bomb radar
        # Expect: UP
        if bomb_radar(self,top_radar,bombs,old_pos):
            state = [5,6] 
        # Expect: DOWN
        elif bomb_radar(self,down_radar,bombs,old_pos):
            state = [5,7] 

        # Crate 
        # Expect: BOMB
        elif field_map[top_pos] == 1 or field_map[low_pos] == 1:
            state = [5,8] 

        # Coin radar
        # Expect: UP
        elif coin_radar(self,top_radar,old_game_state['coins']):
            state = [5,4] 
        # Expect: DOWN
        elif coin_radar(self,down_radar,old_game_state['coins']):
            state = [5,5] 

        # visited top pos
        # Expect: DOWN
        elif top_pos in self.visited_positions and low_pos not in self.visited_positions:
            state = [5,0]
        # visited low pos
        # Expect: UP
        elif low_pos in self.visited_positions and top_pos not in self.visited_positions:
            state = [5,1] 
        # both neighbour positions visited 
        # Expect: UP OR DOWN
        elif low_pos in self.visited_positions and top_pos in self.visited_positions:
            state = [5,2] 
        # both neighbour positions visited or not visited 
        # Expect: UP OR DOWN
        else:
            state = [5,3]

    # left
    elif neighbourhood == [-1,0,0,0]:

        # Bomb radar
        # Expect: UP
        if bomb_radar(self,top_radar,bombs,old_pos):
            state = [6,13] 
        # Expect: RIGHT 
        elif bomb_radar(self,right_radar,bombs,old_pos):
            state = [6,14] 
        # Expect: DOWN
        elif bomb_radar(self,down_radar,bombs,old_pos):
            state = [6,15] 

        # Crate 
        # Expect: BOMB
        elif field_map[top_pos] == 1 or field_map[low_pos] == 1 or field_map[right_pos] == 1:
            state = [6,16] 

        # Coin radar
        # Expect: UP
        elif coin_radar(self,top_radar,old_game_state['coins']):
            state = [6,10] 
        # Expect: RIGHT
        elif coin_radar(self,right_radar,old_game_state['coins']):
            state = [6,11] 
        # Expect: DOWN
        elif coin_radar(self,down_radar,old_game_state['coins']):
            state = [6,12] 

        # visited top pos and right pos 
        # Expect: DOWN
        elif top_pos in self.visited_positions and right_pos in self.visited_positions and low_pos not in self.visited_positions:
            state = [6,0]
        # visited top pos and low pos 
        # Expect: RIGHT
        elif top_pos in self.visited_positions and right_pos not in self.visited_positions and low_pos in self.visited_positions:
            state = [6,1] 
        # visited only top pos
        # Expect: RIGHT OR DOWN
        elif top_pos in self.visited_positions and right_pos not in self.visited_positions and low_pos not in self.visited_positions:
            state = [6,2]

        # visited low pos and right pos 
        # Expect: UP
        elif low_pos in self.visited_positions and right_pos in self.visited_positions and top_pos not in self.visited_positions:            
            state = [6,3]
        # visited low pos and top pos 
        # Expect: RIGHT
        elif low_pos in self.visited_positions and right_pos not in self.visited_positions and top_pos in self.visited_positions:
            state = [6,4] 
        # visited only low pos
        # Expect: RIGHT OR UP
        elif low_pos in self.visited_positions and right_pos not in self.visited_positions and top_pos not in self.visited_positions:            
            state = [6,5]

        # visited right pos and top pos
        # Expect: DOWN
        elif right_pos in self.visited_positions and top_pos in self.visited_positions and low_pos not in self.visited_positions:
            state = [6,6]
        # visited right pos and low pos
        # Expect: UP
        elif right_pos in self.visited_positions and top_pos not in self.visited_positions and low_pos in self.visited_positions:
            state = [6,7] 
        # visited only right pos
        # Expect: UP OR DOWN
        elif right_pos in self.visited_positions and top_pos not in self.visited_positions and low_pos not in self.visited_positions:
            state = [6,8]

        # all neighbour positions vistied or not visited 
        # Expect: RIGHT OR UP OR DOWN
        else:
            state = [6,9]


    # right
    elif neighbourhood == [0,-1,0,0]:

        # Bomb radar
        # Expect: UP
        if bomb_radar(self,top_radar,bombs,old_pos):
            state = [7,13] 
        # Expect: LEFT
        elif bomb_radar(self,left_radar,bombs,old_pos):
            state = [7,14] 
        # Expect: DOWN
        elif bomb_radar(self,down_radar,bombs,old_pos):
            state = [7,15] 

        # Crate 
        # Expect: BOMB
        elif field_map[top_pos] == 1 or field_map[low_pos] == 1 or field_map[left_pos] == 1:
            state = [7,16] 

        # Coin radar
        # Expect: UP
        elif coin_radar(self,top_radar,old_game_state['coins']):
            state = [7,10] 
        # Expect: LEFT
        elif coin_radar(self,left_radar,old_game_state['coins']):
            state = [7,11] 
        # Expect: DOWN
        elif coin_radar(self,down_radar,old_game_state['coins']):
            state = [7,12] 

        #print("right")
        # visited top pos
        # Expect: DOWN
        elif top_pos in self.visited_positions and left_pos in self.visited_positions and low_pos not in self.visited_positions:
            state = [7,0]
        # visited low pos
        # Expect: LEFT
        elif top_pos in self.visited_positions and left_pos not in self.visited_positions and low_pos in self.visited_positions:
            state = [7,1] 
        # visited left pos
        # Expect: LEFT OR DOWN
        elif top_pos in self.visited_positions and left_pos not in self.visited_positions and low_pos not in self.visited_positions:
            state = [7,2]

        # visited low pos
        # Expect: UP
        elif low_pos in self.visited_positions and left_pos in self.visited_positions and top_pos not in self.visited_positions:
            state = [7,3]
        # visited low pos
        # Expect: LEFT
        elif low_pos in self.visited_positions and left_pos not in self.visited_positions and top_pos in self.visited_positions:
            state = [7,4]
        # visited left pos
        # Expect: LEFT OR UP
        elif low_pos in self.visited_positions and left_pos not in self.visited_positions and top_pos not in self.visited_positions:
            state = [7,5]

        # visited left pos
        # Expect: DOWN
        elif left_pos in self.visited_positions and top_pos in self.visited_positions and low_pos not in self.visited_positions:
            state = [7,6]
        # visited low pos
        # Expect: UP
        elif left_pos in self.visited_positions and top_pos not in self.visited_positions and low_pos in self.visited_positions:
            state = [7,7]
        # visited left pos
        # Expect: UP OR DOWN
        elif left_pos in self.visited_positions and top_pos not in self.visited_positions and low_pos not in self.visited_positions:
            state = [7,8]

        # both neighbour positions vistied or not visited 
        # Expect: UP OR DOWN OR LEFT
        else:
            state = [7,9]


    # top
    elif neighbourhood == [0,0,-1,0]:

        # Bomb radar
        # Expect: RIGHT
        if bomb_radar(self,right_radar,bombs,old_pos):
            state = [8,13] 
        # Expect: LEFT
        elif bomb_radar(self,left_radar,bombs,old_pos):
            state = [8,14] 
        # Expect: DOWN
        elif bomb_radar(self,down_radar,bombs,old_pos):
            state = [8,15] 

        # Crate 
        # Expect: BOMB
        elif field_map[left_pos] == 1 or field_map[low_pos] == 1 or field_map[right_pos] == 1:
            state = [8,16] 

        # Coin radar
        # Expect: RIGHT
        elif coin_radar(self,right_radar,old_game_state['coins']):
            state = [8,10] 
        # Expect: LEFT
        elif coin_radar(self,left_radar,old_game_state['coins']):
            state = [8,11] 
        # Expect: DOWN
        elif coin_radar(self,down_radar,old_game_state['coins']):
            state = [8,12] 

        # visited right pos
        # Expect: DOWN
        elif right_pos in self.visited_positions and left_pos in self.visited_positions and low_pos not in self.visited_positions:
            state = [8,0]
        # visited low pos
        # Expect: LEFT
        elif right_pos in self.visited_positions and left_pos not in self.visited_positions and low_pos in self.visited_positions:
            state = [8,1] 
        # visited left pos
        # Expect: LEFT OR DOWN
        elif right_pos in self.visited_positions and left_pos not in self.visited_positions and low_pos not in self.visited_positions:
            state = [8,2]

        # visited right pos
        # Expect: RIGHT
        elif low_pos in self.visited_positions and left_pos in self.visited_positions and right_pos not in self.visited_positions:
            state = [8,3]
        # visited low pos
        # Expect: LEFT
        elif low_pos in self.visited_positions and left_pos not in self.visited_positions and right_pos in self.visited_positions:
            state = [8,4] 
        # visited left pos
        # Expect: RIGHT OR LEFT
        elif low_pos in self.visited_positions and left_pos not in self.visited_positions and right_pos not in self.visited_positions:
            state = [8,5]

        # visited right pos
        # Expect: DOWN
        elif left_pos in self.visited_positions and right_pos in self.visited_positions and low_pos not in self.visited_positions:
            state = [8,6]
        # visited low pos
        # Expect: RIGHT
        elif left_pos in self.visited_positions and right_pos not in self.visited_positions and low_pos in self.visited_positions:
            state = [8,7]
        # visited left pos
        # Expect: RIGHT OR DOWN
        elif left_pos in self.visited_positions and right_pos not in self.visited_positions and low_pos not in self.visited_positions:
            state = [8,8]

        # both neighbour positions vistied or not visited 
        # Expect: RIGHT OR DOWN OR LEFT 
        else:
            state = [8,9]

    # bottom 
    elif neighbourhood == [0,0,0,-1]:

        # Bomb radar
        # Expect: RIGHT
        if bomb_radar(self,right_radar,bombs,old_pos):
            state = [9,13] 
        # Expect: LEFT
        elif bomb_radar(self,left_radar,bombs,old_pos):
            state = [9,14] 
        # Expect: UP
        elif bomb_radar(self,top_radar,bombs,old_pos):
            state = [9,15] 

        # Crate 
        # Expect: BOMB
        elif field_map[top_pos] == 1 or field_map[left_pos] == 1 or field_map[right_pos] == 1:
            state = [9,16] 

        # Coin radar
        # Expect: RIGHT
        elif coin_radar(self,right_radar,old_game_state['coins']):
            state = [9,10] 
        # Expect: LEFT
        elif coin_radar(self,left_radar,old_game_state['coins']):
            state = [9,11] 
        # Expect: UP
        elif coin_radar(self,top_radar,old_game_state['coins']):
            state = [9,12] 
        
        # Expect: UP
        elif right_pos in self.visited_positions and left_pos in self.visited_positions and top_pos not in self.visited_positions:
            state = [9,0]
        # visited low pos
        # Expect: LEFT
        elif right_pos in self.visited_positions and left_pos not in self.visited_positions and top_pos in self.visited_positions:
            state = [9,1]
        # visited left pos
        # Expect: UP OR LEFT
        elif right_pos in self.visited_positions and left_pos not in self.visited_positions and top_pos not in self.visited_positions:
            state = [9,2]

        # Expect: RIGHT
        elif top_pos in self.visited_positions and left_pos in self.visited_positions and right_pos not in self.visited_positions:
            state = [9,3]
        # visited low pos
        # Expect: LEFT
        elif top_pos in self.visited_positions and left_pos not in self.visited_positions and right_pos in self.visited_positions:
            state = [9,4]
        # visited left pos
        # Expect: LEFT OR RIGHT
        elif top_pos in self.visited_positions and left_pos not in self.visited_positions and right_pos not in self.visited_positions:
            state = [9,5]

        # Expect: UP
        elif left_pos in self.visited_positions and right_pos in self.visited_positions and top_pos not in self.visited_positions:
            state = [9,6]
        # visited low pos
        # Expect: RIGHT
        elif left_pos in self.visited_positions and right_pos not in self.visited_positions and top_pos in self.visited_positions:
            state = [9,7]
        # visited left pos
        # Expect: RIGHT OR UP 
        elif left_pos in self.visited_positions and right_pos not in self.visited_positions and top_pos not in self.visited_positions:
            state = [9,8]

        # both neighbour positions vistied or not visited 
        # Expect: RIGHT OR UP OR LEFT 
        else:
            state = [9,9]

    # free
    else:
        # Bomb radar
        # Expect: RIGHT
        if bomb_radar(self,right_radar,bombs,old_pos):
            state = [10,37] 
        # Expect: LEFT
        elif bomb_radar(self,left_radar,bombs,old_pos):
            state = [10,38] 
        # Expect: UP
        elif bomb_radar(self,top_radar,bombs,old_pos):
            state = [10,39] 
        # Expect: DOWN
        elif bomb_radar(self,down_radar,bombs,old_pos):
            state = [10,40] 

        # Crate 
        # Expect: BOMB
        elif field_map[top_pos] == 1 or field_map[low_pos] == 1 or field_map[left_pos] == 1 or field_map[right_pos] == 1:
            state = [10,41] 
        
        # Coin radar
        # Expect: RIGHT
        elif coin_radar(self,right_radar,old_game_state['coins']):
            state = [10,33] 
        # Expect: LEFT
        elif coin_radar(self,left_radar,old_game_state['coins']):
            state = [10,34] 
        # Expect: UP
        elif coin_radar(self,top_radar,old_game_state['coins']):
            state = [10,35] 
        # Expect: DOWN
        elif coin_radar(self,down_radar,old_game_state['coins']):
            state = [10,36] 

        # Expect: Right 
        elif right_pos not in self.visited_positions:
            state = [10,0]
        # Expect: LEFT 
        elif left_pos not in self.visited_positions:
            state = [10,1]
        # Expect: UP
        elif top_pos not in self.visited_positions:
            state = [10,2]
        # Expect: DOWN
        elif low_pos not in self.visited_positions:
            state = [10,3]
        else:
            state = [10,32]

    return state 

def coin_radar(self,radar,coin):
    for location in radar:
        if location in coin:
            self.target_coin = location
            return True
    return False

def bomb_radar(self,radar,bombs,old_pos):
    for location in radar:
        if location in bombs:
            self.bomb = location
            #if t > 0 ? 
            self.bomb_detect_pos = old_pos
            return True
    return False