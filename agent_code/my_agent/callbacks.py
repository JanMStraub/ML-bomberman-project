import os
import pickle
import random
import math

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
    self.new_policy = []
    self.new_value_function = []
    self.feature_list = []

    self.target_coins_history = []   
    self.target_crates_history = []
    self.target_enemy_history = []

    self.bomb_history = []  
    self.bomb_timer_history = []

    self.n_sarsa_ctr = 0
                    

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
            self.new_policy = self.model
            self.feature_list = [row[6] for row in self.new_policy]
            

def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    if game_state['round']>1 and self.train:
        #state = states(self,game_state,False)
        actions = ACTIONS[0:6]
        feature = state_to_features(self,[],game_state,game_state,False)
        if feature in self.feature_list:
            index = self.feature_list.index(feature)
            action = np.random.choice(actions, p = self.new_policy[index][:6])
        else:
            return np.random.choice(actions, p = [0.25,0.25,0.25,0.25,0.0,0.0])
        return action 
    # Initial policy
    elif game_state['round']==1 and self.train:
        # First level actions 
        actions = ACTIONS[0:6]
        return np.random.choice(actions, p = [0.2,0.2,0.2,0.2,0.2,0.0])
    else:
        self.visited_positions.append((game_state['self'][3]))
        actions = ACTIONS[0:6]
        feature = state_to_features(self,[],game_state,game_state,False)
        if feature in self.feature_list:
            index = self.feature_list.index(feature)
            action = actions[np.argmax(self.new_policy[index][:6])]
        else:
            return np.random.choice(actions, p = [0.25,0.25,0.25,0.25,0.0,0.0])
        return action 
   

def coin_radar(self,radar,coin,train):
    for location in radar:
        if location in coin:
            if train:
                self.target_coin = location
            return True
    return False

def bomb_radar(self,radar,bombs,train):
    for location in radar:
        if location in bombs:
            return True
    return False

def crate_radar(self,radar,crates,train):
    for location in radar:
        if location in crates:
            return True
    return False

def wall_radar(self,radar,wall,train):
    for location in radar:
        if location in wall:
            #if train:
                #self.target_coin = location
            return True
    return False


def enemy_radar(self,radar,enemy,train):
    for location in radar:
        if location in enemy:
            return True
    return False

def avoid_check(newPos, bomb):
    if bomb:
        if euclidean_distance(newPos,bomb)%1 != 0:
            av_check = True
        else:
            av_check = False 
    else:
        av_check = False 
    return av_check

def euclidean_distance(coord1, coord2):
    x1, y1 = coord1
    x2, y2 = coord2
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance


def state_to_features(self,value_estimates, old_game_state: dict, new_game_state: dict,train) -> np.array:
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


    # Check environment
    pos = old_game_state['self'][3]
    field_map = old_game_state['field']

    if old_game_state['explosion_map'].max()>0:
        explosion_map = []
        for x in range(17):
            for y in range(17):
                if old_game_state['explosion_map'][x,y] == 1:
                    explosion_map.append((x,y))

    left_radar = [(x,pos[1]) for x in range(pos[0],pos[0]-4,-1)]
    right_radar = [(x,pos[1]) for x in range(pos[0],pos[0]+4,1)]
    top_radar = [(pos[0],y) for y in range(pos[1],pos[1]-4,-1)]
    down_radar = [(pos[0],y) for y in range(pos[1],pos[1]+4,1)]

    # Get bomb locations
    bombs = ([x for (x,y) in old_game_state['bombs']])
    timer = ([t for (x,t) in old_game_state['bombs']])

    # Enemy agents 
    enemies = old_game_state['others']
    enemy_pos = []
    for enemy in enemies:
        enemy_pos.append(enemy[3])

    # Get Crate locations 
    crates = []
    walls = []
    for x in range(17):
        for y in range(17):
            if old_game_state['field'][x,y] == 1:
                crates.append((x,y))
            elif old_game_state['field'][x,y] == -1:
                walls.append((x,y))

    neighbourhood = []

    left = (pos[0]-1,pos[1])
    right = (pos[0]+1,pos[1])
    top = (pos[0],pos[1]-1)
    down = (pos[0],pos[1]+1)


    # Avoid bomb state 
    bomb_state = ['0','0','0','0']
    if bomb_radar(self,top_radar,bombs,train):
        bomb_state[0] = '1'
    elif bomb_radar(self,right_radar,bombs,train):
        bomb_state[1] = '1'
    elif bomb_radar(self,down_radar,bombs,train):
        bomb_state[2] = '1'
    elif bomb_radar(self,left_radar,bombs,train):
        bomb_state[3] = '1'
    
    # Coin state 
    coin_state = ['0','0','0','0']
    if coin_radar(self,top_radar,old_game_state['coins'],train):
        coin_state[0] = '1'
    elif coin_radar(self,right_radar,old_game_state['coins'],train):
        coin_state[1] = '1'
    elif coin_radar(self,down_radar,old_game_state['coins'],train):
        coin_state[2] = '1'
    elif coin_radar(self,left_radar,old_game_state['coins'],train):
        coin_state[3] = '1'

    # Crate state 
    crate_state = ['0','0','0','0']
    if crate_radar(self,top_radar,crates,train):
        crate_state[0] = '1'
    elif crate_radar(self,right_radar,crates,train):
        crate_state[1] = '1'
    elif crate_radar(self,down_radar,crates,train):
        crate_state[2] = '1'
    elif crate_radar(self,left_radar,crates,train):
        crate_state[3] = '1'

    # Wall state 
    wall_state = ['0','0','0','0']
    if wall_radar(self,top_radar,crates,train):
        wall_state[0] = '1'
    elif wall_radar(self,right_radar,crates,train):
        wall_state[1] = '1'
    elif wall_radar(self,down_radar,crates,train):
        wall_state[2] = '1'
    elif wall_radar(self,left_radar,crates,train):
        wall_state[3] = '1'

    # Agent state 
    enemy_state = ['0','0','0','0']
    if enemy_radar(self,top_radar,crates,train):
        enemy_state[0] = '1'
    elif enemy_radar(self,right_radar,crates,train):
        enemy_state[1] = '1'
    elif enemy_radar(self,down_radar,crates,train):
        enemy_state[2] = '1'
    elif enemy_radar(self,left_radar,crates,train):
        enemy_state[3] = '1'

    # Explore state 
    explore_state = ['0','0','0','0']
    if top in self.visited_positions:
        explore_state[0] = '1'
    elif right in self.visited_positions:
        explore_state[1] = '1'
    elif down in self.visited_positions:
        explore_state[2] = '1'
    elif left in self.visited_positions:
        explore_state[3] = '1'

    # Throw bomb state 
    bombing_state = ['0','0','0','0']
    if top in crates or top in enemies:
        bombing_state[0] = '1'
    elif right in crates or right in enemies:
        bombing_state[1] = '1'
    elif down in crates or down in enemies:
        bombing_state[2] = '1'
    elif left in crates or left in enemies:
        bombing_state[3] = '1'

    # Explosion map 
    explosion_state = ['0','0','0','0']
    if top in crates or top in enemies:
        explosion_state[0] = '1'
    elif right in crates or right in enemies:
        explosion_state[1] = '1'
    elif down in crates or down in enemies:
        explosion_state[2] = '1'
    elif left in crates or left in enemies:
        explosion_state[3] = '1'

    binary_feature = bomb_state+coin_state+crate_state+wall_state+enemy_state+explore_state+bomb_state+explosion_state
     
    binary_feature_state = ''.join(binary_feature)
    
    hex_feature = hex(int(binary_feature_state, 2))

    if hex_feature not in self.feature_list:
        self.new_value_function.append([0.2,0.2,0.2,0.2,0.1,0.1,hex_feature])
        self.new_policy.append([0.2,0.2,0.2,0.2,0.1,0.1,hex_feature])
        self.feature_list.append(hex_feature)

    return hex_feature 

