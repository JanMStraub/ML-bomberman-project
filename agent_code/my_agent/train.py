from collections import namedtuple, deque

import pickle
from typing import List

import events as e
from .callbacks import state_to_features,crate_radar,enemy_radar,coin_radar,bomb_radar

import numpy as np

from statistics import mean

import math

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    #self.events.append(events)
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    crates = []
    for x in range(17):
        for y in range(17):
            if old_game_state['field'][x,y] == 1:
                crates.append((x,y))

    # Enemy agents 
    enemies = old_game_state['others']
    enemy_pos = []
    for enemy in enemies:
        enemy_pos.append(enemy[3])

    feature_state = state_to_features(self,self.new_value_function, old_game_state, new_game_state,True)
    self.state_history.append(feature_state)
    self.action_history.append(self_action)
    self.event_history.append(events)

    self.target_coins_history.append(old_game_state['coins'])
    self.target_crates_history.append(crates)
    self.bomb_history.append(([x for (x,y) in old_game_state['bombs']]))
    self.bomb_timer_history.append(([y for (x,y) in old_game_state['bombs']]))
    self.target_enemy_history.append(enemy_pos)
    self.visited_positions.append(old_game_state['self'][3])
    self.new_pos_history.append(new_game_state['self'][3])

    crates = []
    self.n_sarsa_ctr+=1
    if self.n_sarsa_ctr == 3:
        n_sarsa(self,old_game_state,crates)
        self.target_coins_history = []
        self.target_crates_history = []
        self.bomb_history = []
        self.target_enemy_history = []
        self.n_sarsa_ctr = 0
    
    
def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    #self.transitions.append(Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events)))


    crates = []
    for x in range(17):
        for y in range(17):
            if last_game_state['field'][x,y] == 1:
                crates.append((x,y))

    # Enemy agents 
    enemies = last_game_state['others']
    enemy_pos = []
    for enemy in enemies:
        enemy_pos.append(enemy[3])

    feature_state = state_to_features(self,self.new_value_function, last_game_state, last_game_state,True)
    self.state_history.append(feature_state)
    self.action_history.append(last_action)
    self.event_history.append(events)

    self.target_coins_history.append(last_game_state['coins'])
    self.target_crates_history.append(crates)
    self.bomb_history.append(([x for (x,y) in last_game_state['bombs']]))
    self.bomb_timer_history.append(([y for (x,y) in last_game_state['bombs']]))
    self.target_enemy_history.append(enemy_pos)
    self.visited_positions.append(last_game_state['self'][3])
    self.new_pos_history.append(last_game_state['self'][3])

    
    n_sarsa(self,last_game_state,crates)
    self.target_coins_history = []
    self.target_crates_history = []
    self.bomb_history = []
    self.target_enemy_history = []
    self.bomb_timer_history = []
    self.n_sarsa_ctr = 0

    self.model = self.new_policy

    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)


def reset_episode_arrays(self):
    """
    Reset the arrays of the last episode.
    """
    self.state_history = []
    self.action_history = []
    self.event_history = []
    self.reward_history = []
    self.visited_positions = []
    self.new_pos_history = []
    self.target_coin_history = []
    self.bomb_history = []  
    self.bomb_timer_history = []
    self.first_visit_check = []
    self.reward_debug_history = []



def n_sarsa(self,game_state,crates):
    i = 0 
    for event in self.event_history: 
        self.reward_history.append(n_sarsa_reward(self,crates,event,i,game_state))
        i+=1

    if game_state['round'] < 100000:
        epsilon = 0.4
    elif game_state['round'] >= 100000 and game_state['round'] < 300000:
        epsilon = 0.3
    elif game_state['round'] >= 300000 and game_state['round'] < 600000:
        epsilon = 0.2
    else:
        epsilon = 0.1
    g = 0 
    t = 0
    disc= 0.95 
    alpha = 0.1
    for state in self.state_history:
        if self.action_history[t] == 'UP':
            action = 0
        elif self.action_history[t] == 'RIGHT':
            action = 1 
        elif self.action_history[t] == 'DOWN':
            action = 2
        elif self.action_history[t] == 'LEFT':
            action = 3
        elif self.action_history[t] == 'WAIT':
            action = 4
        else:
            action = 5

        g = get_state_return(disc,t,self.reward_history[t:])

        index = self.feature_list.index(state)
        self.new_value_function[index][action] += alpha*(g-self.new_value_function[index][action]) 
        a_star = np.argmax(self.new_value_function[index][:6])

        # greedy 
        num_actions = 6
        for i in range(num_actions):
            if i == a_star:
                self.new_policy[index][i] = 1-epsilon+epsilon/num_actions 
            else:
                self.new_policy[index][i] = epsilon/num_actions 
        t+=1
    
    reset_episode_arrays(self)
    return 0 


def get_closest_object(objects,pos):
    close_object = (0,0)
    old_distance = 100
    if objects:
        for object_pos in objects:
            distance = euclidean_distance(object_pos,pos) 
            if distance < old_distance:
                close_object = object_pos
                old_distance = distance
    return close_object

def coin_reward(reward_ctr,new_pos,pos,close_object):
    if close_object != (0,0):
        if euclidean_distance(pos,close_object) > euclidean_distance(new_pos,close_object):
            reward_ctr += 2
        else:
            reward_ctr -= 2
    return reward_ctr

def euclidean_distance(coord1, coord2):
    x1, y1 = coord1
    x2, y2 = coord2
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance

def avoid_check_(newPos, bomb):
    if euclidean_distance(newPos,bomb)%1 != 0:
        av_check = True
    else:
        av_check = False 
    return av_check

def get_state_return(disc,t,rewards):

    state_return = rewards[0]
    for reward in rewards[1:]:
        state_return += pow(disc,t) * reward
        t+=1

    return state_return

def move_reward(state,new_pos,pos,visited_positions,reward_ctr):
    """
    Reward movement related events. 
    """
    if new_pos in visited_positions:
        reward_ctr -= 4 
    else:
        reward_ctr += 2
    return reward_ctr

def crate_reward(reward_ctr,new_pos,pos,close_object):
    """
    Reward crate related events. 
    """
    if close_object != (0,0):
        if euclidean_distance(pos,close_object) > euclidean_distance(new_pos,close_object):
            reward_ctr += 2
        else:
            reward_ctr -= 2
    return reward_ctr

def bomb_reward(reward_ctr,new_pos,pos,close_object,crates,enemies,events):
    """
    Reward bomb related events. 
    """
    left = (pos[0]-1,pos[1])
    right = (pos[0]+1,pos[1])
    top = (pos[0],pos[1]-1)
    down = (pos[0],pos[1]+1)
    
    if 'BOMB_DROPPED' in events:
        if top in crates or top in enemies:
            reward_ctr += 2
        elif right in crates or right in enemies:
            reward_ctr += 2
        elif down in crates or down in enemies:
            reward_ctr += 2
        elif left in crates or left in enemies:
            reward_ctr += 2
        else:
            reward_ctr -= 2

    if 'WAITED' in events or 'INVALID_ACTION' in events:
        if top in crates or top in enemies:
            reward_ctr -= 3
        elif right in crates or right in enemies:
            reward_ctr -= 3
        elif down in crates or down in enemies:
            reward_ctr -= 3
        elif left in crates or left in enemies:
            reward_ctr -= 3

    new_pos_euclidean = euclidean_distance(new_pos,close_object)

    if euclidean_distance(pos,close_object) >= new_pos_euclidean and euclidean_distance(pos,close_object) < 5:
        reward_ctr -= 2
    elif avoid_check_(pos,close_object):
        reward_ctr += 2 
    elif avoid_check_(new_pos,close_object):
        reward_ctr += 2 
    elif new_pos_euclidean >= 1 and new_pos_euclidean < 2 and new_pos != pos:
        reward_ctr += 2
    elif new_pos_euclidean >= 2 and new_pos_euclidean < 3 and new_pos != pos:
        reward_ctr += 3
    elif new_pos_euclidean >= 3 and new_pos_euclidean < 7 and new_pos != pos:
        reward_ctr += 4
    return reward_ctr

def enemy_reward(reward_ctr,new_pos,pos,close_object):
    """
    Reward enemy related events. 
    """
    if close_object != (0,0):
        if euclidean_distance(pos,close_object) > euclidean_distance(new_pos,close_object):
            reward_ctr += 1
        else:
            reward_ctr -= 1
    return reward_ctr

def n_sarsa_reward(self,crates,events,i,game_state):

    crates = self.target_crates_history[i][:]
    coins = self.target_coins_history[i][:]
    bombs = self.bomb_history[i][:]
    enemies = self.target_enemy_history[i][:]

    state = self.state_history[i]
    pos = self.visited_positions[i]
    visited_positions = self.visited_positions[:i]
    new_pos = self.new_pos_history[i]

    timer = self.bomb_timer_history[i]

    left_radar = [(x,pos[1]) for x in range(pos[0],pos[0]-4,-1)]
    right_radar = [(x,pos[1]) for x in range(pos[0],pos[0]+4,1)]
    top_radar = [(pos[0],y) for y in range(pos[1],pos[1]-4,-1)]
    down_radar = [(pos[0],y) for y in range(pos[1],pos[1]+4,1)]
    
    game_rewards = {
        e.COIN_COLLECTED: 1,
        e.KILLED_OPPONENT: 3,
        e.BOMB_DROPPED: 0,
        e.BOMB_EXPLODED: 0,
        e.MOVED_LEFT: 1,
        e.MOVED_RIGHT: 1,
        e.MOVED_UP: 1,
        e.MOVED_DOWN: 1,
        e.WAITED: -2,
        e.INVALID_ACTION: -2,
        e.TILE_VISITED: -1,
        e.SURVIVED_ROUND: 2,
        e.MOVED_TOWARDS_COIN: 2,
        e.KILLED_SELF: -5,
        e.CRATE_DESTROYED: 3,
        e.COIN_FOUND: 3,
        e.GOT_KILLED:-2,
        e.OPPONENT_ELIMINATED: 3,
        PLACEHOLDER_EVENT: -.1  # idea: the custom event is bad
    }


    reward_ctr = 0

    
    reward_ctr = move_reward(state,new_pos,pos,visited_positions,reward_ctr)

    if coin_radar(self,top_radar,coins,False) or coin_radar(self,right_radar,coins,False)\
        or coin_radar(self,down_radar,coins,False) or coin_radar(self,left_radar,coins,False):
        close_object = get_closest_object(coins,pos)
        reward_ctr = coin_reward(reward_ctr,new_pos,pos,close_object)


    if crate_radar(self,top_radar,crates,False) or crate_radar(self,right_radar,crates,False)\
        or crate_radar(self,down_radar,crates,False) or crate_radar(self,left_radar,crates,False):
        close_object = get_closest_object(crates,pos)
        reward_ctr = crate_reward(reward_ctr,new_pos,pos,close_object)
    
    if enemy_radar(self,top_radar,crates,False) or enemy_radar(self,right_radar,crates,False)\
        or enemy_radar(self,down_radar,crates,False) or enemy_radar(self,left_radar,crates,False):
        close_object = get_closest_object(enemies,pos)
        reward_ctr = enemy_reward(reward_ctr,new_pos,pos,close_object)
    
    max_timer = 0
    if timer:
        max_timer = max(timer)
    if 'BOMB_DROPPED' in events or max_timer != 0:
        close_object = get_closest_object(bombs,pos)
        reward_ctr = bomb_reward(reward_ctr,new_pos,pos,close_object,crates,enemies,events)
        
    for event in events:
        reward_ctr += game_rewards[event]

    return reward_ctr