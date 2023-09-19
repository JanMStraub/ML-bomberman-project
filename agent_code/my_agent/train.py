from collections import namedtuple, deque

import pickle
from typing import List

import events as e
from .callbacks import state_to_features, extract_state

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

    """# Idea: Add your own events to hand out rewards
    if ...:
        events.append(PLACEHOLDER_EVENT)"""

    #sarsa(self,old_game_state,self_action,new_game_state,get_reward(self,old_game_state['self'][3],\
    #                    self.visited_positions,old_game_state['self'][3],new_game_state['self'][3],self.target_coin,self.bomb,events))

    self.state_history.append(extract_state(self,old_game_state))
    self.action_history.append(self_action)
    self.event_history.append(events)
    self.visited_positions.append(old_game_state['self'][3])
    self.new_pos_history.append(new_game_state['self'][3])
    self.target_coins_history.append(self.target_coin)
    if 'BOMB_EXPLODED' in events:
        self.bomb = None
        self.bomb_detect_pos = None
    self.bomb_history.append(self.bomb)
    self.bomb_detect_pos_history.append(self.bomb_detect_pos)
    
    #self.value_estimates = state_to_features(self.value_estimates,old_game_state, new_game_state)
    #trans = Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward_from_events(self, events))

    # state_to_features is defined in callbacks.py
    #self.transitions.append(Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward_from_events(self, events)))

def get_reward(self,game_state,crates,pos,visited_positions,bomb_detect_pos,new_pos,target_pos,bomb,events):
    game_rewards = {
        e.COIN_COLLECTED: 1,
        e.KILLED_OPPONENT: 1,
        e.BOMB_DROPPED: 1,
        e.BOMB_EXPLODED: 0,
        e.MOVED_LEFT: 1,
        e.MOVED_RIGHT: 1,
        e.MOVED_UP: 1,
        e.MOVED_DOWN: 1,
        e.WAITED: -1,
        e.INVALID_ACTION: -1,
        e.TILE_VISITED: -1,
        e.SURVIVED_ROUND: 2,
        e.MOVED_TOWARDS_COIN: 2,
        e.KILLED_SELF: -2,
        e.CRATE_DESTROYED: 2,
        e.COIN_FOUND: 2,
        PLACEHOLDER_EVENT: -.1  # idea: the custom event is bad
    }



    """if pos in visited_positions:
        reward_ctr = -1
    else:
        reward_ctr = 0 """
    
    if new_pos in visited_positions:
        reward_ctr = -1 
    else:
        reward_ctr = 0

    state = game_state

    # DOWN
    if state in [[0,0],[2,0],[5,0],[6,0],[7,0],[8,0],[10,3]]:
        if new_pos[1]-pos[1] == 1:
            reward_ctr += 0.5 
        else:
            reward_ctr -= 0.5 
    # UP 
    elif state in [[1,0],[3,0],[5,1],[6,1],[9,0],[7,2],[10,2]]:
        if pos[1]-new_pos[1] == 1:
            reward_ctr += 0.5 
        else:
            reward_ctr -= 0.5 
    # RIGHT
    elif state in [[2,1],[3,1],[4,1],[6,2],[8,2],[9,2],[10,0]]:
        if new_pos[0]-pos[0] == 1:
            reward_ctr += 0.5 
        else:
            reward_ctr -= 0.5 
    # LEFT
    elif state in [[0,1],[1,1],[4,0],[7,1],[8,1],[9,1],[10,1]]:
        if pos[0]-new_pos[0] == 1:
            reward_ctr += 0.5 
        else:
            reward_ctr -= 0.5 
 
    if euclidean_distance(pos,target_pos) < euclidean_distance(new_pos,target_pos):
        reward_ctr -= 1
    else:
        reward_ctr += 1

    if bomb:      
        if bomb in crates:
            reward_ctr+=2

        new_pos_euclidean = euclidean_distance(new_pos,bomb)

        if euclidean_distance(bomb_detect_pos,bomb) >= new_pos_euclidean:
            reward_ctr -= 2
        elif avoid_check_(bomb_detect_pos, new_pos, bomb):
            reward_ctr += 5 
        elif new_pos_euclidean > 1 and new_pos_euclidean <= 2:
            reward_ctr += 3
        elif new_pos_euclidean > 2 and new_pos_euclidean <= 3:
            reward_ctr += 4
        else:
            reward_ctr += 5

    for event in events:
        reward_ctr += game_rewards[event]
    return reward_ctr

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
    # MC-Control for update value-function and policy 
    mc_control(self,last_game_state,crates)

    self.model = self.policy

    # Store the model

    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 1,
        e.KILLED_OPPONENT: 5,
        PLACEHOLDER_EVENT: -.1  # idea: the custom event is bad
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum


def mc_control(self,game_state,crates):
    """
    Monte-Carlo Control 
    """
    i = 0 
    for event in self.event_history: 
            self.reward_history.append(get_reward(self,self.state_history[i],crates,self.visited_positions[i],self.visited_positions[:i],self.bomb_detect_pos_history[i]\
                                                  ,self.new_pos_history[i],self.target_coins_history[i],self.bomb_history[i],event))
            if self.bomb_history[i] in crates:
                crates.remove(self.bomb_history[i])
            i+=1

    if game_state['round'] < 5000:
        epsilon = 0.5
    elif game_state['round'] >= 5000 and game_state['round'] < 7500:
        epsilon = 0.4
    elif game_state['round'] >= 7500 and game_state['round'] < 12500:
        epsilon = 0.3
    else:
        epsilon = 0.2
    g = 0 
    t = 0
    disc= 0.95 
    for state in self.state_history:
            
        #while t < len(self.state_history)-1: ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
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

            #if state == [1,0]:
            #    print(self.action_history[t])

            if True:#(state,action) not in self.first_visit_check:
                self.first_visit_check.append((state,action))
                #g = pow(gamma,t) * g + self.reward_history[t]
                #g += self.reward_history[t]
                g = get_state_return(disc,t,self.reward_history[t:])


                self.return_val[state[0],state[1],action]+=g
                self.return_ctr[state[0],state[1],action]+=1
                self.value_estimates[state[0],state[1],action] = self.return_val[state[0],state[1],action]/self.return_ctr[state[0],state[1],action]
                a_star = np.argmax(self.value_estimates[state[0],state[1],:])

                # greedy 
                num_actions = 6
                #num_actions = 4
                for i in range(num_actions):
                    if i == a_star:
                        self.policy[state[0],state[1],i] = 1-epsilon+epsilon/num_actions 
                    else:
                        self.policy[state[0],state[1],i] = epsilon/num_actions 
            t+=1
    
    self.state_history = []
    self.action_history = []
    self.event_history = []
    self.reward_history = []
    self.visited_positions = []
    self.new_pos_history = []
    self.target_coin_history = []
    self.bomb_history = []  
    self.bomb_detect_pos_history = []
    self.first_visit_check = []


    return 0

def sarsa(self,old_game_state,self_action,new_game_state,reward):
    
    epsilon = 0.1
    alpha = 0.2
    gamma = 0.95 
    
    old_state = extract_state(self,old_game_state)
    new_state = extract_state(self,new_game_state)

    
    if self_action == 'UP':
        action = 0 
    elif self_action == 'RIGHT':
        action = 1
    elif self_action == 'DOWN':
        action = 2
    elif self_action == 'LEFT':
        action = 3
    elif self_action == 'WAIT':
        action = 4
    else:
        action = 5

    estimated_action = np.random.choice([0,1,2,3,4,5], p = self.policy[new_state[0],new_state[0],:])
    
    self.value_estimates[old_state[0],old_state[1],action] += alpha*(reward+gamma*self.value_estimates[new_state[0],new_state[1],estimated_action]-self.value_estimates[old_state[0],old_state[1],action])

    a_star = np.argmax(self.value_estimates[old_state[0],old_state[1],:])

    # greedy 
    for i in range(6):
        if i == a_star:
            self.policy[old_state[0],old_state[1],i] = 1-epsilon+epsilon/6
        else:
            self.policy[old_state[0],old_state[1],i] = epsilon/6

    return 0 




def euclidean_distance(coord1, coord2):
    x1, y1 = coord1
    x2, y2 = coord2
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance

def avoid_check_(oldPos, newPos, bomb):
    av_check = False 
    if oldPos[0] == bomb[0]:
        if newPos[0] == bomb[0]:
            av_check = False 
        else:
            av_check = True       
    elif oldPos[1] == bomb[1]:
        if newPos[1] == bomb[1]:
            av_check = False 
        else:
            av_check = True 
    else:
        av_check = True
    
    return av_check 

def get_state_return(disc,t,rewards):

    state_return = rewards[0]
    for reward in rewards[1:]:
        state_return += pow(disc,t) * reward
        t+=1

    return state_return