from collections import namedtuple, deque

import pickle
from typing import List

import events as e
from .callbacks import state_to_features,extract_state,crate_radar,enemy_radar,coin_radar,bomb_radar

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

    """self.visited_positions.append(old_game_state['self'][3])
    self.new_pos_history.append(new_game_state['self'][3])
    if 'COIN_COLLECTED' in events:
        self.target_coin = (0,0)
    if 'CRATE_DESTROYED' in events:
        self.target_crate = (0,0)
    if 'OPPONENT_ELIMINATED' in events:
        self.target_enemy = (0,0)"""


    crates = []
    self.n_sarsa_ctr+=1
    if self.n_sarsa_ctr == 3:
        n_sarsa(self,old_game_state,crates)
        self.target_coins_history = []
        self.target_crates_history = []
        self.bomb_history = []
        self.target_enemy_history = []
        self.n_sarsa_ctr = 0
    
    """# Idea: Add your own events to hand out rewards
    if ...:
        events.append(PLACEHOLDER_EVENT)"""
   
    """
    for x in range(17):
        for y in range(17):
            if old_game_state['field'][x,y] == 1:
                crates.append((x,y))"""

    #self.state_history.append(extract_state(self,old_game_state,True))
    """self.state_history.append(states(self,old_game_state,True))
    self.action_history.append(self_action)
    self.event_history.append(events)
    self.visited_positions.append(old_game_state['self'][3])
    self.new_pos_history.append(new_game_state['self'][3])
    if 'COIN_COLLECTED' in events:
        self.target_coin = (0,0)
    if 'CRATE_DESTROYED' in events:
        self.target_crate = (0,0)
    if 'OPPONENT_ELIMINATED' in events:
        self.target_enemy = (0,0)
        
        
    crates = []
    self.n_sarsa_ctr+=1
    if self.n_sarsa_ctr == 3:
        mc_constant_alpha(self,old_game_state,crates)
        #mc_control(self,old_game_state,crates)
        self.target_coins_history = []
        self.target_crates_history = []
        self.bomb_history = []
        self.target_enemy_history = []
        self.n_sarsa_ctr = 0"""

    

    #sarsa(self,old_game_state,self_action,new_game_state,events,crates)
    
    #self.value_estimates = state_to_features(self.value_estimates,old_game_state, new_game_state)
    #trans = Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward_from_events(self, events))

    # state_to_features is defined in callbacks.py
    #self.transitions.append(Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward_from_events(self, events)))


def get_reward(self,crates,events,i,game_state):

    state = self.state_history[i]
    pos = self.visited_positions[i]
    visited_positions = self.visited_positions[:i]
    timer = self.bomb_timer_history[i]
    if len(self.visited_positions) > 3:
        bomb_detected_pos = self.visited_positions[i-(3-timer)]
    else:
        bomb_detected_pos = self.visited_positions[i]
    new_pos = self.new_pos_history[i]
    target_coin = self.target_coins_history[i]
    target_crate = self.target_crates_history[i]
    target_enemy = self.target_enemy_history[i]
    bomb_pos = self.bomb_history[i]

    
    game_rewards = {
        e.COIN_COLLECTED: 1,
        e.KILLED_OPPONENT: 1,
        e.BOMB_DROPPED: 0,
        e.BOMB_EXPLODED: 0,
        e.MOVED_LEFT: 1,
        e.MOVED_RIGHT: 1,
        e.MOVED_UP: 1,
        e.MOVED_DOWN: 1,
        e.WAITED: -6,
        e.INVALID_ACTION: -6,
        e.TILE_VISITED: -1,
        e.SURVIVED_ROUND: 2,
        e.MOVED_TOWARDS_COIN: 2,
        e.KILLED_SELF: -2,
        e.CRATE_DESTROYED: 2,
        e.COIN_FOUND: 2,
        e.GOT_KILLED:-2,
        e.OPPONENT_ELIMINATED: 3,
        PLACEHOLDER_EVENT: -.1  # idea: the custom event is bad
    }


    reward_ctr = 0

    #if state[0] == 3:
    reward_ctr = move_handling(state,new_pos,pos,visited_positions,reward_ctr)
    #if state[0] == 1:
    close_object = get_closest_object(game_state['coins'],pos)
    reward_ctr = coin_reward(reward_ctr,new_pos,pos,close_object)
    #reward_ctr = coin_handling(reward_ctr,new_pos,pos,target_coin)
    #if state[0] == 2:
    close_object = get_closest_object(crates,pos)
    reward_ctr = crate_handling(reward_ctr,target_crate,pos,new_pos)
    #if state[0] == 4:
    reward_ctr = enemy_handling(reward_ctr,target_enemy,pos,new_pos)
    
    if 'BOMB_DROPPED' in events or timer != 0:
        reward_ctr = bomb_handling(reward_ctr,bomb_pos,crates,bomb_detected_pos,new_pos,pos,events,state,game_state)

    for event in events:
        reward_ctr += game_rewards[event]
    self.reward_debug_history.append(reward_ctr)        
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

   

    """self.visited_positions.append(old_game_state['self'][3])
    self.new_pos_history.append(new_game_state['self'][3])
    if 'COIN_COLLECTED' in events:
        self.target_coin = (0,0)
    if 'CRATE_DESTROYED' in events:
        self.target_crate = (0,0)
    if 'OPPONENT_ELIMINATED' in events:
        self.target_enemy = (0,0)"""


    crates = []
    self.n_sarsa_ctr+=1
    if self.n_sarsa_ctr == 3:
        n_sarsa(self,last_game_state,crates)
        self.target_coins_history = []
        self.target_crates_history = []
        self.bomb_history = []
        self.target_enemy_history = []
        self.bomb_timer_history = []
        self.n_sarsa_ctr = 0



    """#self.state_history.append(extract_state(self,last_game_state,True))
    self.state_history.append(states(self,last_game_state,True))
    self.action_history.append(last_action)
    self.event_history.append(events)
    self.visited_positions.append(last_game_state['self'][3])
    self.new_pos_history.append(last_game_state['self'][3])

    crates = []

    for x in range(17):
        for y in range(17):
            if last_game_state['field'][x,y] == 1:
                crates.append((x,y))

    # MC-Control for update value-function and policy 
    #mc_control(self,last_game_state,crates)
    mc_constant_alpha(self,last_game_state,crates)"""

    # For Sarsa
    #self.visited_positions

    self.model = self.new_policy

    # Store the model

    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)

    """def move_handling(state,new_pos,pos,visited_positions):
    """
    #Reward movement related events. 
    """
    if new_pos in visited_positions:
        reward_ctr = -1 
    else:
        reward_ctr = 0

     # DOWN
    if state in [[0,0],[2,0],[5,0],[6,0],[7,0],[8,0],[10,3]]:
        if new_pos[1]-pos[1] == 1:
            reward_ctr += 1 
        else:
            reward_ctr -= 0
    # UP 
    elif state in [[1,0],[3,0],[5,1],[6,1],[9,0],[7,2],[10,2]]:
        if pos[1]-new_pos[1] == 1:
            reward_ctr += 1 
        else:
            reward_ctr -= 0
    # RIGHT
    elif state in [[2,1],[3,1],[4,1],[6,2],[8,2],[9,2],[10,0]]:
        if new_pos[0]-pos[0] == 1:
            reward_ctr += 1 
        else:
            reward_ctr -= 0 
    # LEFT
    elif state in [[0,1],[1,1],[4,0],[7,1],[8,1],[9,1],[10,1]]:
        if pos[0]-new_pos[0] == 1:
            reward_ctr += 1
        else:
            reward_ctr -= 0 

    return reward_ctr"""

def move_handling(state,new_pos,pos,visited_positions,reward_ctr):
    """
    Reward movement related events. 
    """
    if new_pos in visited_positions:
        reward_ctr -= 1 
    else:
        reward_ctr += 1

    """# DOWN
    if state in [[0,2],[1,2],[2,3],[3,2]]:
        if new_pos[1]-pos[1] == 1:
            reward_ctr += 1 
        else:
            reward_ctr -= 0
    # UP 
    elif state in [[0,0],[1,0],[2,1],[3,0]]:
        if pos[1]-new_pos[1] == 1:
            reward_ctr += 1 
        else:
            reward_ctr -= 0
    # RIGHT
    elif state in [[0,1],[1,1],[2,2],[3,1]]:
        if new_pos[0]-pos[0] == 1:
            reward_ctr += 1 
        else:
            reward_ctr -= 0 
    # LEFT
    elif state in [[0,3],[1,3],[2,4],[3,4]]:
        if pos[0]-new_pos[0] == 1:
            reward_ctr += 1
        else:
            reward_ctr -= 0 """

    return reward_ctr

def coin_handling(reward_ctr,new_pos,pos,target_coin):
    if target_coin != (0,0):
        if euclidean_distance(pos,target_coin) <= euclidean_distance(new_pos,target_coin):
            reward_ctr -= 2
        else:
            reward_ctr += 2
    return reward_ctr

def bomb_handling(reward_ctr,bomb_pos,crates,bomb_detected_pos,new_pos,pos,events,state,game_state):
    """
    Reward bomb related events. 
    """
    if 'BOMB_DROPPED' in events: 
        #reward_ctr = crate_handling(pos,reward_ctr,crates,game_state)
        """if state == [2,0]:
            reward_ctr+=3
        else:
            reward_ctr-=2"""
        bomb_pos = pos 

        """if state in [[0,0],[0,1],[0,2],[0,3],[0,4],[0,5],[0,6],\
                     [1,0],[1,1],[1,2],[1,3],[1,4],[1,5],[1,6],\
                     [2,0],[2,1],[2,2],[2,3],[2,4],[2,5],[2,6],\
                     [3,0],[3,1],[3,2],[3,3],[3,4],[3,5],[3,6],\
                     [4,0],[4,1],[4,2],[4,3],[4,4],[4,5],[4,6],\
                     [5,0],[5,1],[5,2],[5,3],[5,4],[5,5],[5,6],\
                     [6,0],[6,1],[6,2],[6,3],[6,4],[6,5],[6,6],[6,7],[6,8],[6,9],\
                     [7,0],[7,1],[7,2],[7,3],[7,4],[7,5],[7,6],[7,7],[7,8],[7,9],\
                     [8,0],[8,1],[8,2],[8,3],[8,4],[8,5],[8,6],[8,7],[8,8],[8,9],\
                     [9,0],[9,1],[9,2],[9,3],[9,4],[9,5],[9,6],[9,7],[9,8],[9,9],\
                     [10,0],[10,1],[10,2],[10,3],[10,4],[10,5],[10,6],[10,7],
                     [10,8],[10,9],[10,10],[10,11],[10,12]]:
            reward_ctr -= 2

        # DOWN
        """
            

    """print("crates pos:",reward_ctr)

    print("New Pos:", new_pos," Bomb Pos:",bomb_pos," Bomb Detected",bomb_deteced_pos)
    print("Euclidean: ",euclidean_distance(new_pos,bomb_pos)," ",euclidean_distance(bomb_deteced_pos,bomb_pos))"""

    #print("Avoid Ckeck:",avoid_check_(bomb_deteced_pos,new_pos,bomb_pos))
    new_pos_euclidean = euclidean_distance(new_pos,bomb_pos)

    if euclidean_distance(bomb_detected_pos,bomb_pos) >= new_pos_euclidean:
        reward_ctr -= 2
    elif avoid_check_(pos,bomb_pos) and 'WAITED' in events and state[0,5]:
        reward_ctr += 5 
    elif avoid_check_(new_pos,bomb_pos):
        reward_ctr += 5 
    elif new_pos_euclidean >= 1 and new_pos_euclidean < 2 and new_pos != pos:
        reward_ctr += 3
    elif new_pos_euclidean >= 2 and new_pos_euclidean < 3 and new_pos != pos:
        reward_ctr += 4
    elif new_pos_euclidean >= 3 and new_pos != pos:
        reward_ctr += 5
    return reward_ctr

def crate_handling(reward_ctr,target_crate,pos,new_pos):
    """
    Reward crate related events. 
    """
    if target_crate != (0,0):
        if euclidean_distance(pos,target_crate) <= euclidean_distance(new_pos,target_crate):
            reward_ctr -= 2
        else:
            reward_ctr += 2
    return reward_ctr
    """top_pos = (bomb_pos[0],bomb_pos[1]-1)
    low_pos = (bomb_pos[0],bomb_pos[1]+1)
    left_pos = (bomb_pos[0]-1,bomb_pos[1])
    right_pos = (bomb_pos[0]+1,bomb_pos[1])

    if game_state['field'][top_pos] == 1:
        reward_ctr+=1
    if game_state['field'][low_pos] == 1:
        reward_ctr+=1
    if game_state['field'][left_pos] == 1:
        reward_ctr+=1
    if game_state['field'][right_pos] == 1:
        reward_ctr+=1

    return reward_ctr"""

def enemy_handling(reward_ctr,target_enemy,pos,new_pos):
    """
    Reward enemy related events. 
    """
    if target_enemy != (0,0):
        if euclidean_distance(pos,target_enemy) <= euclidean_distance(new_pos,target_enemy):
            reward_ctr -= 1
        else:
            reward_ctr += 1
    return reward_ctr
    reward = 0 
    return reward

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

def mc_control(self,game_state,crates):
    """
    Monte-Carlo Control 
    """
    i = 0 
    for event in self.event_history: 
            self.reward_history.append(get_reward(self,crates,event,i))

            if self.bomb_history[i] is not None:
                top_pos = (self.bomb_history[i][0],self.bomb_history[i][1]-1)
                low_pos = (self.bomb_history[i][0],self.bomb_history[i][1]+1)
                left_pos = (self.bomb_history[i][0]-1,self.bomb_history[i][1])
                right_pos = (self.bomb_history[i][0]+1,self.bomb_history[i][1])
                        
                if top_pos in crates:
                    crates.remove(top_pos)
                if low_pos in crates:
                    crates.remove(low_pos)
                if left_pos in crates:
                    crates.remove(left_pos)
                if top_pos in crates:
                    crates.remove(right_pos)
            i+=1

    if game_state['round'] < 10000:
        epsilon = 0.8
    elif game_state['round'] >= 10000 and game_state['round'] < 25000:
        epsilon = 0.5
    elif game_state['round'] >= 25000 and game_state['round'] < 38000:
        epsilon = 0.2
    else:
        epsilon = 0.1
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

            #if state == [0,0]:
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
    
    reset_episode_arrays(self)

    return 0


def mc_constant_alpha(self,game_state,crates):
    """
    Monte-Carlo Control 
    """
    i = 0 
    for event in self.event_history: 
        self.reward_history.append(get_reward(self,crates,event,i,game_state))
        i+=1

    if game_state['round'] < 20000:
        epsilon = 0.4
    elif game_state['round'] >= 20000 and game_state['round'] < 40000:
        epsilon = 0.3
    elif game_state['round'] >= 45000 and game_state['round'] < 60000:
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

        #if state == [1,0]:
        #    print(self.action_history[t])

        if True:#(state,action) not in self.first_visit_check:
            self.first_visit_check.append((state,action))
            g = get_state_return(disc,t,self.reward_history[t:])

            self.value_estimates[state[0],state[1],action] += alpha*(g-self.value_estimates[state[0],state[1],action]) 
            a_star = np.argmax(self.value_estimates[state[0],state[1],:])

            # greedy 
            num_actions = 6
            for i in range(num_actions):
                if i == a_star:
                    self.policy[state[0],state[1],i] = 1-epsilon+epsilon/num_actions 
                else:
                    self.policy[state[0],state[1],i] = epsilon/num_actions 
        t+=1
    
    reset_episode_arrays(self)

    return 0


def sarsa(self,old_game_state,self_action,new_game_state,events,crates):
    
    epsilon = 0.2
    alpha = 0.2
    gamma = 0.95 
    
    old_state = extract_state(self,old_game_state,True)
    new_state = extract_state(self,new_game_state,True)

    
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

    reward = get_sarsa_reward(self,old_state,old_game_state,new_game_state,events,crates)

    estimated_action = np.random.choice([0,1,2,3,4,5], p = self.policy[new_state[0],new_state[1],:])
    
    self.value_estimates[old_state[0],old_state[1],action] += alpha*(reward+gamma*self.value_estimates[new_state[0],new_state[1],estimated_action]-self.value_estimates[old_state[0],old_state[1],action])

    a_star = np.argmax(self.value_estimates[old_state[0],old_state[1],:])

    # greedy 
    num_actions = 6
    for i in range(num_actions):
        if i == a_star:
            self.policy[old_state[0],old_state[1],i] = 1-epsilon+epsilon/num_actions
        else:
            self.policy[old_state[0],old_state[1],i] = epsilon/num_actions

    return 0 

def get_sarsa_reward(self,state,old_state,new_state,events,crates):
   
    pos = old_state['self'][3]
    visited_positions = self.visited_positions

    new_pos = new_state['self'][3]
    target_coin = self.target_coin
    
    game_rewards = {
        e.COIN_COLLECTED: 1,
        e.KILLED_OPPONENT: 1,
        e.BOMB_DROPPED: 0.25,
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
        e.KILLED_SELF: -3,
        e.CRATE_DESTROYED: 2,
        e.COIN_FOUND: 2,
        PLACEHOLDER_EVENT: -.1  # idea: the custom event is bad
    }


    #print(i," ",pos," ",new_pos)#," ",visited_positions)
    reward_ctr = move_handling(state,new_pos,pos,visited_positions)
    #print("Visited Position:", reward_ctr)
    #print("Moved in the right direction:", reward_ctr)
    reward_ctr = coin_handling(reward_ctr,new_pos,pos,target_coin)

    if 'BOMB_DROPPED' in events: 
        reward_ctr = crate_handling(pos,reward_ctr,crates)
        self.bomb = pos 

    if 'BOMB_EXPLODED' in events: 
        self.bomb = None 

    if self.bomb:
        new_pos_euclidean = euclidean_distance(new_pos,self.bomb)

        if avoid_check_(new_pos,self.bomb):
            reward_ctr += 5 
        elif new_pos_euclidean >= 1 and new_pos_euclidean < 2 and new_pos != pos:
            reward_ctr += 3
        elif new_pos_euclidean >= 2 and new_pos_euclidean < 3 and new_pos != pos:
            reward_ctr += 4
        elif new_pos_euclidean >= 3 and new_pos != pos:
            reward_ctr += 5
        return reward_ctr

    for event in events:
        reward_ctr += game_rewards[event]
    
    return reward_ctr


def n_sarsa(self,game_state,crates):
    i = 0 
    for event in self.event_history: 
        self.reward_history.append(n_sarsa_reward(self,crates,event,i,game_state))
        i+=1

    if game_state['round'] < 1000:
        epsilon = 0.4
    elif game_state['round'] >= 1000 and game_state['round'] < 3000:
        epsilon = 0.3
    elif game_state['round'] >= 3000 and game_state['round'] < 5000:
        epsilon = 0.2
    else:
        epsilon = 0.1
    g = 0 
    t = 0
    disc= 0.95 
    alpha = 0.15
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

        #if state == [1,0]:
        #    print(self.action_history[t])

        if True:#(state,action) not in self.first_visit_check:
            self.first_visit_check.append((state,action))
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
        reward_ctr -= 2 
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

    if euclidean_distance(pos,close_object) >= new_pos_euclidean:
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
        e.WAITED: -3,
        e.INVALID_ACTION: -4,
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
    self.reward_debug_history.append(reward_ctr)        
    return reward_ctr