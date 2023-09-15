from collections import namedtuple, deque

import pickle
from typing import List

import events as e
from .callbacks import state_to_features, extract_state

import numpy as np

from statistics import mean

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

    sarsa(self,old_game_state,self_action,new_game_state,get_reward(self,old_game_state,events))

    self.state_history.append(extract_state(self,old_game_state))
    self.action_history.append(self_action)
    self.event_history.append(events)
    self.visited_positions.append(old_game_state['self'][3])

    
    
    #self.value_estimates = state_to_features(self.value_estimates,old_game_state, new_game_state)
    #trans = Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward_from_events(self, events))

    # state_to_features is defined in callbacks.py
    #self.transitions.append(Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward_from_events(self, events)))

def get_reward(self,game_state,events):
    game_rewards = {
        e.COIN_COLLECTED: 5,
        e.KILLED_OPPONENT: 1,
        e.BOMB_DROPPED: -1,
        e.MOVED_LEFT: 0,
        e.MOVED_RIGHT: 0,
        e.MOVED_UP: 0,
        e.MOVED_DOWN: 0,
        e.WAITED: 0,
        e.INVALID_ACTION: -10,
        e.TILE_VISITED: -1,
        e.SURVIVED_ROUND: 0,
        PLACEHOLDER_EVENT: -.1  # idea: the custom event is bad
    }
    pos = game_state['self'][3]
    if pos in self.visited_positions:
        reward_ctr = -20
    else:
        reward_ctr = 0

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

    # MC-Control for update value-function and policy 
    #mc_control(self,last_game_state)

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


def mc_control(self,game_state):
    """
    Monte-Carlo Control 
    """
    for event in self.event_history: 
            self.reward_history.append(get_reward(self,game_state,event))

    epsilon = 0.2
    g = 0 
    t = 0
    gamma = 0.95 
    for state in self.state_history:
        #while t < len(self.state_history)-1:
            if self.action_history[t] == 'UP':
                action = 0
            elif self.action_history[t] == 'RIGHT':
                action = 1 
            elif self.action_history[t] == 'DOWN':
                action = 2
            else:
                action = 3
            g = pow(gamma,t) * g + self.reward_history[t]
            #g += self.reward_history[t]

            self.returns[state][action].append(g)
            self.value_estimates[state,action] = mean(self.returns[state][action])
            a_star = np.argmax(self.value_estimates[state,:])

            # greedy 
            for i in range(4):
                if i == a_star:
                    self.policy[state,i] = 1-epsilon+epsilon/4
                else:
                    self.policy[state,i] = epsilon/4
            t+=1
    
    self.state_history = []
    self.action_history = []
    self.event_history = []
    self.reward_history = []
    self.visited_positions = []

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
        action = 1
    else:
        action = 3

    estimated_action = np.random.choice([0,1,2,3], p = self.policy[new_state,:])
    
    self.value_estimates[old_state,action] += alpha*(reward+gamma*self.value_estimates[new_state,estimated_action]-self.value_estimates[old_state,action])

    a_star = np.argmax(self.value_estimates[old_state,:])

    # greedy 
    for i in range(4):
        if i == a_star:
            self.policy[old_state,i] = 1-epsilon+epsilon/4
        else:
            self.policy[old_state,i] = epsilon/4

    return 0 