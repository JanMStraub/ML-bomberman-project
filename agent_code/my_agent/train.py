from collections import namedtuple, deque

import pickle
from typing import List

import events as e
from .callbacks import state_to_features

import numpy as np

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

    self.state_history.append(old_game_state['self'][3])
    self.action_history.append(self_action)
    self.event_history.append(events)

    #self.events.append(self_action)

    #print(self)
    #print(self.train)
    #print(self.action)
    #print(self.reward)
    

    # state_to_features is defined in callbacks.py
    self.transitions.append(Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward_from_events(self, events)))

def get_reward(event):
    game_rewards = {
        e.COIN_COLLECTED: 5,
        e.KILLED_OPPONENT: 1,
        e.BOMB_DROPPED: -1,
        e.MOVED_LEFT: 1,
        e.MOVED_RIGHT: 1,
        e.MOVED_UP: 1,
        e.MOVED_DOWN: 1,
        e.WAITED: 0,
        e.INVALID_ACTION: -1,
        PLACEHOLDER_EVENT: -.1  # idea: the custom event is bad
    }
    return game_rewards[event[0]]

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
    self.transitions.append(Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events)))

    for event in self.event_history: 
        self.reward_history.append(get_reward(event))

    #print(sum(self.reward_history[3:]))

    t = 0 
    for state_x,state_y in self.state_history:
        self.value_estimates[state_x,state_y] += 0.2 *(sum(self.reward_history[t:])-self.value_estimates[state_x,state_y])
        #print(self.event_history[t])
        t+=1


    epsilon_greedy(self)
    #print(self.total_score)

    self.state_history = []
    self.action_history = []
    self.event_history = []
    self.reward_history = []

    #print(self.reward_history)

    #print(self.events)
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


def epsilon_greedy(self):
    #[0:'UP', 1:'RIGHT', 2:'DOWN', 3:'LEFT']
    for x in range(17):
        for y in range(17):
            neighbour_values = np.zeros(4)
            # Up 
            if y-1 >= 0:
                neighbour_values[0] = self.value_estimates[x,y-1]
            else:
                # left
                if x-1 >= 0:
                    neighbour_values[0] = -9999
                    neighbour_values[3] = -9999
                # right
                elif x+1 <= 16:
                    neighbour_values[0] = -9999
                    neighbour_values[1] = -9999
                else:
                    neighbour_values[0] = -9999

            # Right
            if x+1 <= 16:
                neighbour_values[1] =self.value_estimates[x+1,y]
            else:
                neighbour_values[1] = -9999

            # Down
            if y+1 <= 16:
                neighbour_values[2] = self.value_estimates[x,y+1]
            else:
                # left
                if x-1 >= 0:
                    neighbour_values[2] = -9999
                    neighbour_values[3] = -9999
                # right
                elif x+1 <= 16:
                    neighbour_values[2] = -9999
                    neighbour_values[1] = -9999
                else:
                    neighbour_values[2] = -9999

            # Left
            if x-1 >= 0:
                neighbour_values[3] = self.value_estimates[x-1,y]
            else:
                neighbour_values[3] = -9999

            idx = np.argmax(neighbour_values)
            self.policy[x,y] = idx
            #self.policy[x,y] = np.zeros(4)
            #self.policy[x,y][idx] = 1 

        
    return self.policy


def mc_eval():
    """
    Monte-Carlo evaluation.
    """
    return 0
