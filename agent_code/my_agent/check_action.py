import os
import pickle
import random
import sys 

import numpy as np

with open("my-saved-model.pt", "rb") as file:
    model = pickle.load(file)
    policy = model

x = int(sys.argv[1])
y = int(sys.argv[2])


for x in range(11):
    print(" ")
    for y in range(17):
        action = np.argmax(policy[x,y,:])
        
        if action == 0:
            action_str = "UP"
        elif action == 1:
            action_str = "RIGHT"
        elif action == 2:
            action_str = "DOWN"
        elif action == 3:
            action_str = "LEFT"
        elif action == 4:
            action_str = "WAIT"
        else:
            action_str = "BOMB"

        print(x," ",y," ",action_str)
            
