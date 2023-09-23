import os
import pickle
import random
import sys 

import numpy as np

with open("my-saved-model.pt", "rb") as file:
    model = pickle.load(file)
    policy = model


validation_array = np.zeros((11,17))

# Define dimensions
x = 11
y = 20
z = 1

# Initialize an empty 3D list
validation_array = [[[[] for _ in range(z)] for _ in range(y)] for _ in range(x)]

"0:UP 1:RIGHT 2:DOwN 3:LEFT 4:WAIT 5:BOMB"
validation_array[0][0] = ["UP"]
validation_array[0][1] = ["RIGHT"]
validation_array[0][2] = ["DOWN"]
validation_array[0][3] = ["LEFT"]
validation_array[0][4] = ["UP","RIGHT","DOWN","LEFT"]
validation_array[0][5] = ["NONE"]
validation_array[0][6] = ["NONE"]
validation_array[0][7] = ["NONE"]

validation_array[1][0] = ["UP"]
validation_array[1][1] = ["RIGHT"]
validation_array[1][2] = ["DOWN"]
validation_array[1][3] = ["LEFT"]
validation_array[1][4] = ["NONE"]
validation_array[1][5] = ["NONE"]
validation_array[1][6] = ["NONE"]
validation_array[1][7] = ["Placeholder"]

validation_array[2][0] = ["BOMB"]
validation_array[2][1] = ["UP"]
validation_array[2][2] = ["RIGHT"]
validation_array[2][3] = ["DOWN"]
validation_array[2][4] = ["LEFT"]
validation_array[2][5] = ["NONE"]
validation_array[2][6] = ["NONE"]
validation_array[2][7] = ["NONE"]

"0:UP 1:RIGHT 2:DOwN 3:LEFT 4:WAIT 5:BOMB"
validation_array[3][0] = ["UP"]
validation_array[3][1] = ["RIGHT"]
validation_array[3][2] = ["DOWN"]
validation_array[3][3] = ["LEFT"]
validation_array[3][4] = ["UP"]
validation_array[3][5] = ["RIGHT"]
validation_array[3][6] = ["DOWN"]
validation_array[3][7] = ["LEFT"]



correct_ctr = 0
ctr = 0

def get_value(i):
    if i == 0:
        action = 'UP'
    elif i == 1:
        action = 'RIGHT'
    elif i == 2:
        action = 'DOWN'
    elif i == 3:
        action = 'LEFT'
    elif i == 4:
        action = 'WAIT'
    elif i == 5: 
        action = "BOMB"
    return action 

for x in range(4):
    print(" ")
    for y in range(8):
        if get_value(np.argmax(policy[x,y])) in validation_array[x][y]:
            print(x," ",y," ",np.argmax(policy[x,y])," ",validation_array[x][y]," correct")
            ctr += 1
            correct_ctr +=1
        else:
            print(x," ",y," ",np.argmax(policy[x,y])," ",validation_array[x][y]," not correct")
            if "NONE" not in validation_array[x][y]:
                ctr += 1

print(correct_ctr," of ", ctr, " are correct")
