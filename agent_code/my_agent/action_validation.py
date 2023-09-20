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
validation_array[0][0] = ["DOWN"]
validation_array[0][1] = ["LEFT"]
validation_array[0][2] = ["DOWN","LEFT"]
validation_array[0][3] = ["LEFT"]
validation_array[0][4] = ["DOWN"]
validation_array[0][5] = ["LEFT"]
validation_array[0][6] = ["DOWN"]
validation_array[0][7] = ["NONE"]
validation_array[0][8] = ["NONE"]

validation_array[1][0] = ["UP"]
validation_array[1][1] = ["LEFT"]
validation_array[1][2] = ["UP","LEFT"]
validation_array[1][3] = ["UP"]
validation_array[1][4] = ["LEFT"]
validation_array[1][5] = ["UP"]
validation_array[1][6] = ["LEFT"]
validation_array[1][7] = ["NONE"]
validation_array[1][8] = ["NONE"]

validation_array[2][0] = ["DOWN"]
validation_array[2][1] = ["RIGHT"]
validation_array[2][2] = ["DOWN","RIGHT"]
validation_array[2][3] = ["RIGHT"]
validation_array[2][4] = ["DOWN"]
validation_array[2][5] = ["RIGHT"]
validation_array[2][6] = ["DOWN"]
validation_array[2][7] = ["NONE"]
validation_array[2][8] = ["NONE"]

"0:UP 1:RIGHT 2:DOwN 3:LEFT 4:WAIT 5:BOMB"
validation_array[3][0] = ["UP"]
validation_array[3][1] = ["RIGHT"]
validation_array[3][2] = ["UP","RIGHT"]
validation_array[3][3] = ["UP"]
validation_array[3][4] = ["RIGHT"]
validation_array[3][5] = ["UP"]
validation_array[3][6] = ["RIGHT"]
validation_array[3][7] = ["NONE"]
validation_array[3][8] = ["NONE"]

validation_array[4][0] = ["LEFT"]
validation_array[4][1] = ["RIGHT"]
validation_array[4][2] = ["LEFT","RIGHT"]
validation_array[4][3] = ["LEFT"]
validation_array[4][4] = ["RIGHT"]
validation_array[4][5] = ["LEFT"]
validation_array[4][6] = ["RIGHT"]
validation_array[4][7] = ["NONE"]
validation_array[4][8] = ["NONE"]

validation_array[5][0] = ["DOWN"]
validation_array[5][1] = ["UP"]
validation_array[5][2] = ["DOWN","UP"]
validation_array[5][3] = ["UP"]
validation_array[5][4] = ["DOWN"]
validation_array[5][5] = ["UP"]
validation_array[5][6] = ["DOWN"]
validation_array[5][7] = ["NONE"]
validation_array[5][8] = ["NONE"]

"0:UP 1:RIGHT 2:DOwN 3:LEFT 4:WAIT 5:BOMB"
validation_array[6][0] = ["DOWN"]
validation_array[6][1] = ["UP"]
validation_array[6][2] = ["RIGHT"]
validation_array[6][3] = ["RIGHT","UP","DOWN"]
validation_array[6][4] = ["UP"]
validation_array[6][5]=  ["RIGHT"]
validation_array[6][6] = ["DOWN"]
validation_array[6][7] = ["UP"]
validation_array[6][8] = ["RIGHT"]
validation_array[6][9] = ["DOWN"]


validation_array[7][0] = ["DOWN"]
validation_array[7][1] = ["LEFT"]
validation_array[7][2] = ["UP"]
validation_array[7][3] = ["DOWN","LEFT","UP"]
validation_array[7][4] = ["UP"]
validation_array[7][5] = ["LEFT"]
validation_array[7][6] = ["DOWN"]
validation_array[7][7] = ["UP"]
validation_array[7][8] = ["LEFT"]
validation_array[7][9] = ["DOWN"]

"0:UP 1:RIGHT 2:DOwN 3:LEFT 4:WAIT 5:BOMB"
validation_array[8][0] = ["DOWN"]
validation_array[8][1] = ["LEFT"]
validation_array[8][2] = ["RIGHT"]
validation_array[8][3] = ["DOWN","LEFT","RIGHT"]
validation_array[8][4] = ["RIGHT"]
validation_array[8][5] = ["LEFT"]
validation_array[8][6] = ["DOWN"]
validation_array[8][7] = ["RIGHT"]
validation_array[8][8] = ["LEFT"]
validation_array[8][9] = ["DOWN"]


validation_array[9][0] = ["UP"]
validation_array[9][1] = ["LEFT"]
validation_array[9][2] = ["RIGHT"]
validation_array[9][3] = ["UP","LEFT","RIGHT"]
validation_array[9][4] = ["RIGHT"]
validation_array[9][5] = ["LEFT"]
validation_array[9][6] = ["UP"]
validation_array[9][7] = ["RIGHT"]
validation_array[9][8] = ["LEFT"]
validation_array[9][9] = ["UP"]

"0:UP 1:RIGHT 2:DOwN 3:LEFT 4:WAIT 5:BOMB"
validation_array[10][0] = ["RIGHT"]
validation_array[10][1] = ["LEFT"]
validation_array[10][2] = ["UP"]
validation_array[10][3] = ["DOWN"]
validation_array[10][4] = ["RIGHT","LEFT","DOWN","UP"]
validation_array[10][5] = ["RIGHT"]
validation_array[10][6] = ["LEFT"]
validation_array[10][7] = ["UP"]
validation_array[10][8] = ["DOWN"]
validation_array[10][9] = ["RIGHT"]
validation_array[10][10] = ["LEFT"]
validation_array[10][11] = ["TOP"]
validation_array[10][12] = ["DOWN"]


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

for x in range(11):
    print(" ")
    for y in range(6):
        if get_value(np.argmax(policy[x,y])) in validation_array[x][y]:
            print(x," ",y," ",np.argmax(policy[x,y])," ",validation_array[x][y]," correct")
            ctr += 1
            correct_ctr +=1
        else:
            print(x," ",y," ",np.argmax(policy[x,y])," ",validation_array[x][y]," not correct")
            if "NONE" not in validation_array[x][y]:
                ctr += 1

print(correct_ctr," of ", ctr, " are correct")
