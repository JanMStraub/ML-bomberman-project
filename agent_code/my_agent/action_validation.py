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
validation_array[0][0] = [2]
validation_array[0][1] = [3]
validation_array[0][2] = [2,3]
validation_array[0][3] = [2,3]
validation_array[0][4] = [3]
validation_array[0][5] = [2]
validation_array[0][6] = [3]
validation_array[0][7] = [2]
validation_array[0][8] = [5]

validation_array[1][0] = [0]
validation_array[1][1] = [3]
validation_array[1][2] = [0,3]
validation_array[1][3] = [0,3]
validation_array[1][4] = [0]
validation_array[1][5] = [3]
validation_array[1][6] = [0]
validation_array[1][7] = [3]
validation_array[1][8] = [5]

validation_array[2][0] = [2]
validation_array[2][1] = [1]
validation_array[2][2] = [1,2]
validation_array[2][3] = [1,2]
validation_array[2][4] = [1]
validation_array[2][5] = [2]
validation_array[2][6] = [1]
validation_array[2][7] = [2]
validation_array[2][8] = [5]

"0:UP 1:RIGHT 2:DOwN 3:LEFT 4:WAIT 5:BOMB"
validation_array[3][0] = [0]
validation_array[3][1] = [1]
validation_array[3][2] = [0,1]
validation_array[3][3] = [0,1]
validation_array[3][4] = [0]
validation_array[3][5] = [1]
validation_array[3][6] = [0]
validation_array[3][7] = [1]
validation_array[3][8] = [5]

validation_array[4][0] = [3]
validation_array[4][1] = [1]
validation_array[4][2] = [1,3]
validation_array[4][3] = [1,3]
validation_array[4][4] = [3]
validation_array[4][5] = [1]
validation_array[4][6] = [3]
validation_array[4][7] = [1]
validation_array[4][8] = [5]

validation_array[5][0] = [2]
validation_array[5][1] = [0]
validation_array[5][2] = [0,3]
validation_array[5][3] = [0,3]
validation_array[5][4] = [0]
validation_array[5][5] = [2]
validation_array[5][6] = [0]
validation_array[5][7] = [2]
validation_array[5][8] = [5]

"0:UP 1:RIGHT 2:DOwN 3:LEFT 4:WAIT 5:BOMB"
validation_array[6][0] = [2]
validation_array[6][1] = [1]
validation_array[6][2] = [2,3]
validation_array[6][3] = [0]
validation_array[6][4] = [1]
validation_array[6][5]= [0,1]
validation_array[6][6] = [2]
validation_array[6][7] = [0]
validation_array[6][8] = [0,2]
validation_array[6][9] = [0,1,2]
validation_array[6][10] = [0]
validation_array[6][11] = [1]
validation_array[6][12] = [2]

validation_array[7][0] = [2]
validation_array[7][1] = [3]
validation_array[7][2] = [2,3]
validation_array[7][3] = [0]
validation_array[7][4] = [3]
validation_array[7][5] = [0,3]
validation_array[7][6] = [2]
validation_array[7][7] = [0]
validation_array[7][8] = [0,2]
validation_array[7][9] = [0,2,3]
validation_array[7][10] = [0]
validation_array[7][11] = [3]
validation_array[7][12] = [2]

"0:UP 1:RIGHT 2:DOwN 3:LEFT 4:WAIT 5:BOMB"
validation_array[8][0] = [2]
validation_array[8][1] = [3]
validation_array[8][2] = [2,3]
validation_array[8][3] = [1]
validation_array[8][4] = [3]
validation_array[8][5] = [1,3]
validation_array[8][6] = [2]
validation_array[8][7] = [1]
validation_array[8][8] = [1,2]
validation_array[8][9] = [1,2,3]
validation_array[8][10] = [1]
validation_array[8][11] = [3]
validation_array[8][12] = [2]

validation_array[9][0] = [0]
validation_array[9][1] = [3]
validation_array[9][2] = [0,3]
validation_array[9][3] = [1]
validation_array[9][4] = [3]
validation_array[9][5] = [1,3]
validation_array[9][6] = [0]
validation_array[9][7] = [1]
validation_array[9][8] = [0,1]
validation_array[9][9] = [0,1,3]
validation_array[9][10] = [1]
validation_array[9][11] = [3]
validation_array[9][12] = [0]

"0:UP 1:RIGHT 2:DOwN 3:LEFT 4:WAIT 5:BOMB"
validation_array[10][0] = [1]
validation_array[10][1] = [3]
validation_array[10][2] = [0]
validation_array[10][3] = [2]

validation_array[10][4] = [0,1,2,3]

validation_array[10][5] = [1]
validation_array[10][6] = [3]
validation_array[10][7] = [0]
validation_array[10][8] = [2]


correct_ctr = 0
ctr = 0

for x in range(11):
    print(" ")
    for y in range(6):
        if np.argmax(policy[x,y]) in validation_array[x][y]:
            print(x," ",y," ",np.argmax(policy[x,y])," ",validation_array[x][y]," correct")
            ctr += 1
            correct_ctr +=1
        else:
            print(x," ",y," ",np.argmax(policy[x,y])," ",validation_array[x][y]," not correct")
            
            ctr += 1

print(correct_ctr," of ", ctr, " are correct")
