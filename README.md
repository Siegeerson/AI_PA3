# NANNON

## Usage

Start a python session or open a Jupyter Lab/Jupyter Notebook within the `nannon_skeleton` directory.  

You can import all the functions in each of the modules with the following command:  

```python
from nannon import *
```

Open `Nannon_Demo.ipynb` for an interactive demo of all the features.  

(You can download the Anaconda distribution, which includes Jupter Lab, at the following URL: https://www.anaconda.com/distribution/)  

Otherwise, you can consult the log below and the comments in the code and try it yourself.  

```python
from nannon import *

start_pos                   # ((0, 1, 2), (0, 1, 2))
end_tuple                   # (7, 7, 7)
roll()                      # 5
first_roll()                # 2

pos = ((0, 0, 1), (0, 2, 3))
print_board(pos)            # oo||o--**-||*  
swapped = swap_players(pos)
print(swapped)              # ((0, 2, 3), (0, 0, 1))
print_board(swapped)        # o||-oo--*||**


pos = ((0, 0, 1), (0, 2, 3))
print_board(pos)            # oo||o--**-||*  
print(who_won(pos))         # 0.5

pos = ((7, 7, 7), (0, 0, 1))
print_board(pos)            # ||-----*||**
print(who_won(pos))         # 1.0

pos = ((0, 0, 1), (7, 7, 7))
print_board(pos)            # oo||o-----||  
print(who_won(pos))         # 0.0


pos = ((0, 0, 1), (0, 2, 3))
print_board(pos)           # oo||o--**-||*  
print(legal_moves(pos, 2)) # [0, 2]
legal_moves(pos, 3)        # [0]



pos = ((1, 2, 3), (1, 2, 3))
print_board(pos)           # ||ooo***||  
legal_moves(pos, 2)        # [-1]



pos = ((0, 1, 2), (0, 1, 2))
die = 3
print('start')
print_board(pos)              # o||oo--**||*  
lm = legal_moves(pos, die)
print(die, lm)                # 3 [0, 1]

m0 = make_move(pos, 0, die)
print_board(m0)               # ||ooo-**||*  

m1 = make_move(pos, 1, die)
print_board(m1)               # o||-o-o**||*


pos_dict = explore()
print(len(pos_dict))   # 2530
print(dict(list(pos_dict.items())[:5]))
# {
#    ((6, 7, 7), (4, 5, 6)): 0.5,
#    ((0, 4, 7), (5, 6, 7)): 0.5,
#    ((0, 0, 0), (1, 3, 6)): 0.5,
#    ((3, 5, 6), (3, 5, 6)): 0.5,
#    ((2, 7, 7), (0, 0, 2)): 0.5
# }

rand_play   # <function nannon.players.rand_play(pos, roll)>

die = 2
print_board(start_pos)            # o||oo--**||*  
npos = rand_play(start_pos, die)
print(npos)                       # ((0, 2, 3), (0, 1, 2))
print_board(npos)                 # o||-oo-**||*  


first_play # <function nannon.players.first_play(pos, roll)>

die = 2
print_board(start_pos)           # o||oo--**||*  
npos = first_play(start_pos, die)
print(npos)                      # ((0, 1, 4), (0, 1, 2))
print_board(npos)                # o||o--o**||*  

play_game(rand_play, first_play)  # 'second'

play_tourn(rand_play, first_play) # 0.5385


players = [rand_play, first_play, last_play, score_play]
matrix = round_robin(players)
print(matrix)

#[[0 1 0 0]
# [0 0 0 0]
# [0 0 0 0]
# [0 0 0 0]]

interpret_matrix(matrix)  # {0: 1, 1: 0, 2: 2, 3: 3}

import pickle
mediocre_table = pickle.load(open('nannon/mediocre_table.p', 'rb'))
print(dict(list(mediocre_table.items())[:5]))

# {
#   ((6, 7, 7), (4, 5, 6)): 1.0265792950333243,
#   ((0, 5, 7), (0, 3, 6)): 0.8202917692035061,
#   ((3, 6, 7), (2, 7, 7)): 0.5697342743925066,
#   ((3, 5, 6), (3, 5, 6)): 0.8711220320949948,
#   ((2, 7, 7), (0, 0, 2)): 0.9591601191402004
# }



input_nodes = 2
hidden_nodes = 6
output_nodes = 1
learning_rate = 0.3

# Creates an instance of the scratch neural network.
# Here we teach it how to produce correct "XOR" output.
n = ScratchNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
X = [[0,0],
     [0,1],
     [1,0],
     [1,1]]
y = [[0],
     [1],
     [1],
     [0]]

print('Before:', n.query(X))
for _ in range(5000):
    n.train(X, y)
print('After', n.query(X))

# Before: [[0.60018041 0.60921318 0.74879427 0.72999071]]
# After [[0.02426062 0.98082423 0.97728005 0.01916943]]
```
