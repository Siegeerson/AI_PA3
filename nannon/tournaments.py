import numpy as np
import nannon.global_vars as g
from nannon.logic import who_won, swap_players, dicestream, roll, first_roll

################################################################################

                        ########################
                        # PART 3: TOURNAMENTS! #
                        ########################

# Use dicestream if provided, else use existing methods.
# Play game until a winner is found, swapping players and rolling new die.
# Returns a string (either 'first' or 'second') corresponding to the player
# that won.
#
#   play_game(rand_play, first_play) -> 'first'
#
def play_game(play1, play2, dicestream=None):
    players, pos, r = init_game(play1, play2, dicestream)
    while True:
        currp, nextp = players
        playfunc, playorder = currp
        pos = playfunc(pos, r)
        if who_won(pos) != 0.5:
            return playorder
        players = nextp, currp
        r = dicestream.pop(0) if dicestream else roll()
        pos = swap_players(pos)

def init_game(play1, play2, dicestream):
    players = (play1, 'first'), (play2, 'second')
    r = 0
    if not dicestream:
        r = first_roll()
        if r < 0:
            players = (play2, 'first'), (play1, 'second')
            r = abs(r)
    else:
        r = dicestream.pop(0)
    return players, g.start_pos, r

# Plays 2 * n_pairs games between the provided players.
# n_ds is the length of the dicestream to use.
# Generates dicestream, then tracks the results of both games while swapping the
# players for the second game.
# Returns a float between (0, 1) corresponding the percentage of games that
# player 1 won.
#
#   play_tourn(rand_play, first_play) -> 0.525, for example
#
def play_tourn(play1, play2, n_pairs=1000, n_ds=50):
    play1_wins = 0
    for _ in range(n_pairs):
        ds = dicestream(n_ds)
        res1 = play_game(play1, play2, ds.copy())
        res2 = play_game(play2, play1, ds.copy())
        if res1 == 'first':
            play1_wins += 1
        if res2 == 'second':
            play1_wins += 1
    return play1_wins / (n_pairs * 2)

# Uses the "circle" round-robin tournament algorithm, as described here:
#      https://en.wikipedia.org/wiki/Round-robin_tournament#Scheduling_algorithm
# Adds a 'bye' player in case there is an odd number of players.
# Populates an upper triangular matrix to reflect the result of each pairing.
#
#   players = [rand_play, first_play, last_play, score_play]
#   round_robin(players) -> array([[0, 1, 0, 0],
#                                  [0, 0, 0, 0],
#                                  [0, 0, 0, 0],
#                                  [0, 0, 0, 0]])
#
def round_robin(players, n_pairs=1000, n_ds=50):
    upper_tri_matrix = np.zeros((len(players), len(players)), dtype=np.int)
    circle, first_player, half_idx = init_circle(players)
    for _ in range(len(circle)):
        row1 = [first_player] + circle[:half_idx]
        row2 = circle[half_idx:]
        for (play1, id1), (play2, id2) in zip(row1, row2):
            if play1 == 'bye' or play2 == 'bye': continue
            p1_advg = play_tourn(play1, play2, n_pairs, n_ds)
            if p1_advg > 0.5 and id1 < id2:
                upper_tri_matrix[id1, id2] = 1
        circle.append(circle.pop(0))
    return upper_tri_matrix

def init_circle(players):
    circle = players.copy()
    if len(players) % 2 != 0:
        circle += ['bye']
    circle = list(zip(circle, range(len(circle))))
    first_player = circle.pop(0)
    return circle, first_player, int(len(circle) / 2)

# Helper for interpreting the results of a tournmanet matrix.
# With an input upper-tri-matrix, returns a dictionary recording the number of
# wins per player in the order originally passed in.
#
#   players = [rand_play, first_play, last_play, score_play]
#   matrix = round_robin(players)
#   interpret_matrix(matrix) -> {0: 1, 1: 0, 2: 2, 3: 3}
#
def interpret_matrix(matrix):
    results = {key: 0 for key in range(len(matrix))}
    for idx in range(len(matrix)):
        for jdx in range(len(matrix)):
            if idx < jdx:
                if matrix[idx, jdx] == 1:
                    results[idx] += 1
                else:
                    results[jdx] += 1
    return results
