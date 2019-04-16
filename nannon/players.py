from nannon.logic import *
from functools import reduce

################################################################################

                            ###########
                            # PLAYERS #
                            ###########

# Returns the result of making a random move.
def rand_play(pos, roll):
    move = random.choice(legal_moves(pos, roll))
    return make_move(pos, move, roll)

# Prints board, shows legal moves, and queries the user for input.
def human(pos, roll):
    lm = legal_moves(pos, roll)
    print_board(pos)
    print('You rolled:', roll)
    print('Legal moves:', lm)
    move = int(input('Your move? '))
    if move not in lm:
        print('Try again.')
        human(pos, roll)
    return make_move(pos, move, roll)

# Returns the position resulting from moving the furthest forward piece.
# We can do this because the tuple will always be sorted.
def first_play(pos, roll):
    move = legal_moves(pos, roll)[-1]
    return make_move(pos, move, roll)

# Returns the position resulting from moving the furthest back piece.
def last_play(pos, roll):
    move = legal_moves(pos, roll)[0]
    return make_move(pos, move, roll)

# For each legal move, sums the scores for each player.
# Appends the move along with the score to a list.
# Returns the result of making the move with the highest score value.
def score_play(pos, roll):
    candidates = []
    for move in legal_moves(pos, roll):
        me, you = make_move(pos, move, roll)
        me_score = reduce(lambda x, y: x + y, me)
        you_score = reduce(lambda x, y: x + y, you)
        candidates.append((move, me_score - you_score))
    best_move, _ = max(candidates, key=lambda x: x[1])
    return make_move(pos, best_move, roll)

# You may choose to implement new players below
