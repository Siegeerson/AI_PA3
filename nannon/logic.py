import random
import nannon.global_vars as g
from sys import stdout

################################################################################

                                ##########
                                # NANNON #
                                ##########

# The board representation is as follows: a tuple composed of two tuples, e.g.:
#     ((0, 1, 2), (0, 1, 2)) --> starting position in Nannon {6, 3, 6}
#
#     ((0, 1, 2), (0, 1, 2)) == R<RR--BB>B
#
# Where
#     1. The left tuple corresponds to the current player
#     2. The right tuple corresponds to the inactive player
#     3. The length of each inner tuple is n_chex
#     4. The value of each inner tuple's elements is in range(0, n_pos + 1)
#     5. Each value corresponds to a piece's board position,
#         from its owner's perspective.
#     6. The values of each inner tuple must appear in sorted order.

################################################################################

                ##########################################
                # PART ONE: BASIC FUNCTIONS / GAME LOGIC #
                ##########################################

# Returns a random number from 1 to n_die.
def roll():
    return random.randint(1, g.n_die)

# Keeps "rolling" until a non-zero value is found.
def first_roll():
    diff = 0
    while diff == 0:
        diff = roll() - roll()
    return diff

# The dicestream here acts as a queue. When used, we shift out the first element.
def dicestream(n=100):
    return [abs(first_roll())] + [roll() for _ in range(n - 1)]

# Reverses the position tuple.
def swap_players(pos):
    return tuple(reversed(pos))

# Returns 1.0 if the first player won, 0.0 if second player won, 0.5 otherwise.
def who_won(pos):
    me, you = pos
    if me == g.end_tuple: # E.g., ((7, 7, 7), (0, 1, 2))
        return 1.0
    if you == g.end_tuple: # E.g., ((0, 1, 2), (7, 7, 7))
        return 0.0
    return 0.5

# Builds a readable representation of the board, printing character by character
# to stdout.
def print_board(pos):
    me, you = pos
    for _ in range(g.n_chex - 1): stdout.write(' ')
    for x in reversed(me): stdout.write('o' if x == 0 else ' ')
    stdout.write('||')
    for idx in range(1, g.safety):
        if idx in me: stdout.write('o')
        elif idx in flip(you): stdout.write('*')
        else: stdout.write('-')
    stdout.write('||')
    for x in you: stdout.write('*' if x == 0 else ' ')
    stdout.write('\n')

# A move is represented by the index of the piece to be move in the active
# player's tuple. For example,
#
#   pos = ((0, 1, 2), (0, 1, 2))
#   die = 3
#   legal_moves(pos, die) -> [0, 1]   # Index 2 is blocked by the prime
#   make_move(pos, 0, die) -> ((1, 2, 3), (0, 1, 2))
#   make_move(pos, 1, die) -> ((0, 2, 4), (0, 1, 2))
#   make_move(pos, 2, die) -> Exception: Invalid move!
#
# Recall that passv == -1. When a position is blocked, and the turn must be
# forfeited, a dummy value of [-1] is returned. Making a move with -1 will
# return the same position originally passed.
#
#   pos = ((1, 2, 3), (1, 2, 3))
#   die = 1
#   legal_moves(pos, die) -> [-1]
#   make_move(pos, 0, die) -> Exception: Invalid move!
#   make_move(pos, -1, die) -> ((1, 2, 3), (1, 2, 3))
def legal_moves(pos, die):
    me, you = pos
    lm = []
    for idx, piece in enumerate(me):
        if piece == g.safety: break
        if piece == 0 and idx > 0: continue
        if not piece + die in get_blocked(me, you):
            lm.append(idx)
    return lm if lm else [g.passv]

# Returns a list of the spaces that cannot be moved to.
# Adds one's own pieces as well as the other's primes.
#
#   pos = ((0, 1, 2), (0, 1, 2))
#   me, you = pos
#   get_blocked(me, you) -> {1, 2, 5, 6}
def get_blocked(me, you):
    bl = set(filter(lambda x: x != 0 and x != g.safety, me))
    you = flip(you)
    for x in range(1, g.n_pos):
        if x in you and x + 1 in you:
            bl.update((x, x + 1))
    return bl

# Reverses the values of the tuple, leading to the representation
# from the other's perspective.
#
#   pos = ((0, 1, 2), (0, 7, 7))
#   me, you = pos
#   flip(me) -> (5, 6, 7)
#   flip(you) -> (0, 0, 7)
def flip(player_tup):
    return tuple(reversed(tuple(map(lambda x: g.safety - x, player_tup))))

# See comment above legal_move for demonstration.
# Checks if move is legal first and throws exception if not.
# Returns original position if the pass value (-1) is sent.
# Bounces opponent back to home if the new position lands on it.
def make_move(pos, checker, roll):
    if checker not in legal_moves(pos, roll):
        raise Exception('Invalid move!')
    if checker == g.passv:
        return pos # Returns orig if a pass is sent.
    me, you = pos
    newp = min(me[checker] + roll, g.safety)
    me = me[:checker] + (newp,) + me[checker + 1:] # Update my piece
    you = flip(you)
    if newp in you:
        hit = you.index(newp)
        you = you[:hit] + you[hit + 1:] + (g.safety,) # Bounce yours to home.
    return tuple(sorted(me)), flip(you)
