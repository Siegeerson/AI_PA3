import nannon.global_vars as g
from scipy.special import comb
from nannon.logic import swap_players, who_won, legal_moves, make_move

################################################################################

                ######################################
                # PART 2: GENERALIZATION OF THE GAME #
                ######################################

# The "explore" method uses a queue, initialized with an "all-home" position,
#     eg., ((0, 0, 0), (0, 0, 0))
# and an "all-safe" position,
#     eg., ((7, 7, 7), (7, 7, 7))
# 1) Shift first element of the queue, 2) add the reversed pos to the queue
# 3) if the pos is not the dictionary, add it, along with (-1, 0, 1) value of
# who_won, 4) if not done, add all the positions resulting from legal moves
# to the queue.
def explore():
    V = {}
    queue = init_queue()
    while queue:
        pos = queue.pop(0)
        swapped = swap_players(pos)
        if swapped not in V:
            queue.append(swapped)
        if pos not in V:
            winner = who_won(pos)
            V[pos] = winner
            if winner == 0.5:
                add_branches(pos, queue)
    return V

def init_queue():
    all_home = tuple([tuple(0 for _ in range(g.n_chex))] * 2)
    all_safe = tuple([tuple(g.safety for _ in range(g.n_chex))] * 2)
    return [all_home, all_safe]

def add_branches(pos, queue):
    for roll in range(1, g.n_die + 1):
        moves = legal_moves(pos, roll)
        if g.passv in moves: continue
        for checker in moves:
            npos = make_move(pos, checker, roll)
            queue.append(npos)

# Combinatorical equation that returns the true number of possible positions.
def get_n_items():
    n_items = 0
    n = g.n_pos
    k = g.n_chex
    for i in range(k + 1):
        for j in range(k + 1):
            n_items += comb(n, i) * comb(n - i, j) * (k + 1 - i) * (k + 1 - j)
    return int(n_items)
