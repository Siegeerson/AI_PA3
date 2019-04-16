n_pos = 6
n_chex = 3
n_die = 6

safety = n_pos + 1
passv = -1
start_pos = tuple([tuple(range(n_chex))] * 2) # E.g., ((0, 1, 2), (0, 1, 2))
end_tuple = tuple([safety] * n_chex) # E.g., (7, 7, 7)

def init_globals(np=6, nc=3, nd=6):
    global n_pos, n_chex, n_die, safety, passv, start_pos, end_tuple
    n_pos = np
    n_chex = nc
    n_die = nd
    safety = n_pos + 1 # Index of safety always 1 greater than n_pos
    passv = -1 # A dummy value used to indicate when there are no moves.
    start_pos = tuple([tuple(range(n_chex))] * 2) # E.g., ((0, 1, 2), (0, 1, 2))
    end_tuple = tuple([safety] * n_chex) # E.g., (7, 7, 7)
