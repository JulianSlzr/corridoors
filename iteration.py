import sys
import operator
import copy
import numpy as np
import itertools

# Problem params
K = 2
NUM_CORRIDOORS = 2

# cost function's base; try 1.1, 1.5, 2, etc. to see difference 
BASE = 1.5

# State space border
MAX_DEPTH = 10

# Iterations (can also code as V_old - V < epsilon)
ITERATIONS = 100

# Want possible non-terminal (depth, coins) pairs
# e.g., when K = 3, we want (1, 1), (3, 2), but not (1, 2) or (3, 3)
TUPLES = list(itertools.ifilter(lambda x: x[0] >= x[1], itertools.product(range(MAX_DEPTH+1), range(K))))

# States are of the form ((depth_0, coins_0), (depth_1, coins_1))
STATES = list(itertools.product(TUPLES, TUPLES))

# Actions either travel down corridoor 0 or 1
ACTIONS = list(range(NUM_CORRIDOORS))

# The cost for an action and state
# e.g., if depth_0 = 2 and you proceed down 0, then you incur BASE^2
def cost(state, action):
    return BASE**state[action][0]

# Value iteration
def policy():

    # Setup index correspondence
    lookup = {}
    for i, state in enumerate(STATES):
        lookup[state] = i;

    # stage-optimal value function
    V = np.zeros((len(STATES), 1))

    # policy function
    P = np.zeros((len(STATES), 1))

    # costs over the actions
    costs = np.zeros((len(ACTIONS), 1))

    for i in xrange(ITERATIONS):
        V_old = V

        for state in STATES:

            state_i = lookup[state]

            for action in ACTIONS:

                # "finitizing the state space"
                # e.g., If we went from ((300, 4), (5, 1)) to ((301, 5), (5, 1)), it's like looping back to ((300, 5), (5, 1))
                no_coin = np.array(state)
                if no_coin[action][0] < MAX_DEPTH:
                    no_coin[action][0] += 1

                # no need to finitize wrt coins, because coins being K is terminal
                coin = np.copy(no_coin)
                coin[action][1] += 1

                is_winner = coin[action][1] == K

                no_coin = tuple(map(tuple, no_coin))
                coin = tuple(map(tuple, coin))

                if is_winner:
                    costs[action] = cost(state, action) + 0.5 * V_old[lookup[no_coin]]
                else:
                    costs[action] = cost(state, action) + 0.5 * V_old[lookup[no_coin]] + 0.5 * V_old[lookup[coin]]

            P[state_i] = np.argmin(costs)
            V[state_i] = costs[P[state_i][0]]

        # print STATES[:10]
        # print V[:10].T

    # pretty print
    for state_i, action in enumerate(P):
        print '{}: {}'.format(STATES[state_i], int(action[0]))

if __name__ == "__main__":
    policy()