import random
import numpy as np

queueBFS = []
queueDFS = []
queueUC = []
queueASTAR = []


''' ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ '''
'''
                For Search Algorithms 
'''
''' ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ '''

'''
BFS add to queue 
'''
def add_to_queue_BFS(node_id, parent_node_id, cost, initialize=False):
    global queueBFS
    if initialize == True:
        queueBFS[:] = []
        queueBFS.append((node_id, parent_node_id))
        return

    queueBFS.append((node_id, parent_node_id))

    return

'''
BFS is queue empty
'''
def is_queue_empty_BFS():
    global queueBFS
    return queueBFS == []

'''
BFS pop from queue
'''
def pop_front_BFS():
    global queueBFS
    (node_id, parent_node_id) = queueBFS.pop(0)
    return (node_id, parent_node_id)

'''
DFS add to queue 
'''
def add_to_queue_DFS(node_id, parent_node_id, cost, initialize=False):
    global queueDFS
    if initialize == True:
        queueDFS[:] = []
        queueDFS.append((node_id, parent_node_id))
        return
    queueDFS.append((node_id, parent_node_id))
    return

'''
DFS is queue empty
'''
def is_queue_empty_DFS():
    global queueDFS
    return queueDFS == []

'''
DFS pop from queue
'''
def pop_front_DFS():
    global queueDFS
    (node_id, parent_node_id) = queueDFS.pop()
    return (node_id, parent_node_id)

'''
UC add to queue 
'''
def add_to_queue_UC(node_id, parent_node_id, cost, initialize=False):
    global queueUC
    if initialize == True:
        queueUC[:] = []
        queueUC.append((cost, node_id, parent_node_id))
        return
    queueUC.append((cost, node_id, parent_node_id))
    queueUC.sort(reverse=True)
    return

'''
UC is queue empty
'''
def is_queue_empty_UC():
    global queueUC
    return queueUC == []

'''
UC pop from queue
'''
def pop_front_UC():
    global queueUC
    temp = queueUC.pop()
    (node_id, parent_node_id) = (temp[1], temp[2])
    return (node_id, parent_node_id)

'''
A* add to queue 
'''
def add_to_queue_ASTAR(node_id, parent_node_id, cost, initialize=False):
    global queueASTAR
    if initialize == True:
        queueASTAR[:] = []
        queueASTAR.append((cost, node_id, parent_node_id))
        return
    queueASTAR.append((cost, node_id, parent_node_id))
    queueASTAR.sort(reverse=True)
    return

'''
A* is queue empty
'''
def is_queue_empty_ASTAR():
    global queueASTAR
    return queueASTAR == []

'''
A* pop from queue
'''
def pop_front_ASTAR():
    global queueASTAR
    temp = queueASTAR.pop()
    (node_id, parent_node_id) = (temp[1], temp[2])
    return (node_id, parent_node_id)

''' ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ '''
'''
                For n-queens problem 
'''
''' ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ '''


'''
Compute a random state 
'''
def get_random_state(n):
    state = []
    for i in range(0, n):
        state.append(random.randint(0, n-1))
    return state

'''
Compute pairs of queens in conflict 
'''
def compute_attacking_pairs(state):
    number_attacking_pairs = 0
    n = len(state)
    for column1 in range(n):
        for column2 in range(column1 + 1, n):
            # check same row
            if state[column1] == state[column2]:
                number_attacking_pairs += 1
            # Get difference of column numbers
            diff = column2-column1
            # check diagonals
            if state[column1] == state[column2] - diff or state[column1] == state[column2] + diff:
                number_attacking_pairs += 1
    return number_attacking_pairs

'''
The basic hill-climbing algorithm for n queens
'''
def hill_desending_n_queens(state, comp_att_pairs):

    final_state = list(state)
    n = len(final_state)

    while 1:

        currAttacks = comp_att_pairs(final_state)
        if currAttacks == 0:
            return final_state

        # Define heuristic matrix
        h = np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                if j == final_state[i]:
                    h[i,j] = currAttacks
                else:
                    state_copy = list(final_state)
                    state_copy[i] = j
                    h[i,j] = comp_att_pairs(state_copy)

        # Get lowest heuristic state
        curr_lowest_loc = (0,0)
        curr_lowest_h = float('inf')
        for i in range(n):
            for j in range(n):
                if h[i,j] < curr_lowest_h:
                    curr_lowest_h = h[i,j]
                    curr_lowest_loc = (i,j)

        # If there are no lower heuristic neighbors states, return current state
        if currAttacks <= curr_lowest_h:
            return final_state
        final_state[curr_lowest_loc[0]] = curr_lowest_loc[1]

    return final_state

'''
Hill-climbing algorithm for n queens with restart
'''
def n_queens(n, get_rand_st, comp_att_pairs, hill_descending):

    final_state = hill_descending(get_rand_st(n),comp_att_pairs)
    while comp_att_pairs(final_state) != 0:
        final_state = hill_descending(get_rand_st(n), comp_att_pairs)

    return final_state






