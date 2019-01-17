'''
Compute the value brought by a given move by placing a new token for player
at (row, column). The value is the number of opponent pieces getting flipped
by the move.

A move is valid if for the player, the location specified by (row, column) is
(1) empty and (2) will cause some pieces from the other player to flip. The
return value for the function should be the number of pieces hat will be moved.
If the move is not valid, then the value 0 (zero) should be returned. Note
here that row and column both start with index 0.
'''

truncation = 0
terminal = 0

def get_move_value(state, player, row, column):
    flipped = 0
    n = len(state)

    count = 0
    currentRow = row
    currentColumn = column

    if row >= 2:  # NORTH
        while currentRow > 0:
            currentRow -= 1
            if state[currentRow][currentColumn] == ' ':
                break
            if state[currentRow][currentColumn] == player:
                flipped += count
                break
            count += 1

        count = 0
        currentRow = row
        currentColumn = column
        if column >= 2:  # NORTHWEST
            while currentRow > 0 and currentColumn > 0:
                currentRow -= 1
                currentColumn -= 1
                if state[currentRow][currentColumn] == ' ':
                    break
                if state[currentRow][currentColumn] == player:
                    flipped += count
                    break
                count += 1

        count = 0
        currentRow = row
        currentColumn = column
        if column <= n - 3:  # NORTHEAST
            while currentRow > 0 and currentColumn < n - 1:
                currentRow -= 1
                currentColumn += 1
                if state[currentRow][currentColumn] == ' ':
                    break
                if state[currentRow][currentColumn] == player:
                    flipped += count
                    break
                count += 1

    count = 0
    currentRow = row
    currentColumn = column
    if row <= n - 3:  # SOUTH
        while currentRow < n - 1:
            currentRow += 1
            if state[currentRow][currentColumn] == ' ':
                break
            if state[currentRow][currentColumn] == player:
                flipped += count
                break
            count += 1

        count = 0
        currentRow = row
        currentColumn = column
        if column >= 2:  # SOUTHWEST
            while currentRow < n - 1 and currentColumn > 0:
                currentRow += 1
                currentColumn -= 1
                if state[currentRow][currentColumn] == ' ':
                    break
                if state[currentRow][currentColumn] == player:
                    flipped += count
                    break
                count += 1

        count = 0
        currentRow = row
        currentColumn = column
        if column <= n - 3:  # SOUTHEAST
            while currentRow < n - 1 and currentColumn < n - 1:
                currentRow += 1
                currentColumn += 1
                if state[currentRow][currentColumn] == ' ':
                    break
                if state[currentRow][currentColumn] == player:
                    flipped += count
                    break
                count += 1

    count = 0
    currentRow = row
    currentColumn = column
    if column >= 2:  # WEST
        while currentColumn > 0:
            currentColumn -= 1
            if state[currentRow][currentColumn] == ' ':
                break
            if state[currentRow][currentColumn] == player:
                flipped += count
                break
            count += 1

    count = 0
    currentRow = row
    currentColumn = column
    if column <= n - 3:  # EAST
        while currentColumn < n - 1:
            currentColumn += 1
            if state[currentRow][currentColumn] == ' ':
                break
            if state[currentRow][currentColumn] == player:
                flipped += count
                break
            count += 1

    return flipped


'''
Execute a move that updates the state. A new state should be crated. The move
must be valid. Note that the new state should be a clone of the old state and
in particular, should not share memory with the old state.
'''


def execute_move(state, player, row, column):
    new_state = []
    n = len(state)
    for i in range(0, n):
        emptyRow = []
        for j in range(0, n):
            emptyRow.append(' ')
        new_state.append(emptyRow)

    for i in range(0, n):
        for j in range(0, n):
            new_state[i][j] = state[i][j]

    new_state[row][column] = player

    n = len(state)

    count = 0
    currentRow = row
    currentColumn = column

    if row >= 2:  # NORTH
        while currentRow > 0:
            currentRow -= 1
            if state[currentRow][currentColumn] == ' ':
                break
            if state[currentRow][currentColumn] == player:
                while count > 0:
                    currentRow += 1
                    new_state[currentRow][currentColumn] = player
                    count -= 1
                break
            count += 1

        count = 0
        currentRow = row
        currentColumn = column
        if column >= 2:  # NORTHWEST
            while currentRow > 0 and currentColumn > 0:
                currentRow -= 1
                currentColumn -= 1
                if state[currentRow][currentColumn] == ' ':
                    break
                if state[currentRow][currentColumn] == player:
                    while count > 0:
                        currentRow += 1
                        currentColumn += 1
                        new_state[currentRow][currentColumn] = player
                        count -= 1
                    break
                count += 1

        count = 0
        currentRow = row
        currentColumn = column
        if column <= n - 3:  # NORTHEAST
            while currentRow > 0 and currentColumn < n - 1:
                currentRow -= 1
                currentColumn += 1
                if state[currentRow][currentColumn] == ' ':
                    break
                if state[currentRow][currentColumn] == player:
                    while count > 0:
                        currentRow += 1
                        currentColumn -= 1
                        new_state[currentRow][currentColumn] = player
                        count -= 1
                    break
                count += 1

    count = 0
    currentRow = row
    currentColumn = column
    if row <= n - 3:  # SOUTH
        while currentRow < n - 1:
            currentRow += 1
            if state[currentRow][currentColumn] == ' ':
                break
            if state[currentRow][currentColumn] == player:
                while count > 0:
                    currentRow -= 1
                    new_state[currentRow][currentColumn] = player
                    count -= 1
                break
            count += 1

        count = 0
        currentRow = row
        currentColumn = column
        if column >= 2:  # SOUTHWEST
            while currentRow < n - 1 and currentColumn > 0:
                currentRow += 1
                currentColumn -= 1
                if state[currentRow][currentColumn] == ' ':
                    break
                if state[currentRow][currentColumn] == player:
                    while count > 0:
                        currentRow -= 1
                        currentColumn += 1
                        new_state[currentRow][currentColumn] = player
                        count -= 1
                    break
                count += 1

        count = 0
        currentRow = row
        currentColumn = column
        if column <= n - 3:  # SOUTHEAST
            while currentRow < n - 1 and currentColumn < n - 1:
                currentRow += 1
                currentColumn += 1
                if state[currentRow][currentColumn] == ' ':
                    break
                if state[currentRow][currentColumn] == player:
                    while count > 0:
                        currentRow -= 1
                        currentColumn -= 1
                        new_state[currentRow][currentColumn] = player
                        count -= 1
                    break
                count += 1

    count = 0
    currentRow = row
    currentColumn = column
    if column >= 2:  # WEST
        while currentColumn > 0:
            currentColumn -= 1
            if state[currentRow][currentColumn] == ' ':
                break
            if state[currentRow][currentColumn] == player:
                while count > 0:
                    currentColumn += 1
                    new_state[currentRow][currentColumn] = player
                    count -= 1
                break
            count += 1

    count = 0
    currentRow = row
    currentColumn = column
    if column <= n - 3:  # EAST
        while currentColumn < n - 1:
            currentColumn += 1
            if state[currentRow][currentColumn] == ' ':
                break
            if state[currentRow][currentColumn] == player:
                while count > 0:
                    currentColumn -= 1
                    new_state[currentRow][currentColumn] = player
                    count -= 1
                break
            count += 1

    return new_state


'''
A method for counting the pieces owned by the two players for a given state. The
return value should be two tuple in the format of (blackpeices, white pieces), e.g.,

    return (4, 3)

'''

def count_pieces(state):
    blackpieces = 0
    whitepieces = 0
    n = len(state)
    for i in range(0, n):
        for j in range(0, n):
            if state[i][j] == 'B':
                blackpieces += 1
            elif state[i][j] == 'W':
                whitepieces += 1

    return (blackpieces, whitepieces)


'''
Check whether a state is a terminal state.
'''

def is_terminal_state(state, state_list=None):
    terminal = True
    if state_list:
        for location in state_list:
            if get_move_value(state, 'B',location[0],location[1]) > 0 or get_move_value(state, 'W',location[0],location[1]) > 0:
                return False
    return terminal

'''
Get the other player's string
'''

def otherPlayer(player):
    if player == 'B':
        return 'W'
    return 'B'


'''
The minimax algorithm. Your implementation should return the best value for the
given state and player, as well as the next immediate move to take for the player.
'''

def _print_game_state(state):
    for i in range(0, len(state)):
        print state[i]

def minimax(state, player):
    global terminal
    print "Entering minimax, player", player
    print _print_game_state(state)
    value = 0
    row = -1
    column = -1
    n = len(state)
    state_list = []
    for i in range(0, n):
        for j in range(0, n):
            if state[i][j] == ' ':
                state_list.append((i,j))
    print "state list: ", state_list


    if is_terminal_state(state, state_list):
        terminal += 1
        values = count_pieces(state)
        print "values", values
        return (values[0] - values[1], -1, -1)

    if player == 'B':
        print "Entering B"
        tempValue = -10000000
        movePossible = False
        for location in state_list:
            currMoveVal = get_move_value(state, 'B', location[0], location[1])
            if currMoveVal > 0:
                movePossible = True
                state_list_copy = state_list[:]
                state_list_copy.remove(location)
                currVal = minValue(execute_move(state, 'B', location[0], location[1]), state_list_copy)
                print "currVal:", currVal, " for location:", location
                if currVal > tempValue:
                    value = currVal
                    tempValue = value
                    row = location[0]
                    column = location[1]
        if movePossible == False:
            print "movePossible:", movePossible
            for location in state_list:
                currMoveVal = get_move_value(state, 'W', location[0], location[1])
                print "No Moves currVal:", currMoveVal, " for location:", location
                if currMoveVal > 0:
                    state_list_copy = state_list[:]
                    state_list_copy.remove(location)
                    currVal = maxValue(execute_move(state, 'W', location[0], location[1]), state_list_copy)
                    if currVal < value:
                        value = currVal

    if player == 'W':
        print "Entering W"
        tempValue = 10000000
        movePossible = False
        for location in state_list:
            currMoveVal = get_move_value(state, 'W', location[0], location[1])
            if currMoveVal > 0:
                movePossible = True
                state_list_copy = state_list[:]
                state_list_copy.remove(location)
                currVal = maxValue(execute_move(state, 'W', location[0], location[1]), state_list_copy)
                print "currVal:", currVal, " for location:", location
                if currVal < tempValue:
                    value = currVal
                    tempValue = value
                    row = location[0]
                    column = location[1]
        if movePossible == False:
            print "movePossible:", movePossible
            for location in state_list:
                currMoveVal = get_move_value(state, 'B', location[0], location[1])
                print "No Moves currVal:", currMoveVal, " for location:", location
                if currMoveVal > 0:
                    state_list_copy = state_list[:]
                    state_list_copy.remove(location)
                    currVal = minValue(execute_move(state, 'B', location[0], location[1]), state_list_copy)
                    if currVal > value:
                        value = currVal
    print "Returning: ", (value, row, column)
    return (value, row, column)


def maxValue(state, state_list):
    global terminal
    # print "Entering maxValue"
    if is_terminal_state(state, state_list):
        terminal += 1
        values = count_pieces(state)
        return values[0] - values[1]
    v = -10000000
    for location in state_list:
        state_list_copy = state_list[:]
        state_list_copy.remove(location)
        if get_move_value(state, 'B', location[0], location[1]) > 0:
            v = max(v, minValue(execute_move(state,'B',location[0],location[1]), state_list_copy))
    if v == -10000000:
            v = minValue(state, state_list)
    return v

def minValue(state, state_list):
    global terminal
    # print "Entering minValue"
    if is_terminal_state(state, state_list):
        terminal += 1
        values = count_pieces(state)
        return values[0] - values[1]
    v = 10000000
    for location in state_list:
        state_list_copy = state_list[:]
        state_list_copy.remove(location)
        if get_move_value(state, 'W', location[0], location[1]) > 0:
            v = min(v, maxValue(execute_move(state,'W',location[0],location[1]), state_list_copy))
    if v == 10000000:
            v = maxValue(state, state_list)
    return v


'''
This method should call the minimax algorithm to compute an optimal move sequence
that leads to an end game.
'''


def full_minimax(state, player):
    global truncation
    print "full minimax, player", player
    value = 0
    move_sequence = []
    move = minimax(state, player)
    while 1:
        print move
        if move[1] == -1:
            value = move[0]
            player = otherPlayer(player)
            move = minimax(state, player)
            if move[1] == -1:
                move_sequence.append((otherPlayer(player), -1, -1))
                break
            else:
                value = move[0]
                move_sequence.append((player, move[1], move[2]))
        else:
            value = move[0]
            move_sequence.append((player, move[1], move[2]))
        state = execute_move(state,player,move[1],move[2])
        move = minimax(state, otherPlayer(player))
        player = otherPlayer(player)
    print (value, move_sequence)
    print "terminal", terminal
    return (value, move_sequence)


'''
The minimax algorithm with alpha-beta pruning. Your implementation should return the
best value for the given state and player, as well as the next immediate move to take
for the player.
'''


def minimax_ab(state, player, alpha=-10000000, beta=10000000):
    global truncation
    global terminal
    print "Entering alpha beta minimax, player", player
    print _print_game_state(state)
    value = 0
    row = -1
    column = -1
    n = len(state)
    state_list = []
    for i in range(0, n):
        for j in range(0, n):
            if state[i][j] == ' ':
                state_list.append((i, j))
    print "state list: ", state_list

    if is_terminal_state(state, state_list):
        terminal += 1
        values = count_pieces(state)
        print "values", values
        return (values[0] - values[1], -1, -1)

    if player == 'B':
        print "Entering B"
        tempValue = -10000000
        movePossible = False
        for location in state_list:
            currMoveVal = get_move_value(state, 'B', location[0], location[1])
            if currMoveVal > 0:
                movePossible = True
                state_list_copy = state_list[:]
                state_list_copy.remove(location)
                currVal = minValue_ab(execute_move(state, 'B', location[0], location[1]), alpha, beta, state_list_copy)
                if currVal >= beta:
                    truncation += 1
                    value = currVal
                    row = location[0]
                    column = location[1]
                    break
                alpha = max(alpha, currVal)
                print "currVal:", currVal, " for location:", location
                if currVal > tempValue:
                    value = currVal
                    tempValue = value
                    row = location[0]
                    column = location[1]
        if movePossible == False:
            print "movePossible:", movePossible
            for location in state_list:
                currMoveVal = get_move_value(state, 'W', location[0], location[1])
                print "No Moves currVal:", currMoveVal, " for location:", location
                if currMoveVal > 0:
                    state_list_copy = state_list[:]
                    state_list_copy.remove(location)
                    currVal = maxValue_ab(execute_move(state, 'W', location[0], location[1]),alpha, beta, state_list_copy)
                    if currVal < value:
                        value = currVal

    if player == 'W':
        print "Entering W"
        tempValue = 10000000
        movePossible = False
        for location in state_list:
            currMoveVal = get_move_value(state, 'W', location[0], location[1])
            if currMoveVal > 0:
                movePossible = True
                state_list_copy = state_list[:]
                state_list_copy.remove(location)
                currVal = maxValue_ab(execute_move(state, 'W', location[0], location[1]), alpha, beta, state_list_copy)
                if currVal <= alpha:
                    truncation += 1
                    value = currVal
                    row = location[0]
                    column = location[1]
                    break
                beta = min(beta, currVal)
                print "currVal:", currVal, " for location:", location
                if currVal < tempValue:
                    value = currVal
                    tempValue = value
                    row = location[0]
                    column = location[1]
        if movePossible == False:
            print "movePossible:", movePossible
            for location in state_list:
                currMoveVal = get_move_value(state, 'B', location[0], location[1])
                print "No Moves currVal:", currMoveVal, " for location:", location
                if currMoveVal > 0:
                    state_list_copy = state_list[:]
                    state_list_copy.remove(location)
                    currVal = minValue_ab(execute_move(state, 'B', location[0], location[1]), alpha, beta, state_list_copy)
                    if currVal > value:
                        value = currVal
    print "Returning: ", (value, row, column)
    return (value, row, column)


def maxValue_ab(state, alpha, beta, state_list):
    global truncation
    global terminal
    # print "Entering maxValue"
    if is_terminal_state(state, state_list):
        terminal += 1
        values = count_pieces(state)
        return values[0] - values[1]
    v = -10000000
    for location in state_list:
        state_list_copy = state_list[:]
        state_list_copy.remove(location)
        if get_move_value(state, 'B', location[0], location[1]) > 0:
            v = max(v, minValue_ab(execute_move(state,'B',location[0],location[1]), alpha, beta, state_list_copy))
            if v >= beta:
                truncation += 1
                return v
            alpha = max(v, alpha)
    if v == -10000000:
            v = minValue(state, state_list)
    return v

def minValue_ab(state, alpha, beta, state_list):
    global truncation
    global terminal
    # print "Entering minValue"
    if is_terminal_state(state, state_list):
        terminal += 1
        values = count_pieces(state)
        return values[0] - values[1]
    v = 10000000
    for location in state_list:
        state_list_copy = state_list[:]
        state_list_copy.remove(location)
        if get_move_value(state, 'W', location[0], location[1]) > 0:
            v = min(v, maxValue_ab(execute_move(state,'W',location[0],location[1]), alpha, beta, state_list_copy))
            if v <= alpha:
                truncation += 1
                return v
            beta = min(v, beta)
    if v == 10000000:
            v = maxValue(state, state_list)
    return v

'''
This method should call the minimax_ab algorithm to compute an optimal move sequence
that leads to an end game, using alpha-beta pruning.
'''


def full_minimax_ab(state, player):
    global truncation
    global terminal
    print "full minimax alpha beta, player", player
    value = 0
    move_sequence = []
    move = minimax_ab(state, player)
    while 1:
        print move
        if move[1] == -1:
            value = move[0]
            player = otherPlayer(player)
            move = minimax_ab(state, player)
            if move[1] == -1:
                terminal += 1
                move_sequence.append((otherPlayer(player), -1, -1))
                break
            else:
                value = move[0]
                move_sequence.append((player, move[1], move[2]))
        else:
            value = move[0]
            move_sequence.append((player, move[1], move[2]))
        state = execute_move(state,player,move[1],move[2])
        move = minimax_ab(state, otherPlayer(player))
        player = otherPlayer(player)
    print (value, move_sequence)
    print "truncation", truncation
    print "terminal", terminal
    return (value, move_sequence)
