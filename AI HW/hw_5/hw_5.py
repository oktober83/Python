import numpy as np
import math
import copy

gamma = 0.9

def policyEvaluate(state,policies,R):
    evaluated = [[0, 0, 0, 0, 0, 0], [0, -0.05, -0.05, -0.05, -0.05, 0], [0, -0.05, 0, -0.05, -1, 0],
                 [0, -0.05, 0, -0.05, 1, 0], [0, 0, 0, 0, 0, 0]]
    for i in range(1,4):
        for j in range(1,5):
            if (i==2 and j==2) or (i==2 and j==4) or (i==3 and j==4):
                continue
            move1 = 0
            move2 = 0
            if policies[i][j] == "up":
                if R[i][j+1] != None:
                    move1 = state[i][j+1]
                else:
                    move1 = state[i][j]
                if R[i-1][j] != None:
                    move2 = state[i-1][j]
                else:
                    move2 = state[i][j]
            elif policies[i][j] == "right":
                if R[i+1][j] != None:
                    move1 = state[i+1][j]
                else:
                    move1 = state[i][j]
                if state[i][j+1] != None:
                    move2 = state[i][j+1]
                else:
                    move2 = state[i][j]
            elif policies[i][j] == "left":
                if R[i-1][j] != None:
                    move1 = state[i-1][j]
                else:
                    move1 = state[i][j]
                if R[i][j-1] != None:
                    move2 = state[i][j-1]
                else:
                    move2 = state[i][j]
            elif policies[i][j] == "down":
                if R[i][j-1] != None:
                    move1 = state[i][j-1]
                else:
                    move1 = state[i][j]
                if R[i+1][j] != None:
                    move2 = state[i+1][j]
                else:
                    move2 = state[i][j]
            evaluated[i][j] = R[i][j] + gamma*(0.9*move1+0.1*move2)
    return evaluated

def policyImprove(state,policies,R):
    newPolicies = copy.deepcopy(policies)
    for i in range(1, 4):
        for j in range(1, 5):
            if (i == 2 and j == 2) or (i == 2 and j == 4) or (i == 3 and j == 4):
                continue
            # Get all action values
            if R[i][j+1] != None:
                right = state[i][j+1]
            else:
                right = state[i][j]
            if R[i-1][j] != None:
                down = state[i-1][j]
            else:
                down = state[i][j]
            if R[i+1][j] != None:
                up = state[i+1][j]
            else:
                up = state[i][j]
            if R[i][j-1] != None:
                left = state[i][j-1]
            else:
                left = state[i][j]
            upMove = 0.9*up + 0.1*left
            leftMove = 0.9*left + 0.1*down
            rightMove = 0.9 * right + 0.1 * up
            downMove = 0.9 * down + 0.1 * right
            moveVals = [upMove, downMove, leftMove, rightMove]
            maxVal = max(moveVals)
            maxMoveIndex = moveVals.index(maxVal)
            policyStrings = ["up","down","left","right"]
            maxMoveString = policyStrings[maxMoveIndex]
            currPol = policies[i][j]
            currPolIndex = policyStrings.index(currPol)

            if currPol == maxMoveString or moveVals[currPolIndex] == maxVal:
                continue
            else:
                newPolicies[i][j] = maxMoveString
    return newPolicies


def printState(state):
    print
    print "States"
    print "-----------------------------"
    for i in reversed(range(1,4)):
        for j in range(1,5):
            if (i==2 and j==2):
                print "   x  ",
            else:
                print '%+.3f' % state[i][j],
        print
    print "-----------------------------"


def printPolicy(policy):
    print
    print "Policies"
    print "-----------------------------"
    for i in reversed(range(1, 4)):
        for j in range(1, 5):
            if (i == 2 and j == 2):
                print "   x ",
            elif (i==2 and j==4):
                print "   -1",
            elif (i==3 and j==4):
                print "   1",
            else:
                print '%5s' % policy[i][j],
        print
    print "-----------------------------"

def checkConvergence(oldState, newState):
    maxDelta = -10000000
    for i in range(1, 4):
        for j in range(1, 5):
            if (i == 2 and j == 2) or (i == 2 and j == 4) or (i == 3 and j == 4):
                continue
            delta = abs(oldState[i][j] - newState[i][j])
            if delta > maxDelta:
                maxDelta = delta
    if maxDelta < 0.001:
        return True
    return False

if __name__ == "__main__":

    R = [[None,None,None,None,None,None], [None, -0.05, -0.05, -0.05, -0.05, None], [None, -0.05, None, -0.05, -1, None],
         [None, -0.05, -0.05, -0.05, 1, None], [None,None,None,None,None,None]]

    states = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],[0.0, 0.0, 0.0, 0.0, -1, 0.0],[0.0, 0.0, 0.0, 0.0, 1, 0.0],[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]

    policies = [[None,None,None,None,None,None], [None,"up","up","up","up",None], [None,"up",None,"up",None,None],[None,"up","up","up",None,None],[None,None,None,None,None,None]]

    oldState = states
    newState = None
    oldPolicy = policies
    newPolicy = None
    while 1:
        printState(oldState)
        printPolicy(oldPolicy)

        newState = policyEvaluate(oldState, oldPolicy, R)
        newPolicy = policyImprove(oldState, oldPolicy, R)
        printState(newState)
        printPolicy(newPolicy)

        if checkConvergence(oldState,newState):
            break
        oldState = newState
        oldPolicy = newPolicy

