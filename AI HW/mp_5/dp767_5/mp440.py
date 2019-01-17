import inspect
import sys
import numpy as np
import matplotlib.pyplot as plt

x2=0
y2=0
xk = [0, 0]
Q2 = [[10 ** -4, 2 * (10 ** -5)], [2 * (10 ** -5), 10 ** -4]]
R2 = [[10 ** -2, 5 * (10 ** -3)], [5 * (10 ** -3), 10 ** -2]]
l2 = 0.5
P2 = np.identity(2) * l2

'''
Raise a "not defined" exception as a reminder 
'''
def _raise_not_defined():
    print "Method not implemented: %s" % inspect.stack()[1][3]
    sys.exit(1)

'''
Kalman 2D
'''
def kalman2d(data):
    estimated = []
    xk = [0,0]
    Q = [[10**-4, 2*(10**-5)], [2*(10**-5), 10**-4]]
    R = [[10**-2, 5*(10**-3)], [5*(10**-3), 10**-2]]
    l = 10
    P = np.identity(2)*l
    for i in range(len(data)):
        uk = data[i][0:2]
        zk = data[i][2:4]

        # Time Update
        xkp1 = np.sum([xk,uk],axis=0)
        P = P + Q

        # Measurement Update
        K = np.divide(P,np.sum([P,R]))
        xkp1 = np.sum([xkp1, np.matmul(K, (np.subtract(zk,xkp1)))], axis=0)
        P = np.multiply(np.subtract(np.identity(2), K),P)

        xk = xkp1
        print xk
        estimated.append(xk.tolist())

    return estimated

'''
Plotting
'''
def plot(data, output):
    z_prev = [0,0]
    x_prev = [0,0]
    for i in range(len(data)):
        z = data[i][2:4]
        x = output[i]
        plt.plot([z_prev[0], z[0]], [z_prev[1], z[1]], 'b-o', MarkerSize=3)
        plt.plot([x_prev[0], x[0]], [x_prev[1], x[1]], 'r-o', MarkerSize=3)
        z_prev = z
        x_prev = x
    plt.grid()
    plt.title('Observed vs. Kalman Filtered System State')
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.legend(['Observed State', 'Filtered State'])
    plt.show()
    return

'''
Kalman 2D 
'''
def kalman2d_shoot(ux, uy, ox, oy, reset=False):
    global x2, y2, P2
    if reset == True:
        x2 = 0
        y2 = 0
        P2 = np.identity(2) * l2

    P2i = P2[0][0]

    uk = [ux,uy]
    zk = [ox,oy]
    xk = [x2,y2]

    # Time update
    xkp1 = np.sum([xk, uk], axis=0)
    P2 = P2 + Q2

    # Measurement Update
    K2 = np.divide(P2, np.sum([P2, R2]))

    mean_obs_diff = np.subtract(zk, xkp1)
    xkp1 = np.sum([xkp1, np.matmul(K2, (mean_obs_diff))], axis=0)

    P2 = np.multiply(np.subtract(np.identity(2), K2), P2)

    x2 = xkp1[0]
    y2 = xkp1[1]
    if np.linalg.norm(P2) < 0.01 and abs(P2i - P2[0][0]) < 0.0001:
        decision = (x2,y2,True)
    else:
        decision = (x2,y2, False)

    return decision

'''
Kalman 2D 
'''
def kalman2d_adv_shoot(ux, uy, ox, oy, reset=False):
    global x2, y2, P2, Q2, R2
    if reset == True:
        x2 = 0
        y2 = 0
        P2 = np.identity(2) * l2
        Q2 = [[10 ** -4, 2 * (10 ** -5)], [2 * (10 ** -5), 10 ** -4]]
        R2 = [[10 ** -2, 5 * (10 ** -3)], [5 * (10 ** -3), 10 ** -2]]


    P2i = P2[0][0]

    uk = [ux, uy]
    zk = [ox, oy]
    xk = [x2, y2]

    # Time update
    xkp1 = np.sum([xk, uk], axis=0)
    P2 = P2 + Q2

    # Measurement Update
    K2 = np.divide(P2, np.sum([P2, R2]))

    dk = np.subtract(zk, xkp1)
    xkp1 = np.sum([xkp1, np.matmul(K2, dk)], axis=0)

    P2 = np.multiply(np.subtract(np.identity(2), K2), P2)

    x2 = xkp1[0]
    y2 = xkp1[1]
    if np.linalg.norm(P2) < 0.01 and abs(P2i - P2[0][0]) < 0.0001:
        decision = (x2, y2, True)
    else:
        decision = (x2, y2, False)

    return decision


