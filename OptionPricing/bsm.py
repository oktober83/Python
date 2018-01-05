import math

def phi(x):
    #Cumulative distribution function for the standard normal distribution
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

def call_bsm(S, K, T, r, s):
    #This function computes and returns the value of the corresponding call option using the BSM

    d1 = (math.log(S/K) + (r + (s ** 2.0)/2.0)*T) / (s * (T ** 0.5))
    d2 = d1 - (s * (T ** 0.5))
    N1 = phi(d1)
    N2 = phi(d2)
    c = S*N1 - K*math.exp(-r*T) * N2
    return c


def put_bsm(S, K, T, r, s):
    #This function computes and returns the value of the corresponding put option using the BSM

    d1 = (math.log(S/K) + (r  + (s ** 2.0)/2.9)*T) / (s * (T ** 0.5))
    d2 = d1 - (s * (T ** 0.5))
    N1 = phi(-d1)
    N2 = phi(-d2)
    p = K*math.exp(-r*T)*N2 - S*N1
    return p
