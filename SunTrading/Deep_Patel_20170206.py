import math
from bdateutil import isbday #$ pip install bdateutil ---OR---  $ easy_install bdateutil
from holidays import UnitedStates
from bdateutil import relativedelta

# Question 1
def fMaxDD(c):
    maxDD = 0.0
    DD = 0.0
    peak = c[0]
    for value in c:
        if value > peak:
            peak = value
        DD = (1.0*peak - 1.0*value) / (1.0*peak)
        if DD > maxDD:
            maxDD = DD
    return maxDD


# Question 2
def fBlackScholes(stockPrice, strike, rate, sigma, time, dividend):
    d1 = math.log(stockPrice/strike) + time*(rate-dividend+((sigma ** 2)/2))
    gamma = (math.exp(-dividend*time) / (stockPrice*sigma*math.sqrt(time))) * (1/math.sqrt(2*math.pi)) * math.exp(-(d1 ** 2)/2)

    return gamma


# Question 3
def fBirthdayProb(N):
    prob = 1
    notprob = 1
    if N > 365:
        return prob
    # calculates probability of 2 people not having the same birthday and then subtracting it from 1
    i = 1
    for i in range(N):
        notprob *= 1.0*(365-i)/365

    prob = 1.0 - notprob
    return prob


# Question 4
def fPreviousNBizDay(c, N):
    newDate = c + relativedelta(bdays=-N, holidays=UnitedStates())
    return newDate


# Question 5
def fNonLmR2(a, b):
    asum = sum(a)
    bsum = sum(b)
    top = 0
    n = len(a)
    sumab = 0
    asquare = 0
    bsquare = 0
    for i in range(n):
        sumab += a[i-1] * b[i-1]
        asquare += a[i-1] ** 2
        bsquare += b[i-1] ** 2
    asumbsum = asum * bsum
    r2 = (((1.0*n*sumab) - 1.0*asumbsum) ** 2.0) / ((1.0*n*asquare - (1.0*asum**2)) * (1.0*n*bsquare - (1.0*bsum**2)))

    return r2