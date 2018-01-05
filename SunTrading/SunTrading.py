import math
from bdateutil import isbday #$ pip install bdateutil ---OR---  $ easy_install bdateutil
from holidays import UnitedStates
from bdateutil import relativedelta

def fMaxDD(c):
    maxDD = 0
    peak = c[0]
    for value in c:
        if value > peak:
            peak = value
        DD = (peak - value) / peak
        if DD > maxDD:
            maxDD = DD
    return maxDD


def fBlackScholes(stockPrice=100, strike=100, rate=.01, sigma=.15, time=.3, dividend=0):
    d1 = ln(stockPrice/strike) + time*(rate-dividend+((sigma ** 2)/2))
    gamma = (exp(-dividend*time) / (stockPrice*sigma*sqrt(t))) * (1/sqrt(2*pi)) * exp(-(di ** 2)/2)

    return gamma




def fBirthdayProb(N):
    prob = 1
    notprob = 1
    if N > 365:
        return prob
    # calculates probability of 2 people not having the same birthday and then subtracting it from 1
    i = 1
    while 1 < N:
        notprob *= (365-i)/365
        i = i+1

    prob = 1 - notprob
    return prob



def fPreviousNBizDay(c, N):
    newDate = c + relativedelta(bdays=-N, holidays=UnitedStates())
    return newDate

c = fPreviousNBizDay("2016-10-31", 100)
print c



def fNonLmR2(a, b):
    asum = sum(a)
    bsum = sum(b)
    top = 0
    n = len(a)
    sumab = 0
    asquare = 0
    bsquare = 0
    for i in a:
        sumab += a[i-1] * b[i-1]
        asquare += a[i-1] * a[i-1]
        bsquare += b[i-1] ** 2
        print asquare
    asumbsum = asum * bsum
    print asum
    print bsum
    print sumab
    print asquare
    r2 = (((n*sumab) - asumbsum) ** 2)/ (n*asquare - (asum**2)) * (n*bsquare - (bsum**2))

    return r2

a = (1, 3, 4, 6, 2, 4)
b = (4, 5, 7, 2, 5, 9)

print fNonLmR2(a,b)
