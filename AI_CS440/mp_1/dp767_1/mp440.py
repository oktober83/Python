
'''
GCD algorithm
'''
def gcd(a, b):
    x = None
    if b > a:
        temp = b
        b = a
        a = temp
    while x == None:
        if a == 0:
            x = b
        elif b == 0:
            x = a
        else:
            r = a%b
            a = b
            b = r
    return x

'''
Rectangles on a rubik's cube
'''
def rubiks(n):
    return ((n*(n+1)/2)**2)*6

'''
Guessing a number
'''
def guess_unlimited(n, is_this_it):

    for guess in range(1,n+1):
        if is_this_it(guess) == True:
            return guess
    return -1

'''
Guessing a number where you can only make two guesses that are larger
'''
def guess_limited(n, is_this_smaller):

    l = 1
    h = n
    larger_count = 0

    while larger_count < 1:
        guess = (l + h) / 2
        if l == h:
            return l
        if is_this_smaller(guess) == True:
            l = guess+1
        else:
            h = guess
            larger_count += 1
    for guess in range(l,h+1):
        if is_this_smaller(guess) == False:
            return guess

'''
Guessing a number, bonus problem
'''
def guess_limited_plus(n, is_this_smaller):
    l = 1
    h = n

    while 1:
        guess = (l + h) / 2
        if l == h:
            return l
        if is_this_smaller(guess) == True:
            l = guess + 1
        else:
            h = guess

        

