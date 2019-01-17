import sys
import numpy as np
import mp440 as mp

global the_number

guess_count = 0
larger_count = 0
the_number = 100

def _is_this_it(candidate):
    return (candidate == the_number)

def _is_this_smaller(candidate):
    global guess_count
    global larger_count 
    guess_count = guess_count + 1
    if candidate >= the_number:
        larger_count = larger_count + 1
    return (candidate < the_number)

if __name__ == "__main__":
    

    # GCD
    print mp.gcd(27,27)

    # Rubik's cube
    print mp.rubiks(3)

    # Guessing a number
    n = 1000
    the_number = n/2
    print mp.guess_unlimited(n,_is_this_it)

    # Guessing a number, with busting
    print mp.guess_limited(n,_is_this_smaller)
    print "Number of guesses: " + str(guess_count), ", Number of time busted: " + str(larger_count)

    # Guessing number competition
    print mp.guess_limited_plus(n,_is_this_smaller)
    print "Total score (lower is better): " + str(guess_count + larger_count)

