from math import *
import bsm
import binomialTree

# Uncomment below and remove hard coding to iteratively type all values in command prompt.
# S = float(input("Enter the initial stock price: "));
# K = float(input("Enter the strike price: "));
# T = float(input("Enter the Time duration in years: "));
# n = float(input("Enter the number of binomial tree steps: "));
# r = float(input("Enter the risk-free rate as a decimal (not percent): "));
# s = float(input("Enter the volatility as a decimal (not percent): "));

# Or change the random values below so you do not need to enter the numbers each time you run
S = 50
K = 52
T = 0.5
n = 100
r = 0.05
s = 0.30

c_bsm = bsm.call_bsm(S, K, T, r, s)
p_bsm = bsm.put_bsm(S,K,T,r,s)

c_bin_euro = binomialTree.call_bin_euro(S, K, T, n, r, s)
p_bin_euro = binomialTree.put_bin_euro(S, K, T, n, r, s)

c_bin_amer = binomialTree.call_bin_amer(S, K, T, n, r, s)
p_bin_amer = binomialTree.put_bin_amer(S, K, T, n, r, s)

print("");
print("Option Style | Pricing Method | Call Option Value | Put Option Value ");
print("-------------|----------------|-------------------|-------------------");
print("  European   | Black-Scholes  | ",c_bsm,"|", p_bsm);
print("-------------|----------------|-------------------|-------------------");
print("  European   |    Binomial    | ", c_bin_euro,"|", p_bin_euro);
print("-------------|----------------|-------------------|-------------------");
print("  American   |    Binomial    | ", c_bin_amer, "|", p_bin_amer);
