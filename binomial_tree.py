import math
import numpy as np
from scipy.stats import norm


def buildTree(S, vol, T, N):
    dt = T/N
    matrix = np.zeros((N+1, N+1))
    u = math.exp(vol*math.sqrt(dt))
    d = math.exp(-vol*math.sqrt(dt))
    #Iterateoverthelowertriangle

    for i in np.arange(N+1): # iterate over rows
        for j in np.arange(i+1): # iterate over columns

            #Hint:express each cell as acombination of up and down moves
            matrix[i,j] = S * u ** j * d ** (i - j)

    return matrix

def valueOptionMatrix(tree, T, r ,K, vol) :
    dt = T / N
    u = math.exp(vol*math.sqrt(dt))
    d = math.exp(-vol*math.sqrt(dt))
    p = (math.exp(r*dt) - d) / (u - d)

    columns = tree.shape[1]
    rows = tree.shape[0]
    # Walk backward , we start in last row of the matrix
    # Add the payoff function in the last row
    for c in np.arange(columns):
        S = tree[rows - 1, c] # value in the matrix
        tree[rows - 1, c] = max(S - K, 0)
    
    # For all other rows, we need to combine from previous rows
    # We walk backwards, from the last row to the first row
    for i in np.arange(rows - 1)[::-1]:
        for j in np.arange(i + 1):
            down = tree[i + 1, j]
            up = tree[i + 1, j + 1]
            tree[i, j] = math.exp(-r*dt) * (p * up + (1 - p) * down)
    
    return tree


def black_scholes_formula(S, sigma, tau, r, K):

    d1 = (math.log(S / K) + (r + 0.5 * math.pow(sigma, 2)) * tau ) / (sigma * math.sqrt(tau))
    d2 = d1 - sigma * math.sqrt(tau)

    return S * norm.cdf(d1) - math.exp(-r * tau) * K * norm.cdf(d2)

S = 100
T = 1
N = 1000

K = 99
r = 0.06

for sigma in np.arange(0.1, 1, .1):

    tree = buildTree(S,sigma,T,N)

    matrix = valueOptionMatrix(tree, T, r, K, sigma)

    black_scholes_value = black_scholes_formula(S, sigma, T, r, K)
    print(f"sigma = {sigma:.2f}: {matrix[0][0]} vs {black_scholes_value}")