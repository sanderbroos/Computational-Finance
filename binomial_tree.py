import math
import numpy as np
from scipy.stats import norm

class BinomialTree:
    def __init__(self, S, vol, T, N, r, K, option_type="call", pricing_type="eu"):
        self.S = S
        self.vol = vol
        self.T = T
        self.N = N
        self.r = r
        self.K = K

        self.dt = self.T/self.N
        self.u = math.exp(self.vol*math.sqrt(self.dt))
        self.d = math.exp(-self.vol*math.sqrt(self.dt))
        self.p = (math.exp(self.r*self.dt) - self.d) / (self.u - self.d)

        self.stock_tree = np.zeros((self.N+1, self.N+1))
        self.option_tree = np.zeros((self.N+1, self.N+1))

        self.option_type = option_type
        self.pricing_type = pricing_type

    def payoff(self, i, j):
        S = self.stock_tree[i, j] # value in the matrix

        if self.option_type == "put":
            return max(self.K - S, 0)
        else:
            return max(S - self.K, 0)

    def option_value(self, i, j):
        down = self.option_tree[i + 1, j]
        up = self.option_tree[i + 1, j + 1]

        if self.pricing_type == "eu":
            return math.exp(-self.r * self.dt) * (self.p * up + (1 - self.p) * down)
        else:
            return max(math.exp(-self.r * self.dt) * (self.p * up + (1 - self.p) * down), self.payoff(i, j))

    def build_stock_tree(self):
        # iterate over the lower triangle
        for i in np.arange(self.N + 1): # iterate over rows
            for j in np.arange(i + 1): # iterate over columns
                self.stock_tree[i, j] = self.S * self.u ** j * self.d ** (i - j)

        return self.stock_tree

    def build_option_tree(self):
        columns = self.option_tree.shape[1]
        rows = self.option_tree.shape[0]

        # walk backward, we start in last row of the matrix
        # add the payoff function in the last row
        for c in np.arange(columns):
            self.option_tree[rows - 1, c] = self.payoff(rows - 1, c)
        
        # for all other rows, we need to combine from previous rows
        # we walk backwards, from the last row to the first row
        for i in np.arange(rows - 1)[::-1]:
            for j in np.arange(i + 1):
                self.option_tree[i, j] = self.option_value(i, j)
        
        return self.option_tree

    def black_scholes_formula(self, tau=None):
        if tau == None:
            tau = self.T
            
        d1 = (math.log(self.S / self.K) + (self.r + 0.5 * math.pow(self.vol, 2)) * tau) / (self.vol * math.sqrt(tau))
        d2 = d1 - self.vol * math.sqrt(tau)

        return self.S * norm.cdf(d1) - math.exp(-self.r * tau) * self.K * norm.cdf(d2), norm.cdf(d1)

    def calc_delta(self):
        fu = self.option_tree[1][1]
        fd = self.option_tree[1][0]
        s0 = self.stock_tree[0][0]

        return (fu - fd) / (s0 * self.u - s0 * self.d)

print("                   Tree vs Black-Scholes  |         Tree vs Black-Scholes")
for sigma in np.arange(0.1, 1.1, .1):
    binomial_tree = BinomialTree(S   = 100, 
                                 vol = sigma, 
                                 T   = 1, 
                                 N   = 50, 
                                 r   = 0.06, 
                                 K   = 99)

    stock_tree = binomial_tree.build_stock_tree()
    option_tree = binomial_tree.build_option_tree()
    black_scholes_value = binomial_tree.black_scholes_formula()
    delta = binomial_tree.calc_delta()

    print(f"sigma = {sigma:.2f}: {option_tree[0][0]:>9.6f} vs {black_scholes_value[0]:>9.6f}      |     {delta:>8.6f} vs {black_scholes_value[1]:>8.6f}")