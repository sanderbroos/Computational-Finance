import numpy as np

class FiniteDiff():

    def __init__(self,  T=1, K=110, r=0.04, S=100, vol=0.3,  Nt = 100, NX = 100, S_max = None):
        
        if S_max is None:
            S_max = 10**5 * S

        self.T = T
        self.K = K
        self.r = r
        self.vol = vol

        self.Nt = Nt
        self.NX = NX
        self.S_max = S_max

        self.X_max = self.S_to_X(self.S_max)
        self.X_values = np.linspace(0, self.X_max, NX)

        self.dt = self.T / (Nt - 1)
        self.dX = self.X_max / (NX - 1)

        self.payoff_grid = np.zeros((Nt, NX))

        self.initialize_payoff_grid()

        self.k = np.zeros(self.NX)
        self.k[-1] = S_max

        self.propagate_matrix = self.calc_propagate_matrix()
        self.propagate_field()

    def initialize_payoff_grid(self):
        
        # Calculate payoff at maturity
        for i, X in enumerate(self.X_values):      
            self.payoff_grid[0][i] = self.calc_payoff(self.X_to_S(X), self.K)
            
    def calc_propagate_matrix(self):

        alpha = (self.r - 0.5 * self.vol ** 2) * self.dt / (2 * self.dX)
        beta = self.vol ** 2 * self.dt / self.dX ** 2
        
        a_negative_1 = -alpha + 0.5 * beta
        a_zero = 1 - beta - self.r * self.dt
        a_positive_1 = alpha + 0.5 * beta

        propagate_matrix = np.zeros((self.NX, self.NX))
        for i in range(1, self.NX - 1):

            propagate_matrix[i][i - 1] = a_negative_1
            propagate_matrix[i][i] = a_zero
            propagate_matrix[i][i + 1] = a_positive_1
                
        return propagate_matrix


    def propagate_field(self):

        for i in range(1, self.Nt):
            self.payoff_grid[i] = np.dot(self.propagate_matrix, self.payoff_grid[i - 1]) + self.k

    def calc_payoff(self, S, K):
        return max(S - K, 0)

    def S_to_X(self, S):
        return np.log(S)

    def X_to_S(self, X):
        return np.exp(X)

    def get_payoff_for_S(self, S):

        # Find index of X in the X_values
        X = self.S_to_X(S)
        for i in range(len(self.X_values) - 1):
            if self.X_values[i] <= X < self.X_values[i + 1]:
                X_index = i
                break
                
        # Interpolate between X values
        temp = (X - self.X_values[X_index]) / (self.X_values[i + 1] - self.X_values[i])
        payoff = self.payoff_grid[-1][X_index] + temp * (self.payoff_grid[-1][X_index + 1] - self.payoff_grid[-1][X_index])

        # Without interpolation
        # payoff = self.payoff_grid[-1][X_index]
        
        return payoff


def main():
    
    # Nt and NX have to be of the same order of magnitude, otherwise the finite
    # difference equation is not stable
    field = FiniteDiff(Nt=2000, NX=2000)

    print(field.get_payoff_for_S(100)) # BS expectation: 9.62536
    print(field.get_payoff_for_S(110)) # BS expectation: 15.12859
    print(field.get_payoff_for_S(120)) # BS expectation: 21.78881


if __name__ == "__main__":
    main()