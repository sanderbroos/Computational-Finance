import numpy as np
from scipy.stats import gmean

from analytic import asian_option_value

class MonteCarloStock():
    """
    Simulates a stock through the monte carlo method.
    """

    def __init__(self, T=1, K=99, r=0.06, S=100, vol=0.2, option_type="put"):

        self.T = T
        self.K = K
        self.r = r
        self.S0 = S
        self.S = S
        self.vol = vol
        self.option_type = option_type

    def calc_stock_price(self, tau=None):

        if not tau:
            tau = self.T

        # Equation 2
        return self.S * np.exp((self.r - 0.5 * self.vol ** 2) * tau + self.vol * np.sqrt(tau) * np.random.normal(0, 1))

    def calc_payoff(self):

        if self.option_type == "put":
            return max(self.K - self.calc_stock_price(), 0)
        elif self.option_type == "call":
            return max(self.calc_stock_price() - self.K, 0)
        elif self.option_type == "digital":
            return 1 if self.calc_stock_price() > self.K else 0
        else:
            raise Exception(f"Error: invalid option type {self.option_type}")

class AsianMonteCarloStock(MonteCarloStock):


    def __init__(self, T=1, K=99, r=0.06, S=100, vol=0.2, option_type="put", N=100, mean_type="geometric"):

        super().__init__(T=1, K=99, r=0.06, S=100, vol=0.2, option_type="put")
        self.N = N
        self.mean_type = mean_type

    def calc_payoff(self):

        dt = self.T / self.N
        stock_values = [self.S0]
        for i in range(self.N):

            self.S = self.calc_stock_price(tau = dt)

            stock_values.append(self.S)

        if self.mean_type == "geometric":
            return max(0, gmean(stock_values) - self.K)
        elif self.mean_type == "arithmetic":
            return max(0, np.mean(stock_values) - self.K)
        else:
            raise Exception(f"Error: invalid mean type {mean_type}")


class MonteCarloStockManager():
    """
    Manages a M distinct stocks.
    """

    def __init__(self, M, T=1, K=99, r=0.06, S=100, vol=0.2, option_type="put", N=100, mean_type="geometric"):
        
        self.M = M
        self.T = T
        self.K = K
        self.r = r
        self.S0 = S
        self.vol = vol
        self.option_type = option_type
        self.N = N
        self.mean_type = mean_type

    def calc_option_price(self, epsilon=0):

        payoffs = []
        for i in range(self.M):

            if self.option_type == "asian":
                stock = AsianMonteCarloStock(T=self.T, K=self.K, r=self.r, S=self.S0 + epsilon, vol=self.vol, option_type=self.option_type, N=self.N, mean_type=self.mean_type)
            else:
                stock = MonteCarloStock(T=self.T, K=self.K, r=self.r, S=self.S0 + epsilon, vol=self.vol, option_type=self.option_type)

            payoffs.append(stock.calc_payoff())

        return np.exp(-self.r * self.T) * np.mean(payoffs)

    def calc_hedge_parameter(self, epsilon, fixed_seed=True):

        st0 = np.random.get_state()

        bumped_option_price = self.calc_option_price(epsilon=epsilon)
        
        if fixed_seed:
            np.random.set_state(st0)

        unbumped_option_price = self.calc_option_price()

        return (bumped_option_price - unbumped_option_price) / epsilon



def control_variate_technique_asian(beta, M, T=1, K=99, r=0.06, S=100, vol=0.2, N=100):

    analytic = asian_option_value(T=T, K=K, r=r, S=S, vol=vol, N=N)

    MC_geo = MonteCarloStockManager(M, T=T, K=K, r=r, S=S, vol=vol, N=N, option_type="asian", mean_type="geometric").calc_option_price()
    MC_ari = MonteCarloStockManager(M, T=T, K=K, r=r, S=S, vol=vol, N=N, option_type="asian", mean_type="arithmetic").calc_option_price()

    return MC_ari - beta * (MC_geo - analytic)

def main():

    # manager = MonteCarloStockManager(100000, option_type="asian")

    # print(manager.calc_option_price())

    print(control_variate_technique_asian(0.5, 100000))

if __name__ == "__main__":
    main()

