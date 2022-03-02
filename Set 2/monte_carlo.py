import numpy as np


class MonteCarloStock():
    """
    Simulates a stock through the monte carlo method.
    """

    def __init__(self, T=1, K=99, r=0.06, S=100, vol=0.2, option_type="put", pricing_type="eu"):

        self.T = T
        self.K = K
        self.r = r
        self.S0 = S
        self.vol = vol
        self.option_type = option_type
        self.pricing_type = pricing_type

    def calc_stock_price(self):

        # Equation 2
        return self.S0 * np.exp((self.r - 0.5 * self.vol ** 2) * self.T + self.vol * np.sqrt(self.T) * np.random.normal(0, 1))

    def calc_payoff(self):

        if self.option_type == "put":
            return max(self.K - self.calc_stock_price(), 0)
        elif self.option_type == "call":
            return max(self.calc_stock_price() - self.K, 0)
        else:
            raise Exception(f"Error: invalid option type {self.option_type}")

class MonteCarloStockManager():
    """
    Manages a M distinct stocks.
    """

    def __init__(self, M, T=1, K=99, r=0.06, S=100, vol=0.2, option_type="put", pricing_type="eu"):
        
        self.M = M
        self.T = T
        self.K = K
        self.r = r
        self.S0 = S
        self.vol = vol
        self.option_type = option_type
        self.pricing_type = pricing_type

    def calc_option_price(self, epsilon=0):

        payoffs = []
        for i in range(self.M):
            stock = MonteCarloStock(T=self.T, K=self.K, r=self.r, S=self.S0 + epsilon, vol=self.vol, option_type=self.option_type, pricing_type=self.pricing_type)

            payoffs.append(stock.calc_payoff())

        return np.exp(-self.r * self.T) * np.mean(payoffs)

    def calc_hedge_parameter(self, epsilon, fixed_seed=True):

        st0 = np.random.get_state()

        bumped_option_price = self.calc_option_price(epsilon=epsilon)
        
        if fixed_seed:
            np.random.set_state(st0)

        unbumped_option_price = self.calc_option_price()

        return (bumped_option_price - unbumped_option_price) / epsilon


def main():

    manager = MonteCarloStockManager(1000000, option_type="call")

    print(manager.calc_hedge_parameter(epsilon=0.01, fixed_seed=True))

if __name__ == "__main__":
    main()
