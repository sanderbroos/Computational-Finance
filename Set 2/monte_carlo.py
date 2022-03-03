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

        super().__init__(T=T, K=K, r=r, S=S, vol=vol, option_type=option_type)
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

        payoffs = np.array(payoffs)
        factor = np.exp(-self.r * self.T)

        return factor * payoffs, factor * np.mean(payoffs), factor * np.std(payoffs)

    def calc_hedge_parameter(self, epsilon, fixed_seed=True):

        st0 = np.random.get_state()

        _, bumped_option_price, bumped_option_std = self.calc_option_price(epsilon=epsilon)
        
        if fixed_seed:
            np.random.set_state(st0)

        _, unbumped_option_price, unbumped_option_std = self.calc_option_price()

        return (bumped_option_price - unbumped_option_price) / epsilon, (bumped_option_std + unbumped_option_std) / epsilon



def control_variate_technique_asian(beta, M, T=1, K=99, r=0.06, S=100, vol=0.2, N=100):

    analytic = asian_option_value(T=T, K=K, r=r, S=S, vol=vol, N=N)

    st0 = np.random.get_state()

    MC_geo_payoffs, MC_geo_mean, MC_geo_std = MonteCarloStockManager(M, T=T, K=K, r=r, S=S, vol=vol, N=N, option_type="asian", mean_type="geometric").calc_option_price()
    
    np.random.set_state(st0)

    MC_ari_payoffs, MC_ari_mean, MC_ari_std = MonteCarloStockManager(M, T=T, K=K, r=r, S=S, vol=vol, N=N, option_type="asian", mean_type="arithmetic").calc_option_price()

    # Calculate optimal beta, rho is the correlation
    rho = np.cov(MC_ari_payoffs, MC_geo_payoffs) / (MC_geo_std * MC_ari_std)
    beta = (MC_ari_std / MC_geo_std) * rho

    return MC_ari_mean - beta * (MC_geo_mean - analytic), MC_ari_mean

def main():

    # manager = MonteCarloStockManager(100000, option_type="asian")

    # print(manager.calc_option_price())

    values, aris = [], []
    for i in range(10):

        value, ari = control_variate_technique_asian(0.5, 10000)

        values.append(value)
        aris.append(ari)

    print(np.mean(values), np.std(values), np.mean(aris), np.std(aris))

    
if __name__ == "__main__":
    main()

