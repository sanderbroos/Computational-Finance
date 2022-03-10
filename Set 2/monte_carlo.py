import numpy as np
from scipy.stats import gmean
from scipy.stats import norm

from analytic import asian_option_value
from simulation_manager import SimulationManager

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

    def run(self):
        pass

    def calc_stock_price(self, tau=None, Z=None):

        if not tau:
            tau = self.T

        if not Z:
            Z = np.random.normal(0, 1)

        # Equation 2
        return self.S * np.exp((self.r - 0.5 * self.vol ** 2) * tau + self.vol * np.sqrt(tau) * Z)

    def calc_payoff(self, **kwargs):

        if self.option_type == "put":
            return max(self.K - self.calc_stock_price(), 0)
        elif self.option_type == "call":
            return max(self.calc_stock_price() - self.K, 0)
        elif self.option_type == "digital":
            return 1 if self.calc_stock_price(**kwargs) > self.K else 0
        else:
            raise Exception(f"Error: invalid option type {self.option_type}")

    def calc_likelihood_delta(self):

        factor = np.exp(- self.r * self.T)
        Z = np.random.normal(0, 1)
        payoff = self.calc_payoff(Z=Z)

        return factor * payoff * Z / (self.vol * self.S0 * np.sqrt(self.T))

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

    def __str__(self):
        return f"Manager: M={self.M}"
        
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

    def calc_pathwise_digital_delta(self, smoothing_scale, epsilon):
        results = []

        for _ in range(self.M):
            st0 = np.random.get_state()
            bumped_stock = MonteCarloStock(T=self.T, K=self.K, r=self.r, S=self.S0 + epsilon, vol=self.vol, option_type="digital")
            ST_bumped = bumped_stock.calc_stock_price()

            np.random.set_state(st0)
            unbumped_stock = MonteCarloStock(T=self.T, K=self.K, r=self.r, S=self.S0, vol=self.vol, option_type="digital")
            ST_unbumped = unbumped_stock.calc_stock_price()

            results.append(norm.pdf(ST_unbumped, self.K, smoothing_scale) * (ST_bumped - ST_unbumped) / epsilon)

        return np.mean(results)



def control_variate_technique_asian(M, T=1, K=99, r=0.06, S=100, vol=0.2, N=100):

    analytic = asian_option_value(T=T, K=K, r=r, S=S, vol=vol, N=N)

    st0 = np.random.get_state()

    MC_geo_payoffs, MC_geo_mean, MC_geo_std = MonteCarloStockManager(M, T=T, K=K, r=r, S=S, vol=vol, N=N, option_type="asian", mean_type="geometric").calc_option_price()
    
    np.random.set_state(st0)

    MC_ari_payoffs, MC_ari_mean, MC_ari_std = MonteCarloStockManager(M, T=T, K=K, r=r, S=S, vol=vol, N=N, option_type="asian", mean_type="arithmetic").calc_option_price()

    # Calculate optimal beta, rho is the correlation
    rho = np.cov(MC_ari_payoffs, MC_geo_payoffs) / (MC_geo_std * MC_ari_std)
    beta = (MC_ari_std / MC_geo_std) * rho

    return MC_ari_mean - beta * (MC_geo_mean - analytic), MC_ari_mean

def likelihood_ratio():

    n = 1000000
    sim_manager = SimulationManager(MonteCarloStock, n, option_type="digital") 
    mean, std = sim_manager.calc_attribute(lambda sim: sim.calc_likelihood_delta())

    print(mean, std)

    manager = MonteCarloStockManager(n)
    mean_bump, std_bump = manager.calc_hedge_parameter(0.1)

    print(mean_bump, std_bump)

def obtain_mean_and_stds_control_variate(n_instances, M, T=1, K=99, r=0.06, S=100, vol=0.2, N=100):

    control_variate_values, MC_ari_values = [], []
    for i in range(n_instances):

        control_variate, MC_ari = control_variate_technique_asian(M, T=T, K=K, r=r, S=S, vol=vol, N=N)

        control_variate_values.append(control_variate)
        MC_ari_values.append(MC_ari)

    control_variate_mean = np.mean(control_variate_values)
    control_variate_std = np.std(control_variate_values)

    MC_ari_mean = np.mean(MC_ari_values)
    MC_ari_std = np.std(MC_ari_values)

    return control_variate_mean, control_variate_std, MC_ari_mean, MC_ari_std

def main():

    manager = MonteCarloStockManager(50000, option_type="asian")
    print(manager.calc_pathwise_digital_delta(1, 0.01))

    # print(manager.calc_option_price())

    # values, aris = [], []
    # for i in range(10):

    #     value, ari = control_variate_technique_asian(0.5, 10000)

    #     values.append(value)
    #     aris.append(ari)

    # print(np.mean(values), np.std(values), np.mean(aris), np.std(aris))

    # likelihood_ratio()
    
if __name__ == "__main__":
    main()

