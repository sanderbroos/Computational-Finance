import numpy as np
from scipy.stats import norm


def asian_option_value(T=1, K=99, r=0.06, S=100, vol=0.2, N=100):

    vol_flubber = vol * np.sqrt((2 * N + 1) / (6 * (N + 1)))
    r_flubber = 0.5 * (r - 0.5 * vol ** 2 + vol_flubber ** 2)

    d1_flubber = (np.log(S / K) + (r_flubber + 0.5 * vol_flubber ** 2) * T) / (vol_flubber * np.sqrt(T))
    d2_flubber = (np.log(S / K) + (r_flubber - 0.5 * vol_flubber ** 2) * T) / (vol_flubber * np.sqrt(T))

    return np.exp(-r * T) * (S * np.exp(r_flubber * T) * norm.cdf(d1_flubber) - K * norm.cdf(d2_flubber))

if __name__ == "__main__":
    print(asian_option_value())