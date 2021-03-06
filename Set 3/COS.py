import numpy as np
from scipy.stats import norm

class COS_euro_call:
    def __init__(self, S0, K, r, T, vol):
        self.S0 = S0
        self.K = K
        self.r = r
        self.T = T
        self.vol = vol

        self.a = np.log(S0/K) + r*T - 12*np.sqrt(vol**2 * T)
        self.b = np.log(S0/K) + r*T + 12*np.sqrt(vol**2 * T)
        self.x = np.log(S0/K)

    def phi(self, u, t):
        return np.exp(1j * u * (self.r - 0.5 * self.vol**2) * t - (0.5 * self.vol**2 * t * u**2))

    def chi(self, k, c, d):
        return (1/(1+(k*np.pi/(self.b-self.a))**2) * 
                (np.cos(k*np.pi*(d-self.a)/(self.b-self.a)) * np.exp(d)
                 - np.cos(k*np.pi*(c-self.a)/(self.b-self.a)) * np.exp(c)
                 + k*np.pi/(self.b-self.a) * np.sin(k*np.pi*(d-self.a)/(self.b-self.a)) * np.exp(d)
                 - k*np.pi/(self.b-self.a) * np.sin(k*np.pi*(c-self.a)/(self.b-self.a)) * np.exp(c)))

    def psi(self, k, c, d):
        if k == 0:
            return d - c
        
        return (np.sin(k * np.pi * (d - self.a)/(self.b - self.a)) - np.sin(k * np.pi * (c - self.a)/(self.b - self.a))) * (self.b - self.a)/(k*np.pi)

    def F(self, k, t):
        # TODO why x-a in the exponent instead of -a?
        return 2/(self.b - self.a) * (self.phi((k * np.pi)/(self.b - self.a), t) * np.exp(1j * ((k * (self.x - self.a) * np.pi)/(self.b - self.a)))).real

    def G(self, k):
        return 2/(self.b - self.a) * self.K * (self.chi(k, 0, self.b) - self.psi(k, 0, self.b))

    def V(self, N, t=None):
        if t == None:
            t = self.T

        fourier_sum = 0.5 * self.F(0, t) * self.G(0)

        # go *up to* N
        for k in range(1, N + 1):
            fourier_sum += self.F(k, t) * self.G(k)

        return np.exp(-self.r*t) * (self.b - self.a)/2 * fourier_sum

    def black_scholes(self, t=None):
        if t == None:
            t = self.T
            
        d1 = (np.log(self.S0 / self.K) + (self.r + 0.5 * self.vol**2) * t) / (self.vol * np.sqrt(t))
        d2 = d1 - self.vol * np.sqrt(t)

        return self.S0 * norm.cdf(d1) - np.exp(-self.r * t) * self.K * norm.cdf(d2)

if __name__ == "__main__":
    cos1 = COS_euro_call(S0=120, K=110, r=0.04, T=1, vol=0.3)
    cos2 = COS_euro_call(S0=110, K=110, r=0.04, T=1, vol=0.3)
    cos3 = COS_euro_call(S0=100, K=110, r=0.04, T=1, vol=0.3)
    print(cos1.V(64), cos1.black_scholes())
    print(cos2.V(64), cos2.black_scholes())
    print(cos3.V(64), cos3.black_scholes())