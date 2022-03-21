import numpy as np

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
        return 2/(self.b - self.a) * (self.phi((k * np.pi)/(self.b - self.a), t) * np.exp(1j * ((k * (self.x - self.a) * np.pi)/(self.b - self.a)))).real

    def G(self, k):
        return 2/(self.b - self.a) * self.K * (self.chi(k, 0, self.b) - self.psi(k, 0, self.b))

    def V(self, t, N):
        fourier_sum = 0.5 * self.F(0, t) * self.G(0)

        for k in range(1, N + 1):
            fourier_sum += self.F(k, t) * self.G(k)

        return np.exp(-self.r*t) * (self.b - self.a)/2 * fourier_sum

if __name__ == "__main__":
    cos = COS_euro_call(100, 99, 0.06, 1, 0.2)
    print(cos.V(cos.T, 64))