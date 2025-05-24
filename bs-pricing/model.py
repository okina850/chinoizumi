import numpy as np
from scipy.stats import norm
import math

class BlackScholesModel:
    """
    A class for pricing European call and put options using the Black-Scholes model.
    """
    def __init__(self, S, K, T, r, sigma):
        """
        Initializes the BlackScholesModel with the necessary parameters.

        Args:
            S (float): Current price of the underlying asset. つまりS_t
            K (float): Strike price of the option.
            T (float): Time to expiration in years.
            r (float): Risk-free interest rate.
            sigma (float): Volatility of the underlying asset.
        """
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma 

    def increment(self):
        newS = 0
        self.S = newS
    def _d1(self):
        """Calculates the d1 term in the Black-Scholes formula."""
        return (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))

    def _d2(self):
        """Calculates the d2 term in the Black-Scholes formula."""
        return self._d1() - self.sigma * np.sqrt(self.T)

    def call_price(self):
        """Calculates the price of a European call option."""
        d1 = self._d1()
        d2 = self._d2()
        return self.S * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)

    def put_price(self):
        """Calculates the price of a European put option."""
        d1 = self._d1()
        d2 = self._d2()
        return self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.S * norm.cdf(-d1)

if __name__ == '__main__':
# Example usage:
    S0 = 100.0  # Current stock price
    K = 100.0   # Strike price
    T = 1.0     # Time to expiration (1 year)
    r = 0.05    # Risk-free interest rate
    sigma = 0.2 # Volatility

    
    bs_model = BlackScholesModel(S0, K, T, r, sigma)
    C0 = 0
    C1 = 0
    

    call_price = bs_model.call_price()
    put_price = bs_model.put_price()

    print(f"Black-Scholes Call Option Price (OOP): {call_price:.4f}")
    print(f"Black-Scholes Put Option Price (OOP): {put_price:.4f}")



"""
Inthis version:

We define a class BlackScholesModel that encapsulates all the parameters and methods related to Black-Scholes pricing.
The __init__ method initializes the instance of the class with the underlying asset price (S), strike price (K), time to expiration (T), risk-free rate (r), and volatility (sigma).
The calculation of d1 and d2 are now private methods (_d1 and _d2) within the class, as they are intermediate values used by the pricing formulas. The underscore prefix is a convention in Python to indicate that these methods are intended for internal use within the class.
The call_price and put_price methods calculate and return the respective option prices using the parameters stored in the class instance and the internal _d1 and _d2 methods.
The example usage in the if __name__ == '__main__': block now creates an instance of the BlackScholesModel and then calls its methods to get the option prices.
"""