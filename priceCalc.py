import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import networkx as nx



class blackScholes():
    
    def __init__(self, T: float, K: float, S: float, r: float, sigma: float):
        self.T = T
        self.K = K
        self.S = S
        self.r = r
        self.sigma = sigma
        
        
    def pricing(self):
        
        
        d1 = (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)
        self.callPrice = self.S * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
        self.putPrice = self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.S * norm.cdf(-d1)
        call_price = self.callPrice
        put_price = self.putPrice
#         self.callPrice = call_price
#         self.putPrice = put_price
        
        
        # Greeks for Call
        delta_call = norm.cdf(d1)
       
        gamma = norm.pdf(d1) / (self.S * self.sigma * np.sqrt(self.T))
        vega = self.S * norm.pdf(d1) * np.sqrt(self.T)
        theta_call = (
            - (self.S * norm.pdf(d1) * self.sigma) / (2 * np.sqrt(self.T))
            - self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
        )
        
        # For puts
        delta_put = delta_call - 1
        theta_put = (
            - (self.S * norm.pdf(d1) * self.sigma) / (2 * np.sqrt(self.T))
            + self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-d2)
        )
        rho_call = self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(d2)
        rho_put = -self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-d2)

        
        return {
            'Call Price': call_price,
            'Put Price': put_price,
            'Delta (Call)': delta_call,
            'Delta (Put)': delta_put,
            'Gamma': gamma,
            'Vega': vega,
            'Theta (Call)': theta_call,
            'Theta (Put)': theta_put,
            'Rho (Call)': rho_call,
            'Rho (Put)': rho_put
            }
        
     
    def printing(self):
        print(self.callPrice, self.putPrice, rho_put)
        
        
    def heatmap(self, vol_range, spot_range):
        
        RdGn= LinearSegmentedColormap.from_list("RedGreen", ["#ffcccc", "#ffffff", "#ccffcc"])
        
        
        call_prices = np.zeros((len(vol_range), len(spot_range)))
        put_prices = np.zeros((len(vol_range), len(spot_range)))
        
        strike = self.K
        

        for i, vol in enumerate(vol_range):
            for j, spot in enumerate(spot_range):
                bs_temp = blackScholes(
                    T=self.T,
                    K=strike,
                    S=spot,
                    r=self.r,
                    sigma=vol
                )
                bs_temp.pricing()
                call_prices[i, j] = bs_temp.callPrice
                put_prices[i, j] = bs_temp.putPrice
         
        self.call_prices = call_prices

        # Plot Call Price Heatmap
        fig_call, ax_call = plt.subplots(figsize=(8, 8))
        sns.heatmap(
            call_prices,
            xticklabels=np.round(spot_range, 2),
            yticklabels=np.round(vol_range, 2),
            annot=True,
            fmt=".2f",
            cmap=RdGn,
            ax=ax_call
        )
        ax_call.set_title('Call Price Heatmap')
        ax_call.set_xlabel('Spot Price')
        ax_call.set_ylabel('Volatility')

        # Plot Put Price Heatmap
        fig_put, ax_put = plt.subplots(figsize=(8, 8))
        sns.heatmap(
            put_prices,
            xticklabels=np.round(spot_range, 2),
            yticklabels=np.round(vol_range, 2),
            annot=True,
            fmt=".2f",
            cmap=RdGn,
            ax=ax_put
        )
        ax_put.set_title('Put Price Heatmap')
        ax_put.set_xlabel('Spot Price')
        ax_put.set_ylabel('Volatility')

        return fig_call, fig_put
        
        
    def option_payoff(self, S_range, K, premium=0):
        payoff_call = np.maximum(S_range - K, 0) - premium
        payoff_put = np.maximum(K - S_range, 0) - premium   

        # Plot Call
        fig_call, ax1 = plt.subplots()
        ax1.plot(S_range, payoff_call, label="Call Option", color="green")
        ax1.axhline(0, color="black", linestyle="--")
        ax1.axvline(K, color="red", linestyle="--", label="Strike Price")
        ax1.set_xlabel("Stock Price at Expiration")
        ax1.set_ylabel("Profit / Loss")
        ax1.set_title("Call Option Payoff")
        ax1.legend()

        # Plot Put
        fig_put, ax2 = plt.subplots()
        ax2.plot(S_range, payoff_put, label="Put Option", color="orange")
        ax2.axhline(0, color="black", linestyle="--")
        ax2.axvline(K, color="red", linestyle="--", label="Strike Price")
        ax2.set_xlabel("Stock Price at Expiration")
        ax2.set_ylabel("Profit / Loss")
        ax2.set_title("Put Option Payoff")
        ax2.legend()

        return fig_call, fig_put

        
        
        
        
        
        
        
        
        
class binomialTree:
    def __init__(self, T: float, K: float, S: float, r: float, sigma: float, steps: int, option_type: str = "call"):
        self.T = T
        self.K = K
        self.S = S
        self.r = r
        self.sigma = sigma
        self.steps= steps
        self.option_type = option_type.lower()
        
        
    def pricing(self):
        
        dt = self.T / self.steps # Change in time
        u = np.exp(self.sigma * np.sqrt(dt)) # upwards change
        d = 1 / u # downwards change
        p = (np.exp(self.r * dt) - d) / (u - d) # risk neutral probability




        stock_tree = np.zeros((self.steps + 1, self.steps + 1))
        for i in range(self.steps + 1):
            for j in range(i + 1):
                stock_tree[j, i] = self.S * (u ** (i - j)) * (d ** j)


        option_tree = np.zeros_like(stock_tree)


        for j in range(self.steps + 1):
            if self.option_type == "call":
                option_tree[j, self.steps] = max(0, stock_tree[j, self.steps] - self.K)
            elif self.option_type == "put":
                option_tree[j, self.steps] = max(0, self.K - stock_tree[j, self.steps])

        # Backward induction
        for i in range(self.steps - 1, -1, -1):
            for j in range(i + 1):
                option_tree[j, i] = np.exp(-self.r * dt) * (
                    p * option_tree[j, i + 1] + (1 - p) * option_tree[j + 1, i + 1]
                )

        self.price = option_tree[0, 0]
        
        
        self.stock_tree = stock_tree
        self.option_tree = option_tree
        return self.price


    def plot_binomial_tree(self, tree, title="Tree", node_labels=True):
        steps = tree.shape[1] - 1
        fig, ax = plt.subplots(figsize=(1.2 * steps, steps))

        for i in range(steps + 1):
            for j in range(i + 1):
                x = i
                y = steps - j
                value = tree[j, i]
                ax.plot(x, y, 'o', color='black')
                if node_labels:
                    ax.text(x, y + 0.3, f"{value:.2f}", ha='center', fontsize=8)

        ax.set_title(title)
        ax.axis('off')
        return fig
    



class trinomialTree:
    def __init__(self, T: float, K: float, S: float, r: float, sigma: float, steps: int, option_type: str = "call"):
        self.T = T
        self.K = K
        self.S = S
        self.r = r
        self.sigma = sigma
        self.steps = steps
        self.option_type = option_type.lower()

    def pricing(self):
        dt = self.T / self.steps
        nu = self.r - 0.5 * self.sigma**2
        dx = self.sigma * np.sqrt(3 * dt)

        # Up, middle, and down factors
        u = np.exp(dx)
        d = 1 / u
        m = 1

        disc = np.exp(-self.r * dt)

        pu = 1/6 + (nu * np.sqrt(dt) / (2 * self.sigma * np.sqrt(3)))
        pm = 2/3
        pd = 1/6 - (nu * np.sqrt(dt) / (2 * self.sigma * np.sqrt(3)))
        

        grid_size = 2 * self.steps + 1
        stock_tree = np.zeros((grid_size, self.steps + 1))
        option_tree = np.zeros_like(stock_tree)

        for i in range(grid_size):
            level = i - self.steps
            stock_tree[i, self.steps] = self.S * (u ** level)

        # Fill terminal option prices
        for i in range(grid_size):
            if self.option_type == "call":
                option_tree[i, self.steps] = max(0, stock_tree[i, self.steps] - self.K)
            elif self.option_type == "put":
                option_tree[i, self.steps] = max(0, self.K - stock_tree[i, self.steps])

        # Backward induction
        for t in range(self.steps - 1, -1, -1):
            for i in range(t + 1, 2 * self.steps - t):
                option_tree[i, t] = disc * (
                    pu * option_tree[i - 1, t + 1] +
                    pm * option_tree[i, t + 1] +
                    pd * option_tree[i + 1, t + 1]
                )


                level = i - self.steps
                stock_tree[i, t] = self.S * np.exp(level * dx)

        self.price = option_tree[self.steps, 0]
        return self.price


class monteCarloOption:
    def __init__(self, S, K, T, r, sigma, N=10000, steps=100):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.N = N
        self.steps = steps

    def simulate(self):
        dt = self.T / self.steps
        prices = np.zeros((self.N, self.steps + 1))
        prices[:, 0] = self.S

        for t in range(1, self.steps + 1):
            z = np.random.standard_normal(self.N)
            prices[:, t] = prices[:, t - 1] * np.exp(
                (self.r - 0.5 * self.sigma ** 2) * dt + self.sigma * np.sqrt(dt) * z
            )

        self.simulated_paths = prices
        return prices

    def price_option(self):
        if not hasattr(self, 'simulated_paths'):
            self.simulate()

        final_prices = self.simulated_paths[:, -1]
        call_payoffs = np.maximum(final_prices - self.K, 0)
        put_payoffs = np.maximum(self.K - final_prices, 0)

        call_price = np.exp(-self.r * self.T) * np.mean(call_payoffs)
        put_price = np.exp(-self.r * self.T) * np.mean(put_payoffs)

        self.call_price = call_price
        self.put_price = put_price

        return {
            "Call Price": call_price,
            "Put Price": put_price
        }

    def plot_paths(self, num_paths=50):
        if not hasattr(self, 'simulated_paths'):
            self.simulate()

        plt.figure(figsize=(10, 6))
        for i in range(min(num_paths, self.N)):
            plt.plot(self.simulated_paths[i], lw=0.5, alpha=0.6)
        plt.title("Simulated Stock Price Paths")
        plt.xlabel("Time Steps")
        plt.ylabel("Stock Price")
        plt.grid(True)
        return plt

    
    
        