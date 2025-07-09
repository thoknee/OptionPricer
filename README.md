#  Option Pricer

This is a streamlit project that goes through ways to price options. It includes the binomial tree, monte carlo simulations, and the black-scholes merton model as 3 different ways to price options. 

The deployed website is at: https://derivativepricing.streamlit.app



---

##  Binomial Tree Model Pricing

Binomial tree option pricing is the most simple form of pricing options. It assumes that any stock can go up and down a calculated amount with a calculated probability. 

When looking at binomial option calculation, if step count is less than 10 you can view the values of each jump as well as the option price associated with said jump. 

To compare alongside the monte carlo simulation and black scholes, let's keep a common input that we can compare prices at the end.

For this, we will have the following:

Spot price = $100
Strike price = $110
Time to expiry = 1 year
Risk free rate: 5%
Volatility: 0.2

With these inputs, we get the following binomial option pricings:

```
Call: $6.099
Put: $10.7344
```



##  Monte Carlo Simulations


Monte Carlo simulations are ver similar to a binomial tree we just use a random brownian motion to predict the price change in stock. From this we are also able to estimate the price of options over time. 

Using the same parameters as above we obtained the following:

```
Call Price: $6.0104
Put Price: $10.7616
```

This isn't very far off from our binomial tree and if you look at the streamlit yourself you can see a subset of all the random paths that we looked at to calculate the movement of the stock. Interestingly enough, if you run this a couple times you'll actually get slightly different answers everytime. This is because of the inherent randomness of the stock price path. 

##  Black Scholes Merton

Now we are going to look at the most common way of calculating the price of stocks. The Black Scholes Merton model is the closed solution to the partial differntial equation that describes prices evolution of derivatives. This gives us a closed form solution that with the right parameters allows us to calculate the fair value of options. 

In the app the Black Scholes feature comes with some other things that you can experiment with. First is a heatmap that will show you how the price of a put/call will change as the volatility and spot price change. Furthermore, you can look at a P/L graph for simple calls and puts as the stock price increases.



To test the black schole smodel using the same parameters that we have throughout, we get: 

```
Call Price: $6.04
Put price: $10.68
```


It is no conincedence that all of these are in a similar range. They all build off one another, some are just more accurate and easier to use than others.