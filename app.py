import streamlit as st
import numpy as np
from scipy.stats import norm
from priceCalc import blackScholes, binomialTree, trinomialTree, monteCarloOption


def wide_space_default():
    st.set_page_config(layout="wide")

wide_space_default()

model = st.sidebar.selectbox(
    "Choose Option Pricing Model:",
    ("Black-Scholes","Monte Carlo Simulation","Binomial Tree")
)


if model == "Black-Scholes":
    # Streamlit UI
    st.title("Black-Scholes Option Pricing and Greeks")

    st.sidebar.header("Input Parameters")
    S = st.sidebar.number_input("Stock Price (S)", value=100.0)
    K = st.sidebar.number_input("Strike Price (K)", value=100.0)
    T = st.sidebar.number_input("Time to Maturity (T in years)", value=1.0)
    r = st.sidebar.number_input("Risk-free Rate (r)", value=0.05)
    sigma = st.sidebar.number_input("Volatility (σ)", value=0.2)
    
    # Checks if user wants heatmaps shown
    
    incremenets = 10
    
    col1, col2, col3 = st.columns(3)
    with col1:
        hm = st.checkbox("Show Heatmaps")
    with col2:
        pl = st.checkbox("Option P/L")
    with col3:
        gr = st.checkbox("Show Greeks")
     
    
    results = blackScholes(T,K,S, r, sigma)
    results.pricing()
#         for key, value in results.pricing().items():
#             st.write(f"**{key}:** {value:.4f}")


    st.markdown(f"""
        <div style="font-size:30px; color:green; font-weight:bold;">
            Call Value: ${results.callPrice:.2f}
        </div>
        """, unsafe_allow_html=True)

        # Big "Put Value" in orange
    st.markdown(f"""
        <div style="font-size:30px; color:DarkOrange; font-weight:bold;">
            Put Value: ${results.putPrice:.2f}
        </div>
        """, unsafe_allow_html=True)

    if gr:
        skip_keys = {"Call Price", "Put Price"}
        greek_items = [(key, value) for key, value in results.pricing().items() if key not in skip_keys]

        cols = st.columns(4) 

        for i, (key, value) in enumerate(greek_items):
            with cols[i % 4]:  
                st.write(f"**{key}:** {value:.4f}")
    
    
    if hm:
        spot_min = st.number_input('Min Spot Price', min_value=0.01, value=S*0.8, step=0.01)
        spot_max = st.number_input('Max Spot Price', min_value=0.01, value=S*1.2, step=0.01)
        vol_min = st.slider('Min Volatility for Heatmap', min_value=0.01, max_value=1.0, value=sigma*0.5, step=0.01)
        vol_max = st.slider('Max Volatility for Heatmap', min_value=0.01, max_value=1.0, value=sigma*1.5, step=0.01)
        increments = st.number_input('Heatmap Increments', min_value = 2, max_value=25, value = 10, step=1)  
        
        spot_range = np.linspace(spot_min, spot_max, increments)
        vol_range = np.linspace(vol_min, vol_max, increments)

        fig_call, fig_put = results.heatmap(vol_range, spot_range)

        col1, col2 = st.columns([0.5, 0.5])

        with col1:
            st.subheader("Call Price Heatmap")
            st.pyplot(fig_call)

        with col2:
            st.subheader("Put Price Heatmap")
            st.pyplot(fig_put)
            
    if pl:
        st.header("Payoff Graph Parameters")

        K = st.number_input("Strike Price", value=100.0)
        premium = st.number_input("Premium (Cost of Option)", value=0.0)
        S_min = st.number_input("Min Stock Price", value=50.0)
        S_max = st.number_input("Max Stock Price", value=150.0)

        S_range = np.linspace(S_min, S_max, 100)

        fig_call, fig_put = results.option_payoff(S_range, K, premium)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Call Option Payoff")
            st.pyplot(fig_call)

        with col2:
            st.subheader("Put Option Payoff")
            st.pyplot(fig_put)

elif model =="Monte Carlo Simulation":
    st.title("Monte Carlo Simulation")

    st.sidebar.header("Input Parameters")
    S = st.sidebar.number_input("Stock Price (S)", value=100.0)
    K = st.sidebar.number_input("Strike Price (K)", value=100.0)
    T = st.sidebar.number_input("Time to Maturity (T in years)", value=1.0)
    r = st.sidebar.number_input("Risk-free Rate (r)", value=0.05)
    sigma = st.sidebar.number_input("Volatility (σ)", value=0.2)
    N = st.sidebar.number_input("Simulation Number", value=10000)
    steps = st.sidebar.number_input("number of steps", value=100)
    
    mc = monteCarloOption(S,K,T,r,sigma,N,steps)
    
    mc.simulate()
    for key, value in mc.price_option().items():
            st.write(f"**{key}:** {value:.4f}")
    st.pyplot(mc.plot_paths())
    
        

            
elif model =="Binomial Tree":
    st.title("Binomial Tree Option Pricer")
    
    st.sidebar.header("Input Parameters")
    option_type = st.sidebar.selectbox("What type of option do you want to price?", ("call","put"))
    S = st.sidebar.number_input("Stock Price (S)", value=100.0)
    K = st.sidebar.number_input("Strike Price (K)", value=100.0)
    T = st.sidebar.number_input("Time to Maturity (T in years)", value=1.0)
    r = st.sidebar.number_input("Risk-free Rate (r)", value=0.05)
    sigma = st.sidebar.number_input("Volatility (σ)", value=0.2)
    steps = st.sidebar.number_input("Time Step Number (N)", value=100)
    


    results = binomialTree(T, K, S, r, sigma, steps, option_type)
    st.subheader("Results")
    
    price = results.pricing()
        
        

    st.write(f"Price: {price}")
    
    if steps > 10:
        st.write(f"\n To see price and option trees steps must be less than 10.")

    if steps <= 10:
        pt = st.checkbox("Show Price Tree")
        ot = st.checkbox("Show Option Tree")
        fig1 = results.plot_binomial_tree(results.stock_tree, title="Stock Price Tree")
        fig2 = results.plot_binomial_tree(results.option_tree, title="Option Price Tree")
        

        fig1 = results.plot_binomial_tree(results.stock_tree, title="Stock Price Tree")
        fig2 = results.plot_binomial_tree(results.option_tree, title="Option Price Tree")
        
        if pt:
            st.pyplot(fig1)
            
        if ot:
            st.pyplot(fig2)
    
# elif model =="Trinomial Tree":
#     st.title("Trinomial Tree Option Pricer")
    
#     st.sidebar.header("Input Parameters")
#     option_type = st.sidebar.selectbox("What type of option do you want to price?", ("call","put"))
#     S = st.sidebar.number_input("Stock Price (S)", value=100.0)
#     K = st.sidebar.number_input("Strike Price (K)", value=100.0)
#     T = st.sidebar.number_input("Time to Maturity (T in years)", value=1.0)
#     r = st.sidebar.number_input("Risk-free Rate (r)", value=0.05)
#     sigma = st.sidebar.number_input("Volatility (σ)", value=0.2)
#     steps = st.sidebar.number_input("Time Step Number (N)", value=100)
    

    
#     results = trinomialTree(T, K, S, r, sigma, steps, option_type).pricing()
#     st.subheader("Results")

#     st.write(f"Price: {results}")
    
