from encodings.rot_13 import rot13

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import norm
from datetime import date, datetime, timedelta, time
import matplotlib.pyplot as plt

class BlackScholes:

    def __init__(self, r, S, K, T, σ):
        self.r = r # Risk-free rate
        self.S = S # Spot price
        self.K = K # Strike price
        self.T = T # Time to maturity in years
        self.σ = σ # Volatility

    def return_params(self):
        r = self.r
        S = self.S
        K = self.K
        T = self.T
        σ = self.σ

        return r, S, K, T, σ

    def calculate_ds(self):

        r, S, K, T, σ = self.return_params()

        d1 = ((np.log(S / K) + (r + 0.5 * σ ** 2) * T)
              / (σ * np.sqrt(T)))

        d2 = d1 - σ * np.sqrt(T)

        return d1, d2

    def black_scholes(self):

        r, S, K, T, σ = self.return_params()
        d1, d2 = self.calculate_ds()

        # Calculate call price
        call_price = S * norm.cdf(d1) - (
            (K * np.exp(-r*T)) * norm.cdf(d2))

        # Calculate put price
        put_price = K * np.exp(-r*T) * norm.cdf(-d2) - (
            S * norm.cdf(-d1))

        return call_price, put_price

    def calculate_greeks(self):

        r, S, K, T, σ = self.return_params()
        d1, d2 = self.calculate_ds()

        # Calculate deltas
        call_delta = norm.cdf(d1)
        put_delta = -norm.cdf(-d1)

        # Calculate gamma (same for call and put)
        gamma = norm.pdf(d1) / (S * σ * np.sqrt(T))

        # Calculate thetas
        call_theta = - (S * norm.pdf(d1) * σ) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)
        put_theta = - (S * norm.pdf(d1) * σ) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)

        # Calculate vega (same for call and put)
        vega = S * np.sqrt(T) * norm.pdf(d1)

        # Calculate rhos
        call_rho = K * T * np.exp(-r * T) * norm.cdf(d2)
        put_rho = - K * T * np.exp(-r * T) * norm.cdf(-d2)

        # Return greeks
        return round(float(call_delta), 3), round(float(put_delta), 3), round(float(gamma), 3), round(float(call_theta/365), 3), round(float(put_theta/365), 3), round(float(vega/100), 3), round(float(call_rho/100), 3), round(float(put_rho/100), 3)

def main():
    st.set_page_config(page_title="📈Black-Scholes Calculator", layout="wide")

    with st.sidebar:
        st.title("📈Black-Scholes Calculator")
        linkedIn_url = "www.linkedin.com/in/b-cavanagh"
        st.markdown("Created by: <a href='{linkedIn_url}' target='_blank'>Ben Cavanagh</a>", unsafe_allow_html=True)

        spot_price = st.number_input("Spot Price", min_value=0.0, value=100.0)
        strike_price = st.number_input("Strike Price", min_value=0.0, value=100.0)
        expiration_date = st.date_input("Expiration Date", min_value=date.today(), value=date.today() + timedelta(days=90))
        volatility = st.slider("Volatility (%)", 0.0, 200.0, 20.0, 0.5) / 100
        risk_free_rate = st.slider("Risk-Free Rate (%)", 0.1, 25.0, 3.0, 0.1) / 100

        st.markdown("---")
        st.write("Sensitivity Heatmaps")

        min_spot = st.number_input("Min Spot Price", 0.0, value=90.0)
        max_spot = st.number_input("Max Spot Price", 0.0, value=110.0)
        min_volatility = st.slider("Min Volatility (%)", 0.0, 200.0, 10.0, 0.5) / 100
        max_volatiltiy = st.slider("Max Volatility (%)", 0.0, 200.0, 30.0, 0.5) / 100

        st.markdown("---")
        st.write("PnL Heatmaps")

        call_purchase_price = st.number_input("Call Purchase Price", min_value=0.0, value=5.0)
        put_purchase_price = st.number_input("Put Purchase Price", min_value=0.0, value=5.0)


        # Calculate time to expire in years
        if expiration_date == date.today(): # If the option expires today
            current_datetime = datetime.now()
            market_close = datetime.combine(date.today(), time(16,0)) # Market closes at 4:00pm

            if current_datetime > market_close:
                time_to_maturity = 0 # Option expired
            else:
                time_remaining_today = market_close - current_datetime
                time_to_maturity = time_remaining_today.total_seconds() / (60 * 60 * 24 * 365) # Time in years

        else:
            time_to_maturity = (expiration_date - date.today()).days / 365

    # Create BlackScholes object
    bs = BlackScholes(r=risk_free_rate, S=spot_price, K=strike_price, T=time_to_maturity, σ=volatility)

    # Calculate call and put prices
    call_price, put_price = bs.black_scholes()

    price_container = st.container(border=True)
    price_container.subheader("Price Summary", divider="gray")
    with price_container:

        col1, col2 = st.columns(2)

        with col1:
            st.metric(label="**Call Option Price**", value=f"${call_price:.2f}")

        with col2:
            st.metric(label="**Put Option Price**", value=f"${put_price:.2f}")


    # Greeks
    call_delta, put_delta, gamma, call_theta, put_theta, vega, call_rho, put_rho = bs.calculate_greeks()
    greek_container = st.container(border=True)
    greek_container.subheader("Greeks", divider="gray")

    # Display greeks
    with greek_container:

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Call Option Greeks**")
            st.markdown(f"""
                       | Delta  | Gamma  | Theta  | Vega   | Rho    |
                       |--------|--------|--------|--------|--------|
                       | {call_delta:>6} | {gamma:>6} | {call_theta:>6} | {vega:>6} | {call_rho:>6} |
                   """)

        with col2:
            st.write("**Put Option Greeks**")
            st.markdown(f"""
                        | Delta  | Gamma  | Theta  | Vega   | Rho    |
                        |--------|--------|--------|--------|--------|
                        | {put_delta:>6} | {gamma:>6} | {put_theta:>6} | {vega:>6} | {put_rho:>6} |
                    """)

    # Volatiltiy Heatmaps

    spot_prices = np.linspace(min_spot, max_spot, 10)
    volatilities = np.linspace(min_volatility, max_volatiltiy, 10)

    spot_prices_rounded = np.round(spot_prices, 0)
    volatilities_rounded = np.round(volatilities, 2)

    call_prices = []
    put_prices = []
    call_pnl = []
    put_pnl = []

    for spot in spot_prices:
        row_call_prices = []
        row_put_prices = []
        row_call_pnl = []
        row_put_pnl = []
        for vol in volatilities:
            bs = BlackScholes(r=risk_free_rate, S=spot, K=strike_price, T=time_to_maturity, σ=vol)
            call_price, put_price = bs.black_scholes()

            row_call_prices.append(call_price)
            row_put_prices.append(put_price)

            row_call_pnl.append(call_price - call_purchase_price)
            row_put_pnl.append(put_price - put_purchase_price)

        call_prices.append(row_call_prices)
        put_prices.append(row_put_prices)

        call_pnl.append(row_call_pnl)
        put_pnl.append(row_put_pnl)

    call_prices_df = pd.DataFrame(call_prices, columns=spot_prices, index=volatilities)
    put_prices_df = pd.DataFrame(put_prices, columns=spot_prices, index=volatilities)

    call_pnl_df = pd.DataFrame(call_pnl, columns=spot_prices, index=volatilities)
    put_pnl_df = pd.DataFrame(put_pnl, columns=spot_prices, index=volatilities)


    heatmap_container = st.container(border=True)
    heatmap_container.subheader("Sensitivity Analysis", divider="grey")
    with heatmap_container:
        col1, col2 = st.columns(2)

        with col1:
            fig, ax = plt.subplots(figsize=(10,8))
            sns.heatmap(call_prices_df, cmap="viridis", annot=True, fmt=".2f", xticklabels=spot_prices_rounded, yticklabels=volatilities_rounded)
            ax.set_xlabel("Spot Price")
            ax.set_ylabel("Volatility")
            ax.set_title("Call Price Heatmap")
            st.pyplot(fig)

        with col2:
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(put_prices_df, cmap="viridis", annot=True, fmt=".2f", xticklabels=spot_prices_rounded, yticklabels=volatilities_rounded)
            ax.set_xlabel("Spot Price")
            ax.set_ylabel("Volatility")
            ax.set_title("Put Price Heatmap")
            st.pyplot(fig)


    # PnL Heatmaps
    pnl_container = st.container(border=True)
    pnl_container.subheader("PnL Summary", divider="gray")
    with pnl_container:
        col1, col2 = st.columns(2)

        with col1:
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(call_pnl_df, cmap="RdYlGn", annot=True, fmt=".2f", xticklabels=spot_prices_rounded, yticklabels=volatilities_rounded)
            ax.set_xlabel("Spot Price")
            ax.set_ylabel("Volatility")
            ax.set_title("Call PnL Heatmap")
            st.pyplot(fig)

        with col2:
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(put_pnl_df, cmap="RdYlGn", annot=True, fmt=".2f", xticklabels=spot_prices_rounded, yticklabels=volatilities_rounded)
            ax.set_xlabel("Spot Price")
            ax.set_ylabel("Volatility")
            ax.set_title("Put PnL Heatmap")
            st.pyplot(fig)


if __name__ == "__main__":
    main()