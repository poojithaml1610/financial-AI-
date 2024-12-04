import streamlit as st
import yfinance as yf
import pickle
import os
import pandas as pd
from datetime import date
from prophet import Prophet
from prophet.plot import plot_plotly
import numpy as np
from scipy.optimize import minimize
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objs as go


# Bank Account Functions
def load_bank_accounts():
    if os.path.exists('bank_accounts.pkl'):
        with open('bank_accounts.pkl', 'rb') as f:
            return pickle.load(f)
    return {}


def save_bank_accounts(accounts):
    with open('bank_accounts.pkl', 'wb') as f:
        pickle.dump(accounts, f)


def add_bank_account(accounts, name, balance):
    accounts[name] = balance
    save_bank_accounts(accounts)
    return f"Added bank account '{name}' with balance ${balance:.2f}."


def remove_bank_account(accounts, name):
    if name in accounts:
        del accounts[name]
        save_bank_accounts(accounts)
        return f"Removed bank account '{name}'."
    return f"Bank account '{name}' not found."


def show_bank_accounts(accounts):
    if accounts:
        return "\n".join([f"{name}: ${balance:.2f}" for name, balance in accounts.items()])
    return "No bank accounts found."


# Stock Portfolio Functions
def load_portfolio():
    if os.path.exists('portfolio.pkl'):
        with open('portfolio.pkl', 'rb') as f:
            return pickle.load(f)
    return {}


def save_portfolio(portfolio):
    with open('portfolio.pkl', 'wb') as f:
        pickle.dump(portfolio, f)


def add_stock(portfolio, ticker, amount):
    ticker = ticker.upper()
    portfolio[ticker] = portfolio.get(ticker, 0) + amount
    save_portfolio(portfolio)
    return f"Added {amount} shares of {ticker}."


def show_portfolio(portfolio):
    if portfolio:
        return "\n".join([f"{ticker}: {shares} shares" for ticker, shares in portfolio.items()])
    return "Your portfolio is empty."


# Portfolio Optimization Functions
def load_data_for_optimization(tickers, period="1y"):
    data = yf.download(tickers, period=period)["Close"]
    return data


def optimize_portfolio(mean_returns, cov_matrix, risk_free_rate=0.02):
    num_assets = len(mean_returns)

    # Negative Sharpe Ratio (to minimize)
    def negative_sharpe_ratio(weights):
        portfolio_return = np.dot(weights, mean_returns)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
        return -sharpe_ratio

    # Constraints
    constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}  # Sum of weights = 1
    bounds = tuple((0, 1) for _ in range(num_assets))  # Weights between 0 and 1
    initial_weights = num_assets * [1.0 / num_assets]  # Equal weights

    # Optimization
    result = minimize(
        negative_sharpe_ratio,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    return result


def display_allocation_chart(tickers, weights):
    fig, ax = plt.subplots()
    ax.pie(weights, labels=tickers, autopct="%1.1f%%", startangle=90)
    ax.axis("equal")  # Equal aspect ratio ensures the pie is drawn as a circle
    st.pyplot(fig)


# Stock Forecast Functions
def stock_forecast_app():
    st.title("Stock Forecast App")

    # Input for stock symbol
    user_input_stock = st.text_input("Enter the stock symbol (e.g., AAPL, TSLA):", "AAPL")
    n_years = st.slider("Years of prediction:", 1, 4)
    period = n_years * 365

    # Load data
    @st.cache_data
    def load_data(ticker):
        data = yf.download(ticker, start="2015-01-01", end=date.today().strftime("%Y-%m-%d"))
        data.reset_index(inplace=True)  # Reset index to make 'Date' a column
        return data

    try:
        data = load_data(user_input_stock)

        if data.empty:
            st.error(f"No data found for {user_input_stock}. Please check the stock symbol.")
            return

        # Display historical data
        st.subheader("Historical Data")
        st.write(data.tail())

        # Plot Historical Stock Data (Close Price)
        st.subheader(f"Historical Close Price for {user_input_stock}")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Close Price'))
        fig.update_layout(title=f"Historical Close Price for {user_input_stock}",
                          xaxis_title="Date", yaxis_title="Price (USD)")
        st.plotly_chart(fig)

        # Prepare data for Prophet
        df_train = data[['Date', 'Close']].rename(columns={"Date": "ds", "Close": "y"})
        
        # Ensure the 'ds' column is in datetime format and 'y' is numeric
        df_train['ds'] = pd.to_datetime(df_train['ds'])
        df_train['y'] = pd.to_numeric(df_train['y'], errors='coerce')
        df_train = df_train.dropna()  # Drop rows with missing or invalid data

        # Initialize and fit the Prophet model
        m = Prophet(yearly_seasonality=True)
        m.fit(df_train)

        # Create a future dataframe for the forecast
        future = m.make_future_dataframe(periods=period)
        forecast = m.predict(future)

        # Display forecast data
        st.subheader("Forecast Data")
        st.write(forecast.tail())

        # Plot the forecast using Plotly
        st.subheader(f"Forecast Plot for {n_years} Year(s)")
        fig1 = plot_plotly(m, forecast)
        st.plotly_chart(fig1)

        # Plot forecast components (Trend, Yearly, etc.)
        st.subheader("Forecast Components")
        fig2 = m.plot_components(forecast)
        st.pyplot(fig2)

        # Year-wise Variation of Predicted Prices (Line Plot)
        st.subheader("Year-Wise Variation of Predicted Prices")
        forecast['year'] = forecast['ds'].dt.year
        yearly_variation = forecast.groupby('year')['yhat'].mean()

        # Plotting the yearly variation
        plt.figure(figsize=(10, 6))
        plt.plot(yearly_variation.index, yearly_variation.values, marker='o', color='b', linestyle='--')
        plt.title(f"Year-wise Variation of Predicted Prices for {user_input_stock}")
        plt.xlabel("Year")
        plt.ylabel("Predicted Price (USD)")
        plt.grid(True)
        st.pyplot(plt)

    except Exception as e:
        st.error(f"An error occurred: {e}")
# Main Function
def main():
    st.title("Investment Portfolio Management")
    st.sidebar.title("Navigation")
    options = st.sidebar.radio("Select an option", ("Bank Accounts", "Stocks", "Portfolio Optimization", "Stock Forecast App"))

    if options == "Bank Accounts":
        st.header("Manage Bank Accounts")
        accounts = load_bank_accounts()
        action = st.selectbox("Action", ("Add Account", "Remove Account", "Show Accounts"))
        if action == "Add Account":
            name = st.text_input("Account Name")
            balance = st.number_input("Initial Balance", min_value=0.0)
            if st.button("Add Account"):
                st.write(add_bank_account(accounts, name, balance))
        elif action == "Remove Account":
            name = st.text_input("Account Name")
            if st.button("Remove Account"):
                st.write(remove_bank_account(accounts, name))
        elif action == "Show Accounts":
            st.text(show_bank_accounts(accounts))

    elif options == "Stocks":
        st.header("Manage Stock Portfolio")
        portfolio = load_portfolio()
        action = st.selectbox("Action", ("Add Stock", "Show Portfolio"))
        if action == "Add Stock":
            ticker = st.text_input("Stock Ticker")
            amount = st.number_input("Number of Shares", min_value=1)
            if st.button("Add Stock"):
                st.write(add_stock(portfolio, ticker, amount))
        elif action == "Show Portfolio":
            st.text(show_portfolio(portfolio))

    elif options == "Portfolio Optimization":
        st.header("Portfolio Optimization")

        # Input: Stock Tickers
        tickers_input = st.text_input("Enter stock tickers (comma-separated, e.g., AAPL, MSFT, TSLA):", "AAPL, MSFT, TSLA")
        tickers = [ticker.strip().upper() for ticker in tickers_input.split(",")]

        # Input: Risk-Free Rate
        risk_free_rate = st.number_input("Enter the risk-free rate (as a percentage, e.g., 2 for 2%):", 2.0) / 100.0

        # Load Historical Data
        st.write("Loading historical data...")
        data = load_data_for_optimization(tickers)

        if data.empty:
            st.error("Failed to load data. Please check the stock tickers.")
            return

        # Display historical data
        st.subheader("Historical Data")
        st.write(data.tail())

        # Calculate Returns and Covariance
        returns = data.pct_change().dropna()
        mean_returns = returns.mean()
        cov_matrix = returns.cov()

        # Perform Optimization
        st.subheader("Optimizing Portfolio...")
        result = optimize_portfolio(mean_returns.values, cov_matrix.values, risk_free_rate)
        optimized_weights = result.x

        # Display Optimal Allocation
        st.subheader("Optimal Portfolio Allocation")
        for ticker, weight in zip(tickers, optimized_weights):
            st.write(f"{ticker}: {weight * 100:.2f}%")

        # Display Allocation Chart
        st.subheader("Allocation Pie Chart")
        display_allocation_chart(tickers, optimized_weights)

        # Portfolio Metrics
        portfolio_return = np.dot(optimized_weights, mean_returns)
        portfolio_volatility = np.sqrt(np.dot(optimized_weights.T, np.dot(cov_matrix, optimized_weights)))
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility

        st.subheader("Optimized Portfolio Performance")
        st.write(f"Expected Annual Return: {portfolio_return * 100:.2f}%")
        st.write(f"Portfolio Risk (Volatility): {portfolio_volatility * 100:.2f}%")
        st.write(f"Sharpe Ratio: {sharpe_ratio:.2f}")

    elif options == "Stock Forecast App":
        stock_forecast_app()


if __name__ == "__main__":
    main()
