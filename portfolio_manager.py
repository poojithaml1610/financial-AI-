import streamlit as st
import pickle
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import seaborn as sns
from datetime import date
import os
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objects as go

# ---- Bank Functions ----
def load_bank_accounts():
    try:
        if os.path.exists('bank_accounts.pkl'):
            with open('bank_accounts.pkl', 'rb') as f:
                return pickle.load(f)
        return {}
    except (FileNotFoundError, EOFError, pickle.UnpicklingError) as e:
        st.warning(f"Error loading bank accounts: {e}")
        return {}

bank_accounts = load_bank_accounts()

def save_bank_accounts():
    try:
        with open('bank_accounts.pkl', 'wb') as f:
            pickle.dump(bank_accounts, f)
    except Exception as e:
        st.error(f"Error saving bank accounts: {e}")

def add_bank_account(account_name, balance):
    bank_accounts[account_name] = balance
    save_bank_accounts()
    return f"Added bank account '{account_name}' with balance ${balance:.2f}."

def remove_bank_account(account_name):
    if account_name in bank_accounts:
        del bank_accounts[account_name]
        save_bank_accounts()
        return f"Removed bank account '{account_name}'."
    else:
        return f"Bank account '{account_name}' not found."

def update_bank_account(account_name, balance):
    if account_name in bank_accounts:
        bank_accounts[account_name] = balance
        save_bank_accounts()
        return f"Updated bank account '{account_name}' to balance ${balance:.2f}."
    else:
        return f"Bank account '{account_name}' not found."

def show_bank_accounts():
    if bank_accounts:
        response = "Your bank accounts:\n"
        for account_name, balance in bank_accounts.items():
            response += f"{account_name}: ${balance:.2f}\n"
        return response.strip()
    else:
        return "No bank accounts found."

def total_balance():
    return f"Total balance across all accounts: ${sum(bank_accounts.values()):.2f}"

# ---- Stock Functions ----
def load_portfolio():
    try:
        if os.path.exists('portfolio.pkl'):
            with open('portfolio.pkl', 'rb') as f:
                return pickle.load(f)
        return {}
    except (FileNotFoundError, EOFError, pickle.UnpicklingError) as e:
        st.warning(f"Error loading portfolio: {e}")
        return {}

portfolio = load_portfolio()

def save_portfolio():
    try:
        with open('portfolio.pkl', 'wb') as f:
            pickle.dump(portfolio, f)
    except Exception as e:
        st.error(f"Error saving portfolio: {e}")

def add_stock(ticker, amount):
    ticker = ticker.upper()
    if ticker in portfolio:
        portfolio[ticker] += amount
    else:
        portfolio[ticker] = amount
    save_portfolio()
    return f"Added {amount} shares of {ticker} to your portfolio."

def remove_stock(ticker, amount):
    ticker = ticker.upper()
    if ticker in portfolio:
        if portfolio[ticker] >= amount:
            portfolio[ticker] -= amount
            if portfolio[ticker] == 0:
                del portfolio[ticker]
            save_portfolio()
            return f"Removed {amount} shares of {ticker} from your portfolio."
        else:
            return "You don't have that many shares."
    return f"No shares of {ticker} found in your portfolio."

def show_portfolio():
    if portfolio:
        response = "Your current portfolio:\n"
        for ticker, shares in portfolio.items():
            response += f"{ticker}: {shares} shares\n"
        return response.strip()
    else:
        return "Your portfolio is empty."

def portfolio_worth():
    total_value = 0
    response = ""
    for ticker, shares in portfolio.items():
        try:
            stock = yf.Ticker(ticker)
            price = stock.history(period="1d")['Close'].iloc[-1]
            total_value += shares * price
            response += f"{ticker}: {shares} shares at ${price:.2f} each, total ${shares * price:.2f}\n"
        except IndexError:
            response += f"Could not retrieve data for {ticker}. Data might be unavailable.\n"
        except Exception as e:
            response += f"Error retrieving data for {ticker}: {e}\n"
    response += f"Your portfolio is worth: ${total_value:.2f} USD"
    return response

# ---- Stock Prediction Chart Function ----
@st.cache_data(ttl=3600)
def stock_prediction_chart(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="1mo")
    if not hist.empty:
        last_price = hist['Close'].iloc[-1]
        st.line_chart(hist['Close'])
        st.write(f"The last price of {ticker} is ${last_price:.2f}.")
    else:
        st.write(f"No historical data available for {ticker}.")

# ---- Stock Forecast App ----
def stock_forecast_app():
    START = "2015-01-01"
    TODAY = date.today().strftime("%Y-%m-%d")
    user_input_stock = st.text_input('Enter the stock symbol (e.g. AAPL, TSLA, GOOGL):', 'AAPL')
    n_years = st.slider('Years of prediction:', 1, 4)
    period = n_years * 365

    @st.cache_data
    def load_data(ticker):
        data = yf.download(ticker, START, TODAY)
        data.reset_index(inplace=True)
        return data

    data_load_state = st.text('Loading data...')
    data = load_data(user_input_stock)
    data_load_state.text('Loading data... done!')
    st.subheader('Raw data')
    st.write(data.tail())

    if not data.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Stock Close"))
        fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)

        df_train = data[['Date', 'Close']].rename(columns={"Date": "ds", "Close": "y"})
        m = Prophet()
        m.fit(df_train)
        future = m.make_future_dataframe(periods=period)
        forecast = m.predict(future)

        st.subheader('Forecast data')
        st.write(forecast.tail())
        st.write(f'Forecast plot for {n_years} years')
        fig1 = plot_plotly(m, forecast)
        st.plotly_chart(fig1)

# ---- Main Streamlit Application ----
def main():
    st.title("Investment Portfolio Management")
    st.sidebar.title("Navigation")
    options = st.sidebar.radio("Select an option", ("Bank Accounts", "Stocks", "Portfolio Optimization", "Stock Forecast App"))

    if options == "Bank Accounts":
        st.header("Manage Bank Accounts")
        action = st.selectbox("Action", ("Add Account", "Remove Account", "Update Account", "Show Accounts"))
        # Add logic for managing accounts

    elif options == "Stocks":
        st.header("Manage Stock Portfolio")
        action = st.selectbox("Action", ("Add Stock", "Remove Stock", "Show Portfolio", "View Stock Prediction"))
        # Add logic for managing stocks

    elif options == "Portfolio Optimization":
        st.header("Portfolio Optimization Using Modern Portfolio Theory")
        # Portfolio optimization logic here

    elif options == "Stock Forecast App":
        stock_forecast_app()

if __name__ == "__main__":
    main()
