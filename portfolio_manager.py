{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "e9bb7019-a57f-434c-9bb5-d6057a4d38a7",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: streamlit in /opt/anaconda3/lib/python3.12/site-packages (1.32.0)\n",
      "Requirement already satisfied: altair<6,>=4.0 in /opt/anaconda3/lib/python3.12/site-packages (from streamlit) (5.0.1)\n",
      "Requirement already satisfied: blinker<2,>=1.0.0 in /opt/anaconda3/lib/python3.12/site-packages (from streamlit) (1.6.2)\n",
      "Requirement already satisfied: cachetools<6,>=4.0 in /opt/anaconda3/lib/python3.12/site-packages (from streamlit) (5.3.3)\n",
      "Requirement already satisfied: click<9,>=7.0 in /opt/anaconda3/lib/python3.12/site-packages (from streamlit) (8.1.7)\n",
      "Requirement already satisfied: numpy<2,>=1.19.3 in /opt/anaconda3/lib/python3.12/site-packages (from streamlit) (1.26.4)\n",
      "Requirement already satisfied: packaging<24,>=16.8 in /opt/anaconda3/lib/python3.12/site-packages (from streamlit) (23.2)\n",
      "Requirement already satisfied: pandas<3,>=1.3.0 in /opt/anaconda3/lib/python3.12/site-packages (from streamlit) (2.2.2)\n",
      "Requirement already satisfied: pillow<11,>=7.1.0 in /opt/anaconda3/lib/python3.12/site-packages (from streamlit) (10.3.0)\n",
      "Requirement already satisfied: protobuf<5,>=3.20 in /opt/anaconda3/lib/python3.12/site-packages (from streamlit) (3.20.3)\n",
      "Requirement already satisfied: pyarrow>=7.0 in /opt/anaconda3/lib/python3.12/site-packages (from streamlit) (14.0.2)\n",
      "Requirement already satisfied: requests<3,>=2.27 in /opt/anaconda3/lib/python3.12/site-packages (from streamlit) (2.32.2)\n",
      "Requirement already satisfied: rich<14,>=10.14.0 in /opt/anaconda3/lib/python3.12/site-packages (from streamlit) (13.3.5)\n",
      "Requirement already satisfied: tenacity<9,>=8.1.0 in /opt/anaconda3/lib/python3.12/site-packages (from streamlit) (8.2.2)\n",
      "Requirement already satisfied: toml<2,>=0.10.1 in /opt/anaconda3/lib/python3.12/site-packages (from streamlit) (0.10.2)\n",
      "Requirement already satisfied: typing-extensions<5,>=4.3.0 in /opt/anaconda3/lib/python3.12/site-packages (from streamlit) (4.11.0)\n",
      "Requirement already satisfied: gitpython!=3.1.19,<4,>=3.0.7 in /opt/anaconda3/lib/python3.12/site-packages (from streamlit) (3.1.37)\n",
      "Requirement already satisfied: pydeck<1,>=0.8.0b4 in /opt/anaconda3/lib/python3.12/site-packages (from streamlit) (0.8.0)\n",
      "Requirement already satisfied: tornado<7,>=6.0.3 in /opt/anaconda3/lib/python3.12/site-packages (from streamlit) (6.4.1)\n",
      "Requirement already satisfied: jinja2 in /opt/anaconda3/lib/python3.12/site-packages (from altair<6,>=4.0->streamlit) (3.1.4)\n",
      "Requirement already satisfied: jsonschema>=3.0 in /opt/anaconda3/lib/python3.12/site-packages (from altair<6,>=4.0->streamlit) (4.19.2)\n",
      "Requirement already satisfied: toolz in /opt/anaconda3/lib/python3.12/site-packages (from altair<6,>=4.0->streamlit) (0.12.0)\n",
      "Requirement already satisfied: gitdb<5,>=4.0.1 in /opt/anaconda3/lib/python3.12/site-packages (from gitpython!=3.1.19,<4,>=3.0.7->streamlit) (4.0.7)\n",
      "Requirement already satisfied: python-dateutil>=2.8.2 in /opt/anaconda3/lib/python3.12/site-packages (from pandas<3,>=1.3.0->streamlit) (2.9.0.post0)\n",
      "Requirement already satisfied: pytz>=2020.1 in /opt/anaconda3/lib/python3.12/site-packages (from pandas<3,>=1.3.0->streamlit) (2024.1)\n",
      "Requirement already satisfied: tzdata>=2022.7 in /opt/anaconda3/lib/python3.12/site-packages (from pandas<3,>=1.3.0->streamlit) (2023.3)\n",
      "Requirement already satisfied: charset-normalizer<4,>=2 in /opt/anaconda3/lib/python3.12/site-packages (from requests<3,>=2.27->streamlit) (2.0.4)\n",
      "Requirement already satisfied: idna<4,>=2.5 in /opt/anaconda3/lib/python3.12/site-packages (from requests<3,>=2.27->streamlit) (3.7)\n",
      "Requirement already satisfied: urllib3<3,>=1.21.1 in /opt/anaconda3/lib/python3.12/site-packages (from requests<3,>=2.27->streamlit) (2.2.2)\n",
      "Requirement already satisfied: certifi>=2017.4.17 in /opt/anaconda3/lib/python3.12/site-packages (from requests<3,>=2.27->streamlit) (2024.7.4)\n",
      "Requirement already satisfied: markdown-it-py<3.0.0,>=2.2.0 in /opt/anaconda3/lib/python3.12/site-packages (from rich<14,>=10.14.0->streamlit) (2.2.0)\n",
      "Requirement already satisfied: pygments<3.0.0,>=2.13.0 in /opt/anaconda3/lib/python3.12/site-packages (from rich<14,>=10.14.0->streamlit) (2.15.1)\n",
      "Requirement already satisfied: smmap<5,>=3.0.1 in /opt/anaconda3/lib/python3.12/site-packages (from gitdb<5,>=4.0.1->gitpython!=3.1.19,<4,>=3.0.7->streamlit) (4.0.0)\n",
      "Requirement already satisfied: MarkupSafe>=2.0 in /opt/anaconda3/lib/python3.12/site-packages (from jinja2->altair<6,>=4.0->streamlit) (2.1.3)\n",
      "Requirement already satisfied: attrs>=22.2.0 in /opt/anaconda3/lib/python3.12/site-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (23.1.0)\n",
      "Requirement already satisfied: jsonschema-specifications>=2023.03.6 in /opt/anaconda3/lib/python3.12/site-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (2023.7.1)\n",
      "Requirement already satisfied: referencing>=0.28.4 in /opt/anaconda3/lib/python3.12/site-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (0.30.2)\n",
      "Requirement already satisfied: rpds-py>=0.7.1 in /opt/anaconda3/lib/python3.12/site-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (0.10.6)\n",
      "Requirement already satisfied: mdurl~=0.1 in /opt/anaconda3/lib/python3.12/site-packages (from markdown-it-py<3.0.0,>=2.2.0->rich<14,>=10.14.0->streamlit) (0.1.0)\n",
      "Requirement already satisfied: six>=1.5 in /opt/anaconda3/lib/python3.12/site-packages (from python-dateutil>=2.8.2->pandas<3,>=1.3.0->streamlit) (1.16.0)\n",
      "Note: you may need to restart the kernel to use updated packages.\n"
     ]
    }
   ],
   "source": [
    "pip install streamlit\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "a16ec6cf-a44c-4d37-9fe3-94de1b75f38a",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pickle"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "2f4070ba-a44b-42ae-91fc-ad1fd20ba9c9",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "c051467d-52b4-4bb7-bb42-181e8d064075",
   "metadata": {},
   "outputs": [],
   "source": [
    "import yfinance as yf\n",
    "import numpy as np\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "492e2b81-240e-4b87-9ee4-7a02181f6235",
   "metadata": {},
   "outputs": [],
   "source": [
    "import matplotlib.pyplot as p"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "04b82964-abd1-49e6-adaf-b85fc44e0c29",
   "metadata": {},
   "outputs": [],
   "source": [
    "from scipy.optimize import minimize"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "272a7be1-cdac-494b-aed4-c1692dc55b35",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: prophet in /opt/anaconda3/lib/python3.12/site-packages (1.1.6)\n",
      "Requirement already satisfied: cmdstanpy>=1.0.4 in /opt/anaconda3/lib/python3.12/site-packages (from prophet) (1.2.4)\n",
      "Requirement already satisfied: numpy>=1.15.4 in /opt/anaconda3/lib/python3.12/site-packages (from prophet) (1.26.4)\n",
      "Requirement already satisfied: matplotlib>=2.0.0 in /opt/anaconda3/lib/python3.12/site-packages (from prophet) (3.8.4)\n",
      "Requirement already satisfied: pandas>=1.0.4 in /opt/anaconda3/lib/python3.12/site-packages (from prophet) (2.2.2)\n",
      "Requirement already satisfied: holidays<1,>=0.25 in /opt/anaconda3/lib/python3.12/site-packages (from prophet) (0.62)\n",
      "Requirement already satisfied: tqdm>=4.36.1 in /opt/anaconda3/lib/python3.12/site-packages (from prophet) (4.66.4)\n",
      "Requirement already satisfied: importlib-resources in /opt/anaconda3/lib/python3.12/site-packages (from prophet) (6.4.5)\n",
      "Requirement already satisfied: stanio<2.0.0,>=0.4.0 in /opt/anaconda3/lib/python3.12/site-packages (from cmdstanpy>=1.0.4->prophet) (0.5.1)\n",
      "Requirement already satisfied: python-dateutil in /opt/anaconda3/lib/python3.12/site-packages (from holidays<1,>=0.25->prophet) (2.9.0.post0)\n",
      "Requirement already satisfied: contourpy>=1.0.1 in /opt/anaconda3/lib/python3.12/site-packages (from matplotlib>=2.0.0->prophet) (1.2.0)\n",
      "Requirement already satisfied: cycler>=0.10 in /opt/anaconda3/lib/python3.12/site-packages (from matplotlib>=2.0.0->prophet) (0.11.0)\n",
      "Requirement already satisfied: fonttools>=4.22.0 in /opt/anaconda3/lib/python3.12/site-packages (from matplotlib>=2.0.0->prophet) (4.51.0)\n",
      "Requirement already satisfied: kiwisolver>=1.3.1 in /opt/anaconda3/lib/python3.12/site-packages (from matplotlib>=2.0.0->prophet) (1.4.4)\n",
      "Requirement already satisfied: packaging>=20.0 in /opt/anaconda3/lib/python3.12/site-packages (from matplotlib>=2.0.0->prophet) (23.2)\n",
      "Requirement already satisfied: pillow>=8 in /opt/anaconda3/lib/python3.12/site-packages (from matplotlib>=2.0.0->prophet) (10.3.0)\n",
      "Requirement already satisfied: pyparsing>=2.3.1 in /opt/anaconda3/lib/python3.12/site-packages (from matplotlib>=2.0.0->prophet) (3.0.9)\n",
      "Requirement already satisfied: pytz>=2020.1 in /opt/anaconda3/lib/python3.12/site-packages (from pandas>=1.0.4->prophet) (2024.1)\n",
      "Requirement already satisfied: tzdata>=2022.7 in /opt/anaconda3/lib/python3.12/site-packages (from pandas>=1.0.4->prophet) (2023.3)\n",
      "Requirement already satisfied: six>=1.5 in /opt/anaconda3/lib/python3.12/site-packages (from python-dateutil->holidays<1,>=0.25->prophet) (1.16.0)\n"
     ]
    }
   ],
   "source": [
    "!pip install prophet\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "5e398a78-f12a-491f-8ca6-3970b7fdaaf2",
   "metadata": {},
   "outputs": [],
   "source": [
    "import prophet\n",
    "print(prophet.__version__)\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "dd768433-a4af-4948-bbc2-76e76b9d78b8",
   "metadata": {},
   "outputs": [],
   "source": [
    "pip install --no-cache-dir prophet\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "e01029a5-e541-47fd-811b-b11668862285",
   "metadata": {},
   "outputs": [],
   "source": [
    "sudo apt-get install gcc"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "246b3c74-8c19-4567-b9b8-f6893ea2b346",
   "metadata": {},
   "outputs": [],
   "source": [
    "pip install fbprophet\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "e3bb42ed-3ead-4c0c-b947-e51d41339ce8",
   "metadata": {},
   "outputs": [],
   "source": [
    "import prophet\n",
    "print(\"Prophet installed successfully\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "f9c7ec51-227b-4249-8ac3-8ac866382d39",
   "metadata": {},
   "outputs": [],
   "source": [
    "import seaborn as sns\n",
    "import os\n",
    "from datetime import date\n",
    "from prophet import Prophet\n",
    "from prophet.plot import plot_plotly\n",
    "from plotly import graph_objs as go\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "id": "585b3516-4f13-4a17-8c74-8b265a44b99f",
   "metadata": {},
   "outputs": [],
   "source": [
    "def load_bank_accounts(): \n",
    "    try:\n",
    "        if os.path.exists('bank_accounts.pkl'):\n",
    "            with open('bank_accounts.pkl', 'rb') as f: \n",
    "                return pickle.load(f)\n",
    "        return {}  # This return statement should not be inside the \"with\" block\n",
    "    except (FileNotFoundError, EOFError, pickle.UnpicklingError) as e:\n",
    "        print(f\"Error loading bank accounts: {e}\")  # Changed 'st.warning' to 'print' unless 'st' is defined for Streamlit\n",
    "        return {}\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "id": "b6b2e775-adde-4006-9c17-e09e0d8d19ee",
   "metadata": {},
   "outputs": [],
   "source": [
    "bank_accounts = load_bank_accounts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "id": "96a5122b-c2a3-412b-bbf7-2bf3daf79fd2",
   "metadata": {},
   "outputs": [],
   "source": [
    "def save_bank_accounts(): \n",
    "    try:\n",
    "        with open('bank_accounts.pkl', 'wb') as f: \n",
    "            pickle.dump(bank_accounts, f)\n",
    "    except Exception as e:\n",
    "        st.error(f\"Error saving bank accounts: {e}\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "id": "1142f0ca-3a08-4b09-b564-f70ea820d402",
   "metadata": {},
   "outputs": [],
   "source": [
    "def add_bank_account(account_name, balance): \n",
    "    bank_accounts[account_name] = balance\n",
    "    save_bank_accounts()\n",
    "    return f\"Added bank account '{account_name}' with balance ${balance:.2f}.\"\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "id": "fcbdb26a-4bcc-4bf2-8d60-479c0f89bfd1",
   "metadata": {},
   "outputs": [],
   "source": [
    "def remove_bank_account(account_name, bank_accounts): \n",
    "    if account_name in bank_accounts:\n",
    "        del bank_accounts[account_name]\n",
    "        save_bank_accounts()  # Ensure this function is defined elsewhere\n",
    "        return f\"Removed bank account '{account_name}'.\"\n",
    "    else:\n",
    "        return f\"Bank account '{account_name}' not found.\"\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "id": "dfc25843-afbd-473b-b760-18c2b9bdb9b5",
   "metadata": {},
   "outputs": [],
   "source": [
    "def update_bank_account(account_name, balance):\n",
    "    if account_name in bank_accounts:\n",
    "        bank_accounts[account_name] = balance\n",
    "        save_bank_accounts()\n",
    "        return f\"Updated bank account '{account_name}' to balance ${balance:.2f}.\"\n",
    "    else:\n",
    "        return f\"Bank account '{account_name}' not found.\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "id": "8b9c77d1-1636-47b5-acc4-25a07e276de1",
   "metadata": {},
   "outputs": [],
   "source": [
    "def show_bank_accounts():\n",
    "    if bank_accounts:\n",
    "        response = \"Your bank accounts:\\n\"\n",
    "        for account_name, balance in bank_accounts.items():\n",
    "            response += f\"{account_name}: ${balance:.2f}\\n\"\n",
    "        return response.strip()\n",
    "    else:\n",
    "        return \"No bank accounts found.\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "id": "f0817b35-64a0-4710-98b6-6417191357f6",
   "metadata": {},
   "outputs": [],
   "source": [
    "def total_balance():\n",
    "    return f\"Total balance across all accounts: ${sum(bank_accounts.values()):.2f}\"\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "id": "3a938718-b5ca-4113-a478-3f3959f3574c",
   "metadata": {},
   "outputs": [],
   "source": [
    "# ---- Stock Portfolio Functions ----\n",
    "portfolio = {}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "id": "130c3889-4fda-45ec-81b3-a6e3ca757863",
   "metadata": {},
   "outputs": [],
   "source": [
    "def load_portfolio():\n",
    "    try:\n",
    "        if os.path.exists('portfolio.pkl'):\n",
    "            with open('portfolio.pkl', 'rb') as f:\n",
    "                return pickle.load(f)\n",
    "        return {}\n",
    "    except (FileNotFoundError, EOFError, pickle.UnpicklingError) as e:\n",
    "        st.warning(f\"Error loading portfolio: {e}\")\n",
    "        return {}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "id": "20aba265-6376-405b-9042-79d06d9cd0a1",
   "metadata": {},
   "outputs": [],
   "source": [
    "def save_portfolio():\n",
    "    try:\n",
    "        with open('portfolio.pkl', 'wb') as f:\n",
    "            pickle.dump(portfolio, f)\n",
    "    except Exception as e:\n",
    "        st.error(f\"Error saving portfolio: {e}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "id": "a6931bca-2b9c-4bc0-b633-1ef24cc48655",
   "metadata": {},
   "outputs": [],
   "source": [
    "def add_stock(ticker, amount):\n",
    "    ticker = ticker.upper()\n",
    "    if ticker in portfolio:\n",
    "        portfolio[ticker] += amount\n",
    "    else:\n",
    "        portfolio[ticker] = amount\n",
    "    save_portfolio()\n",
    "    return f\"Added {amount} shares of {ticker} to your portfolio.\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 59,
   "id": "62454b1b-a037-484c-9607-12ac40180677",
   "metadata": {},
   "outputs": [],
   "source": [
    "def remove_stock(ticker, amount):\n",
    "    ticker = ticker.upper()\n",
    "    if ticker in portfolio:\n",
    "        if portfolio[ticker] >= amount:\n",
    "            portfolio[ticker] -= amount\n",
    "            if portfolio[ticker] == 0:\n",
    "                del portfolio[ticker]\n",
    "            save_portfolio()\n",
    "            return f\"Removed {amount} shares of {ticker} from your portfolio.\"\n",
    "        else:\n",
    "            return \"You don't have that many shares.\"\n",
    "    else:\n",
    "        return f\"No shares of {ticker} found in your portfolio.\"\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "id": "44a91523-5279-4f10-b38f-9ad212cb9fa9",
   "metadata": {},
   "outputs": [],
   "source": [
    "def show_portfolio():\n",
    "    if portfolio:\n",
    "        response = \"Your current portfolio:\\n\"\n",
    "        for ticker, shares in portfolio.items():\n",
    "            response += f\"{ticker}: {shares} shares\\n\"\n",
    "        return response.strip()\n",
    "    else:\n",
    "        return \"Your portfolio is empty.\"\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 63,
   "id": "775f8209-2a80-42b8-949e-a1de313aff43",
   "metadata": {},
   "outputs": [],
   "source": [
    "def portfolio_worth():\n",
    "    total_value = 0\n",
    "    response = \"\"\n",
    "    for ticker, shares in portfolio.items():\n",
    "        try:\n",
    "            stock = yf.Ticker(ticker)\n",
    "            price = stock.history(period=\"1d\")['Close'].iloc[-1]\n",
    "            total_value += shares * price\n",
    "            response += f\"{ticker}: {shares} shares at ${price:.2f} each, total ${shares * price:.2f}\\n\"\n",
    "        except IndexError:\n",
    "            response += f\"Could not retrieve data for {ticker}. Data might be unavailable.\\n\"\n",
    "        except Exception as e:\n",
    "            response += f\"Error retrieving data for {ticker}: {e}\\n\"\n",
    "    response += f\"Your portfolio is worth: ${total_value:.2f} USD\"\n",
    "    return response\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 69,
   "id": "54ad5872-bab1-46fe-9c92-ab7cf86aa8fe",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 71,
   "id": "7476e575-031b-4c92-985f-e02ddd7df0d2",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2024-12-03 22:09:05.799 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run /opt/anaconda3/lib/python3.12/site-packages/ipykernel_launcher.py [ARGUMENTS]\n"
     ]
    }
   ],
   "source": [
    "def main():\n",
    "    st.title(\"Investment Portfolio Management\")\n",
    "    st.sidebar.title(\"Navigation\")\n",
    "    options = st.sidebar.radio(\n",
    "        \"Select an option\", \n",
    "        (\"Bank Accounts\", \"Stocks\", \"Portfolio Worth\")\n",
    "    )\n",
    "\n",
    "    if options == \"Bank Accounts\":\n",
    "        st.header(\"Manage Bank Accounts\")\n",
    "        action = st.selectbox(\"Action\", (\"Add Account\", \"Remove Account\", \"Update Account\", \"Show Accounts\"))\n",
    "        if action == \"Add Account\":\n",
    "            account_name = st.text_input(\"Account Name\")\n",
    "            balance = st.number_input(\"Balance\", min_value=0.0)\n",
    "            if st.button(\"Add Account\"):\n",
    "                st.write(add_bank_account(account_name, balance))\n",
    "        elif action == \"Remove Account\":\n",
    "            account_name = st.text_input(\"Account Name\")\n",
    "            if st.button(\"Remove Account\"):\n",
    "                st.write(remove_bank_account(account_name))\n",
    "        elif action == \"Update Account\":\n",
    "            account_name = st.text_input(\"Account Name\")\n",
    "            balance = st.number_input(\"New Balance\", min_value=0.0)\n",
    "            if st.button(\"Update Account\"):\n",
    "                st.write(update_bank_account(account_name, balance))\n",
    "        elif action == \"Show Accounts\":\n",
    "            st.write(show_bank_accounts())\n",
    "            st.write(total_balance())\n",
    "\n",
    "    elif options == \"Stocks\":\n",
    "        st.header(\"Manage Stock Portfolio\")\n",
    "        action = st.selectbox(\"Action\", (\"Add Stock\", \"Remove Stock\", \"Show Portfolio\"))\n",
    "        if action == \"Add Stock\":\n",
    "            ticker = st.text_input(\"Stock Ticker\")\n",
    "            amount = st.number_input(\"Number of Shares\", min_value=0)\n",
    "            if st.button(\"Add Stock\"):\n",
    "                st.write(add_stock(ticker, amount))\n",
    "        elif action == \"Remove Stock\":\n",
    "            ticker = st.text_input(\"Stock Ticker\")\n",
    "            amount = st.number_input(\"Number of Shares\", min_value=0)\n",
    "            if st.button(\"Remove Stock\"):\n",
    "                st.write(remove_stock(ticker, amount))\n",
    "        elif action == \"Show Portfolio\":\n",
    "            st.write(show_portfolio())\n",
    "            st.write(portfolio_worth())\n",
    "\n",
    "    elif options == \"Portfolio Worth\":\n",
    "        st.header(\"Portfolio Worth\")\n",
    "        st.write(portfolio_worth())\n",
    "\n",
    "if __name__ == \"__main__\":\n",
    "    main()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "3057fa11-b51c-4f14-86e6-ddaa804a3e35",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "7d6664c3-1fa4-41d8-be1d-4ce411e5519c",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
