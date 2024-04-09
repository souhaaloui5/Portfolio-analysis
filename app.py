# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 23:56:36 2024

@author: aloui
"""

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as pdr
from finquant.portfolio import build_portfolio
import numpy as np
import streamlit as st
import datetime as dt



pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 200)

def get_stock_data_with_streamlit():
    # Streamlit user input widgets
    stock_symbol = st.text_input('Enter stock symbol', value='AAPL')
    start_date = st.date_input('Start date', value=pd.to_datetime('2023-01-01'))
    end_date = st.date_input('End date', value=pd.to_datetime('2023-12-31'))
    
    if st.button('Fetch Data'):
        # Fetch the stock data
        st.write(f"Fetching data for {stock_symbol} from {start_date} to {end_date}...")
        stock_data = yf.download(stock_symbol, start=start_date, end=end_date)
        
        # Choose 'Close' or 'Adj Close'
        data_to_use = stock_data['Close'] if 'Close' in stock_data.columns else stock_data['Adj Close']
        
        # Optionally display the data in the app
        st.write(data_to_use)
        
        return data_to_use
    
def evaluate_portfolio_impact_with_streamlit(portfolio, start_date, end_date):
    st.header('Evaluate Portfolio Impact')
    
    portfolio_returns = pd.DataFrame()
    for asset in portfolio:
        stock_symbol = asset["stock_name"]
        weight = asset["weight"]
        
        # Fetch the stock data
        stock_data = yf.download(stock_symbol, start=start_date, end=end_date)
        
        # Choose 'Close' or 'Adj Close' and ensure only one column is assigned
        data_to_use = stock_data['Close'] if 'Close' in stock_data.columns else stock_data['Adj Close']
        # Assign to DataFrame, ensuring it's a Series (single column)
        portfolio_returns[stock_symbol] = (data_to_use.pct_change() * weight).fillna(0)

    # Calculate total portfolio returns by summing individual asset returns
    portfolio_returns['Portfolio'] = portfolio_returns.sum(axis=1)

    # Fetch macroeconomic indicators
    gdp_data = pdr.get_data_fred('GDP', start_date, end_date)
    unemployment_data = pdr.get_data_fred('UNRATE', start_date, end_date)
    interest_rate_data = pdr.get_data_fred('GS10', start_date, end_date)
    macroeconomic_data = pd.concat([gdp_data, unemployment_data, interest_rate_data], axis=1)
    macroeconomic_data.columns = ['GDP', 'Unemployment Rate', 'Interest Rate']
    
    # Display the data
    st.write("Portfolio Returns:", portfolio_returns)
    st.write("Macroeconomic Indicators:", macroeconomic_data)

    # Correlation matrix
    combined_data = pd.concat([portfolio_returns[['Portfolio']], macroeconomic_data], axis=1)
    correlation_matrix = combined_data.corr()
    st.write("Correlation Matrix:", correlation_matrix)
def evaluate_portfolio_impact_finquant_with_streamlit(portfolio, start_date, end_date):
    st.subheader('Portfolio Evaluation Results')
    stock_data = {}
    for asset in portfolio:
        stock_symbol = asset["stock_name"]
        stock_data[stock_symbol] = yf.download(stock_symbol, start=start_date, end=end_date)['Close']

    stock_prices = pd.DataFrame(stock_data)
    daily_returns = stock_prices.pct_change()

    weights = np.array([asset["weight"] for asset in portfolio])

    # Perform calculations
    annual_returns = daily_returns.mean() * 252
    portfolio_return = np.dot(annual_returns, weights)
    cov_matrix = daily_returns.cov() * 252
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    risk_free_rate = 0.02
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility

    # Display results
    st.metric("Expected Annual Return", f"{portfolio_return*100:.2f}%")
    st.metric("Annual Volatility/Risk", f"{portfolio_volatility*100:.2f}%")
    st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")

def generate_efficient_frontier_with_streamlit(portfolio, start_date, end_date):
    st.write("Generating Efficient Frontier...")
    stock_data = {}
    for asset in portfolio:
        stock_symbol = asset["stock_name"]
        stock_data[stock_symbol] = yf.download(stock_symbol, start=start_date, end=end_date)['Close']

    stock_prices = pd.DataFrame(stock_data)

    if not stock_prices.empty:
        pf = build_portfolio(data=stock_prices)
        # Directly plotting without using 'ax' argument
        fig = plt.figure(figsize=(10, 7))
        pf.ef_plot_efrontier()
        plt.title("Efficient Frontier")
        plt.xlabel("Annualized Volatility")
        plt.ylabel("Annualized Returns")
        st.pyplot(fig)
    else:
        st.error("Error: No stock data fetched. Check your symbols and date range.")


    
    
def main_with_streamlit():
    st.title("Portfolio Analysis Tool")
    
    # Portfolio setup with Streamlit
    st.header("Portfolio Setup")
    num_assets = st.number_input("Enter the number of assets in your portfolio:", min_value=1, value=3, step=1)
    portfolio = []
    for i in range(num_assets):
        col1, col2 = st.columns(2)
        with col1:
            stock_name = st.text_input(f"Asset {i+1} stock name:", key=f'stock_name_{i}')
        with col2:
            weight = st.number_input(f"Asset {i+1} weight:", min_value=0.0, max_value=1.0, value=0.33, step=0.01, key=f'weight_{i}')
        portfolio.append({"stock_name": stock_name, "weight": weight})

    # Time period selection
    st.header("Select Time Period")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start date", value=pd.to_datetime('2023-01-01'))
    with col2:
        end_date = st.date_input("End date", value=pd.to_datetime('2023-12-31'))

    if st.button('Evaluate Portfolio'):
        # Trigger evaluation and display results
        evaluate_portfolio_impact_with_streamlit(portfolio, start_date, end_date)
        evaluate_portfolio_impact_finquant_with_streamlit(portfolio, start_date, end_date)
        generate_efficient_frontier_with_streamlit(portfolio, start_date, end_date)

if __name__ == "__main__":
    main_with_streamlit()    
    
    
    
    
    
    
    
    
    
    
    
  