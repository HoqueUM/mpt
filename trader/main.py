import streamlit as st
from strat_finder import get_best_strat
import yfinance as yf

st.title('Best Strategy Finder')

min_price = st.number_input('Minimum Price', min_value=1.0, value=50.0)
max_price = st.number_input('Maximum Price', min_value=1.0,  value=100.0)
total_portfolio_value = st.number_input('Total Portfolio Value', min_value=10.0, value=1000.0)

if st.button('Find Best Strategy'):
    try:
        with st.spinner('Finding the best strategy... Please wait'):
            best_strat = get_best_strat(min_price=min_price, max_price=max_price, total_portfolio_value=total_portfolio_value)
        
        if best_strat:
            for symbol, strategy in best_strat.items():
                st.subheader(f"Symbol: {symbol}")
                st.write(f"Fast MA: {strategy['slow_ma']}")
                st.write(f"Slow MA: {strategy['fast_ma']}")
                st.write(f"Return: {strategy['return']:.2f}%")
                st.write(f"Number of Shares: {strategy['num_shares']}")
        else:
            st.write("No strategies found.")
    except Exception as e:
        st.error(f"An error occurred: {e}")