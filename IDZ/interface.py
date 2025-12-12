import streamlit as st
import data
import datetime

st.set_page_config(layout="wide")

assets = {
    "IBM": "IBM",
    "S&P500": "^GSPC",
    "Dow30": "^DJI",
    "Nasdaq100": "^NDX",
    "Russell2000": "^RUT",
    "VIX": "^VIX",
    "FTSE100": "^FTSE",
    "DAX": "^GDAXI",
    "CAC40": "^FCHI",
    "Nikkei225": "^N225",
    "Gold": "GC=F",
    "Silver": "SI=F",
    "CrudeOil": "CL=F",
    "Brent": "BZ=F",
    "NaturalGas": "NG=F",
    "Copper": "HG=F",
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "JPY=X",
    "USDCAD": "CAD=X",
    "USDCHF": "CHF=X",
    "AUDUSD": "AUDUSD=X",
    "Bitcoin": "BTC-USD",
    "Ethereum": "ETH-USD",
    "Solana": "SOL-USD",
    "Apple": "AAPL",
    "Microsoft": "MSFT",
    "Google": "GOOGL",
    "Amazon": "AMZN",
    "Nvidia": "NVDA",
    "Tesla": "TSLA",
    "Meta": "META",
}

cols = st.columns([1, 3])

top_par = cols[0].container()

sel = top_par.multiselect(
    "Stock tickers",
    list(assets),
    default=["IBM", "Gold", "Solana", "Nvidia"],
)

start_time = top_par.slider(
    "Tim line",
    value = (datetime.date(2015, 1, 1), datetime.date(2025, 4, 13)),
    format = "MM/DD/YYYY"
)

tickers = dict()
for i in sel:
    tickers[i] = assets[i]
cols[1].title("Clear data")
cols[1].line_chart(data.prices_to_dataframe(data.iter_market_data(tickers, start = start_time[0], end=start_time[1])))
st.title("Normalization of data")
st.line_chart(data.prices_to_dataframe(data.iter_norm(data.iter_norm(data.iter_market_data(tickers, start = start_time[0], end=start_time[1])))))
st.title("Statistics")
st.write(data.descriptive_statistics(data.prices_to_dataframe(data.iter_market_data(tickers, start = start_time[0], end=start_time[1]))))
st.title("Univariate")
st.write(data.univariate_dataset(data.iter_market_data(tickers, start = start_time[0], end=start_time[1]), 30))

st.title("Multivariate")

selt = st.selectbox(
    "Stock tickers",
    list(tickers),
)

if sel != "":
    st.write(data.multivariate_dataset(data.iter_market_data(tickers, start = start_time[0], end=start_time[1]), selt, 30, 1))
