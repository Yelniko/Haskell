from itertools import islice
import pandas as pd
import pylab as p
import yfinance
from dataclasses import dataclass
import numpy as np
from typing import Iterator, Iterable
import math
import plotly.express as px
import plotly.graph_objects as go
from scipy.signal import argrelextrema




@dataclass(frozen=True)
class PricePoint:
    asset: str
    date:pd.Timestamp
    close: float

@dataclass(frozen=True)
class ReturnPoit:
    asset: str
    date: pd.Timestamp
    log_return: float

def clear(df):
    if isinstance(df, pd.DataFrame):
        df = df.iloc[:, 0]
    df = df.dropna()
    return df

def iter_market_data(start:str = "2015-01-01", end:str = "2025-04-13") -> Iterator[PricePoint]:
    tickers = { "IBM": "IBM", "S&P500":"^GSPC", "Dow30": "^DJI", "Gold": "GC=F", "CrudeOil": "CL=F"}

    for name, element in tickers.items():
        raw = yfinance.download(element, start=start, end=end, progress=False)

        close = raw["Close"]
        close = clear(close)

        for date, price in close.items():
            yield PricePoint(asset=name, date=date, close=float(price))

def iter_log(points: Iterable[PricePoint]) -> Iterator[ReturnPoit]:

    l_close: dict[str, float] = {}

    for i in points:
        prev = l_close.get(i.asset)
        if prev is not None:
            lr = float(np.log(i.close / prev))
            yield ReturnPoit(asset=i.asset, date=i.date, log_return=lr)
        l_close[i.asset] = i.close

def returns_to_dataframe(df: Iterator[ReturnPoit]) -> pd.DataFrame:
    recorder = [{"Date": r.date, "Asset": r.asset, "LogReturn": r.log_return} for r in df]

    df_is = pd.DataFrame(recorder)
    if df_is.empty:
        return df_is

    df_pivot = df_is.pivot(index="Date", columns="Asset", values="LogReturn").sort_index()
    return df_pivot

def prices_to_dataframe(df: Iterator[PricePoint]) -> pd.DataFrame:
    recorder = [{"Date": r.date, "Asset": r.asset, "LogReturn": r.close} for r in df]

    df_is = pd.DataFrame(recorder)
    if df_is.empty:
        return df_is

    df_pivot = df_is.pivot(index="Date", columns="Asset", values="LogReturn").sort_index()
    return df_pivot

def figt(df:pd.DataFrame, name:str, name_y:str):
    fignt = go.Figure()
    for i in df.columns:
        fignt.add_trace(go.Scatter(x=df.index, y=df[i], mode="lines", name=i))
    fignt.update_layout(title=name, xaxis_title="Date", yaxis_title=name_y, height=500)
    fignt.show()

def plot_price_sma(df_prices, sma_windows=[10,50]):
    fig = go.Figure()
    for asset in df_prices.columns:
        prices = df_prices[asset]
        for window in sma_windows:
            sma = prices.rolling(window=window).mean()
            fig.add_trace(go.Scatter(
                x=df_prices.index, y=sma, mode='lines', name=f'{asset} SMA{window}', line=dict(dash='dot')
            ))
    fig.update_layout(title='Simple Moving Averages (SMA)', xaxis_title='Date', yaxis_title='Price')
    fig.show()

def plot_local_extremes(df_prices, order=5):
    fig = go.Figure()
    for asset in df_prices.columns:
        prices = df_prices[asset]
        maxima_idx = argrelextrema(prices.values, np.greater, order=order)[0]
        minima_idx = argrelextrema(prices.values, np.less, order=order)[0]
        fig.add_trace(go.Scatter(
            x=df_prices.index[maxima_idx], y=prices.values[maxima_idx],
            mode='markers', marker=dict(color='green', size=8), name=f'{asset} Peaks'
        ))
        fig.add_trace(go.Scatter(
            x=df_prices.index[minima_idx], y=prices.values[minima_idx],
            mode='markers', marker=dict(color='red', size=8), name=f'{asset} Troughs'
        ))
    fig.update_layout(title='Local Maxima and Minima', xaxis_title='Date', yaxis_title='Price')
    fig.show()

def plot_cumulative_growth(df_prices):
    fig = go.Figure()
    for asset in df_prices.columns:
        cum_growth = df_prices[asset] / df_prices[asset].iloc[0]
        fig.add_trace(go.Scatter(
            x=df_prices.index, y=cum_growth, mode='lines', name=f'{asset} CumGrowth', line=dict(dash='dash')
        ))
    fig.update_layout(title='Cumulative Growth of Assets', xaxis_title='Date', yaxis_title='Cumulative Growth')
    fig.show()

def plot_cumulative_returns(df_returns: pd.DataFrame):
    fig = go.Figure()
    for asset in df_returns.columns:
        cum_return = np.exp(df_returns[asset].cumsum())
        fig.add_trace(go.Scatter(
            x=df_returns.index,
            y=cum_return,
            mode='lines',
            name=asset
        ))
    fig.update_layout(
        title="Cumulative Returns of Assets",
        xaxis_title="Date",
        yaxis_title="Cumulative Growth",
        template="plotly_white"
    )
    fig.show()

def plot_correlation(df_returns: pd.DataFrame):
    corr = df_returns.corr()
    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale='RdBu_r',
        title="Correlation Matrix of Assets' Log Returns"
    )
    fig.show()

def plot_histogram_returns(df_returns: pd.DataFrame, nbins=50):
    df_long = df_returns.reset_index().melt(id_vars='Date', var_name='Asset', value_name='LogReturn')
    fig = px.histogram(df_long, x='LogReturn', color='Asset', nbins=nbins,
                       title="Distribution of Log Returns", marginal="box", opacity=0.7)
    fig.show()

def main():
    figt(prices_to_dataframe(iter_market_data()), "Prices", "USD")
    plot_price_sma(prices_to_dataframe(iter_market_data()))
    plot_local_extremes(prices_to_dataframe(iter_market_data()))
    plot_cumulative_growth(prices_to_dataframe(iter_market_data()))
    figt(returns_to_dataframe(iter_log(iter_market_data())), "Log Returns of Assets Over Time", "LogReturn")
    plot_cumulative_returns(returns_to_dataframe(iter_log(iter_market_data())))
    plot_correlation(returns_to_dataframe(iter_log(iter_market_data())))
    plot_histogram_returns(returns_to_dataframe(iter_log(iter_market_data())))

if __name__ == "__main__":
    main()