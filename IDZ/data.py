import pandas as pd
import yfinance
from dataclasses import dataclass
import numpy as np
from typing import Iterator, Iterable


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
    df = df.dropna(inplace=False)
    return df

def iter_market_data(tickers:dict, start:str = "2024-01-01", end:str = "2025-04-13") -> Iterator[PricePoint]:
    #tickers = { "IBM": "IBM", "S&P500":"^GSPC", "Dow30": "^DJI", "Gold": "GC=F", "CrudeOil": "CL=F"}

    for name, element in tickers.items():
        raw = yfinance.download(element, start=start, end=end, progress=False)

        close = raw["Close"]
        close = clear(close)

        for date, price in close.items():
            yield PricePoint(asset=name, date=date, close=float(price))

def norm(lis: list) -> Iterator[PricePoint]:
    min_number = min([number.close for number in lis])
    max_number = max([number.close for number in lis])
    for j in lis:
        yield PricePoint(asset=j.asset, date=j.date, close=((j.close - min_number) / (max_number - min_number)))

def iter_norm(points: Iterable[PricePoint]) -> Iterator[PricePoint]:
    lis = [next(points)]
    for i in points:
        if lis[-1].asset != i.asset:
            yield from norm(lis)
            lis.clear()
        lis.append(i)
    else:
        yield from norm(lis)

def univariate_dataset(points: Iterable[PricePoint], window_size: int, forecast_horizon: int = 1) -> pd.DataFrame:
    records = [{"Date": i.date, "Asset": i.asset, "Close": i.close} for i in points]
    df = pd.DataFrame(records)

    result = []
    asserts = df["Asset"].unique()

    for i in asserts:
        series = df[df["Asset"] == i]["Close"].values
        n = len(series)
        for j in range(n - window_size - forecast_horizon + 1):
            row = {}
            for ij in range(window_size):
                row[f"t-{ij}"] = series[j+ij]
            for ji in range(forecast_horizon):
                row[f"y+{ji}"] = series[j + window_size + ji]
            row["Asset"] = i
            result.append(row)

    return pd.DataFrame(result)

def multivariate_dataset(points: Iterable[PricePoint], target_col: str, windows_size:int, forecast_horizon:int):
    records = [{"Date": i.date, "Asset": i.asset, "Close": i.close} for i in points]
    df = pd.DataFrame(records)
    df = df.pivot(index="Date", columns="Asset", values="Close").dropna()

    x, y = [], []
    cols = df.columns
    values = df.values
    target_idx = df.columns.get_loc(target_col)

    for i in range(len(df) - windows_size - forecast_horizon + 1):
        x_value = (values[i:i + windows_size])
        future_values = values[i + windows_size: i + windows_size + forecast_horizon, target_idx]

        flat = x_value.flatten()
        x.append(flat)
        y.append(future_values)

    features_cols = []
    for i in cols:
        for j in range(windows_size):
            features_cols.append(f"{i}_t-{windows_size-j}")

    y_cols = [f"{target_col}_t+{i+1}" for i in range(forecast_horizon)]

    df_out = pd.DataFrame(x, columns=features_cols)
    df_target = pd.DataFrame(y, columns=y_cols)

    return pd.concat([df_out, df_target], axis=1)

def iter_log(points: Iterable[PricePoint]) -> Iterator[ReturnPoit]:

    l_close: dict[str, float] = {}

    for i in points:
        prev = l_close.get(i.asset)
        if prev is not None:
            lr = float(np.log(i.close / prev))
            yield ReturnPoit(asset=i.asset, date=i.date, log_return=lr)
        l_close[i.asset] = i.close

def descriptive_statistics(df: pd.DataFrame):
    return df.describe()

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

