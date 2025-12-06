import data

def main():
    #data.figt(data.prices_to_dataframe(data.iter_market_data()), "Prices", "USD")
    #data.figt(data.returns_to_dataframe(data.iter_log(data.iter_market_data())), "Log Returns of Assets Over Time", "LogReturn")
    #data.figt(data.prices_to_dataframe(data.iter_norm(data.iter_market_data())), "test", "usd")
    print(data.descriptive_statistics(data.prices_to_dataframe(data.iter_market_data())))
    print(data.univariate_dataset(data.iter_market_data(), 30))
    print(data.multivariate_dataset(data.iter_market_data(), "IBM", 30, 1))

if __name__ == "__main__":
    main()