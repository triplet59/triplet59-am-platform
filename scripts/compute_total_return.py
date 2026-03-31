import pandas as pd


def compute_total_return(price_df, dividends_dict):
    returns = pd.DataFrame(index=price_df.index)

    for col in price_df.columns:
        prices = price_df[col]
        divs = dividends_dict.get(col)

        daily_returns = []

        for i in range(1, len(prices)):
            p_today = prices.iloc[i]
            p_yesterday = prices.iloc[i - 1]

            if pd.isna(p_today) or pd.isna(p_yesterday):
                daily_returns.append(0)
                continue

            ret = (p_today / p_yesterday) - 1

            if divs is not None:
                date = prices.index[i]
                if date in divs.index:
                    ret += divs.loc[date] / p_yesterday

            daily_returns.append(ret)

        returns[col] = [0] + daily_returns

    return returns


def compute_index_return(weights_df, returns_df):
    index_returns = []

    for date in weights_df.index:
        weights = weights_df.loc[date]
        rets = returns_df.loc[date]

        index_ret = (weights * rets).sum()
        index_returns.append(index_ret)

    return pd.Series(index_returns, index=weights_df.index)


def build_index(index_returns):
    index_level = (1 + index_returns).cumprod()
    index_level.iloc[0] = 1.0
    return index_level
