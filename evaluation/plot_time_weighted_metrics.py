import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

"""
Time weighted indicates that we are not considering the cash flows in the calculation of returns.
This is useful for comparing strategies on a level playing field, as it removes their impact on returns.
"""

def calculate_time_weighted_return(total_values: pd.Series, monthly_cash: float) -> pd.Series:
    """
    Calculate the time-weighted return (TWR) series given a portfolio's total values and monthly contributions.

    Parameters:
    - total_values: Series of total portfolio values including monthly deposits
    - monthly_cash: Amount added at the start of each month

    Returns:
    - Series of cumulative time-weighted returns (in decimal, not percent)
    """

    # Compute net return each period, adjusting for the contribution at the start
    # Example
    # tv = [10 20 30] (assuming contribution: 5, net gain:5)
    # start_value = [NaN 10 30] (shift to the right)
    # net_gain for first elements = 20 - (10 + 5) = 5 which is the contribution
    tv = total_values.dropna()
    start_value = tv.shift(1) + monthly_cash
    net_gain = (tv - start_value)
    net_return = net_gain / start_value # belongs to [0, 1]
    net_return = net_return.dropna()

    # Chain-link returns
    twr = (1 + net_return).cumprod()

    # Manually inserting starting vlaue point 1.0  and reindex to original series
    twr = pd.concat([pd.Series([1.0], index=[tv.index[0]]), twr])
    twr = twr.sort_index()

    # twr is in growth terms (e.g. [1.00, 1.05, 1.1 , ... , 10.05, ...])
    # we subtract 1 to express it as return
    return twr - 1 # Return as growth from 0


def plot_time_weighted_returns(strategy_data, monthly_cash, benchmarks=None):
    fig, ax = plt.subplots(figsize=(16, 6))
    for name, data in strategy_data.items():
        tv = data['total_values']
        twr = calculate_time_weighted_return(tv, monthly_cash)
        ax.plot(tv.index, twr * 100, label=f"{name} ({twr.iloc[-1]:.2%})")
    for bench in benchmarks.columns:
        bench_twr = calculate_time_weighted_return(tv, monthly_cash)
        ax.plot(benchmarks.index, bench_twr * 100, linestyle='--', label=f"Benchmark: {bench} ({bench_twr.iloc[-1]:.2%})")
    ax.set_title("Time-Weighted Returns (%) with Benchmarks")
    ax.set_xlabel("Date")
    ax.set_ylabel("Return (%)")
    ax.legend(loc='upper left')
    ax.grid(True)
    plt.tight_layout()
    plt.show()


def plot_drawdowns(strategy_data, monthly_cash, benchmarks=None):
    """
    Plot drawdowns of each strategy using time-weighted returns.
    """
    fig, ax = plt.subplots(figsize=(16, 6))
    for name, data in strategy_data.items():
        twr = calculate_time_weighted_return(data['total_values'], monthly_cash)
        peak = twr.cummax()
        drawdown = (twr - peak) * 100
        ax.plot(twr.index, drawdown, label=f"{name} (Max DD: {drawdown.min():.2f}%)")
    if benchmarks is not None:
        for bench in benchmarks.columns:
            twr = calculate_time_weighted_return(benchmarks[bench], monthly_cash)
            peak = twr.cummax()
            drawdown = (twr - peak) * 100
            ax.plot(twr.index, drawdown, linestyle='--', label=f"Benchmark: {bench} (Max DD: {drawdown.min():.2f}%)")
    ax.set_title("Strategy Drawdowns (%)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown (%)")
    ax.legend(loc='lower left')
    ax.grid(True)
    plt.tight_layout()
    plt.show()
    
    
def plot_rolling_metrics(strategy_data, monthly_cash, benchmarks=None, rolling_window=6):
    """
    Plot rolling volatility and returns using time-weighted returns.
    """
    fig, ax = plt.subplots(figsize=(16, 6))
    for name, data in strategy_data.items():
        twr = calculate_time_weighted_return(data['total_values'], monthly_cash)
        ret = twr.diff()
        rolling_vol = ret.rolling(rolling_window).std()
        rolling_ret = ret.rolling(rolling_window).mean()
        ax.plot(ret.index, rolling_ret * 100, label=f"{name} (Avg Ret: {rolling_ret.mean() * 100:.2f}%)")
    if benchmarks is not None:
        for bench in benchmarks.columns:
            twr = calculate_time_weighted_return(benchmarks[bench], monthly_cash)
            ret = twr.diff()
            rolling_ret = ret.rolling(rolling_window).mean()
            ax.plot(ret.index, rolling_ret * 100, linestyle='--', label=f"Benchmark: {bench} (Avg Ret: {rolling_ret.mean() * 100:.2f}%)")
    ax.set_title(f"Rolling {rolling_window}-Period Return (%)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Return (%)")
    ax.legend(loc='upper left')
    ax.grid(True)
    plt.tight_layout()
    plt.show()
    

def plot_risk_return_scatter(strategy_data, monthly_cash, benchmarks=None):
    """
    Risk-return scatter plot using time-weighted returns.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, data in strategy_data.items():
        twr = calculate_time_weighted_return(data['total_values'], monthly_cash)
        returns = twr.diff().dropna()
        mean_return = returns.mean() * 12  # annualized
        volatility = returns.std() * (12 ** 0.5)
        ax.scatter(volatility, mean_return, label=f"{name}", s=100)
        ax.annotate(name, (volatility, mean_return), textcoords="offset points", xytext=(5,5), ha='left')
    if benchmarks is not None:
        for bench in benchmarks.columns:
            twr = calculate_time_weighted_return(benchmarks[bench], monthly_cash)
            returns = twr.diff().dropna()
            mean_return = returns.mean() * 12
            volatility = returns.std() * (12 ** 0.5)
            ax.scatter(volatility, mean_return, label=f"Benchmark: {bench}", marker='x', s=100)
            ax.annotate(bench, (volatility, mean_return), textcoords="offset points", xytext=(5,5), ha='left')
    ax.set_title("Risk vs. Return (Annualized TWR)")
    ax.set_xlabel("Volatility")
    ax.set_ylabel("Return")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.show()
    


def compute_strategy_metrics(strategy_data, monthly_cash, benchmarks=None, risk_free_rate=0.0125, key_benchmark=None):
    """
    Compute a table of key performance metrics for all strategies and benchmarks using time-weighted returns.

    Parameters:
    - strategy_data: dict of strategy results
    - monthly_cash: monthly capital added
    - benchmark_series: pd.DataFrame with benchmark series (e.g. 'SPY')
    - risk_free_rate: annual risk-free rate in decimal (e.g., 0.02)

    Returns:
    - pd.DataFrame with strategy/benchmark metrics
    """
    metrics = []

    def compute_indicators(twr_series, name, key_benchmark=None):
        returns = twr_series.diff().dropna()
        ann_return = returns.mean() * 12
        ann_vol = returns.std() * np.sqrt(12)
        sharpe = (ann_return - risk_free_rate) / ann_vol if ann_vol != 0 else np.nan

        downside = returns[returns < 0]
        sortino = (ann_return - risk_free_rate) / (downside.std() * np.sqrt(12)) if downside.std() > 0 else np.nan

        peak = twr_series.cummax()
        drawdown = (twr_series - peak)
        max_dd = drawdown.min()

        cagr = (1 + twr_series.iloc[-1])**(12 / len(twr_series)) - 1

        if key_benchmark is not None:
            key_bench_twr = calculate_time_weighted_return(key_benchmark, monthly_cash)
            key_bench_returns = key_bench_twr.diff().dropna()
            key_bench_returns.reindex(returns.index).dropna()
            excess_returns = returns - key_bench_returns
            info_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(12) if excess_returns.std() > 0 else np.nan
        else:
            info_ratio = np.nan

        return {
            'Name': name,
            'CAGR': cagr,
            'Volatility': ann_vol,
            'Downside Volatility': downside.std(),
            'Sharpe': sharpe,
            'Sortino': sortino,
            'Max Drawdown': max_dd,
            'Information Ratio': info_ratio
        }

    # Strategy metrics
    for name, data in strategy_data.items():
        twr = calculate_time_weighted_return(data['total_values'], monthly_cash)
        #bench_twr = calculate_time_weighted_return(benchmarks.iloc[:, 0], monthly_cash) if benchmarks is not None else None
        metrics.append(compute_indicators(twr, name, key_benchmark))

    # Benchmark metrics
    if benchmarks is not None:
        for col in benchmarks.columns:
            twr = calculate_time_weighted_return(benchmarks[col], monthly_cash)
            metrics.append(compute_indicators(twr, f"Benchmark: {col}", key_benchmark))

    return pd.DataFrame(metrics).set_index("Name").round(4)