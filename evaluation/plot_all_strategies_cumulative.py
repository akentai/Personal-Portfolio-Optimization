import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_all_strategies_cumulative(
    strategy_data: dict,
    initial_capital: float,
    monthly_cash: float,
    benchmarks: pd.DataFrame = None
):
    """
    Compare N strategies + optional benchmarks on cumulative‐return evolution.

    Parameters
    ----------
    backtest_results : dict
    initial_capital : float
    monthly_cash : float
    benchmarks : pd.DataFrame, optional
        same index as total_values; columns are benchmark value series
        (e.g. 'Cash','Risk Free','SPY')
    """
    # pick up dates from the first strategy
    first = next(iter(strategy_data))
    dates = strategy_data[first]['total_values'].index
    n     = len(dates)
    t     = np.arange(n)
    invested = initial_capital + monthly_cash * (t + 1)
    total_invested = invested[-1]

    fix, ax = plt.subplots(figsize=(16, 8))

    for name, strategy in strategy_data.items():
        total_vals = strategy['total_values']
        cum = total_vals / invested - 1
        ax.plot(
            dates,
            100 * cum,
            label=f"{name} ({cum.iloc[-1]:.2%})"
        )

    if benchmarks is not None:
        for bench in benchmarks.columns:
            series   = benchmarks[bench].reindex(dates)
            bench_cum = series / invested - 1
            ax.plot(dates,
                    bench_cum * 100,
                    linestyle='--',
                    label=f"{bench} ({bench_cum.iloc[-1]:.2%})")

    ax.set_title("Cumulative Return Comparison (%)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return (%)")
    ax.legend(loc='upper left', bbox_to_anchor=(1,1))
    ax.grid(True)
    plt.tight_layout()
    plt.show()