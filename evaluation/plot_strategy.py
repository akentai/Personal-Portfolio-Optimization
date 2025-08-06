import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_strategy(
    strategy_data: dict,
    initial_capital: float,
    monthly_cash: float,
    benchmarks: pd.DataFrame = None
):
    """
    4-panel plot:
      1) Portfolio & component values
      2) Allocation evolution
      3) Weight evolution
      4) Cumulative return evolution vs. benchmarks
    """
    asset_vals   = strategy_data['asset_values']
    allocations  = strategy_data['allocations']
    weights      = strategy_data['weights']
    total_vals   = strategy_data['total_values']
    dates        = total_vals.index

    n = len(dates)
    t = np.arange(n)
    invested = initial_capital + monthly_cash * (t + 1)
    total_invested = invested[-1]

    fig, axes = plt.subplots(4, 1, figsize=(9, 14), sharex=True)

    # 1) Portfolio & component values
    ax = axes[0]
    cum_ret_port = total_vals.iloc[-1] / total_invested - 1
    ax.plot(dates, total_vals, label=f'Portfolio ({cum_ret_port:.2%})', linewidth=2)
    for asset in asset_vals.columns:
        ax.plot(dates, asset_vals[asset], label=asset)
    if benchmarks is not None:
        for bench in benchmarks.columns:
            series = benchmarks[bench].reindex(dates)
            cum_ret_b = series.iloc[-1] / total_invested - 1
            ax.plot(dates, series, linestyle='--', label=f'{bench} ({cum_ret_b:.2%})')
    ax.set_title('Portfolio & Component Values')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.grid(True)

    # 2) Allocation evolution
    ax = axes[1]
    for asset in allocations.columns:
        mean_alloc = allocations[asset].mean()
        ax.plot(dates, allocations[asset], label=f'{asset} (mean: {mean_alloc:.2f})')
    ax.set_title('Allocation Evolution')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.grid(True)

    # 3) Weight evolution
    ax = axes[2]
    for asset in weights.columns:
        mean_wt = weights[asset].mean()
        ax.plot(dates, weights[asset], label=f'{asset} (mean: {mean_wt:.2f})')
    ax.set_title('Weight Evolution')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.grid(True)

    # 4) Cumulative return evolution
    ax = axes[3]
    port_cum = total_vals / invested - 1
    ax.plot(dates,
            100 * port_cum,
            label=f'Portfolio ({port_cum.iloc[-1]*100:.2f}%)',
            linewidth=2)

    # Benchmarks
    if benchmarks is not None:
        for bench in benchmarks.columns:
            series = benchmarks[bench].reindex(dates)
            bench_cum = series / invested - 1
            ax.plot(dates,
                    100 * bench_cum,
                    linestyle='--',
                    label=f'{bench} ({bench_cum.iloc[-1]*100:.2f}%)')
    ax.set_title('Cumulative Return Evolution (%)')
    ax.set_ylabel('Cumulative Return')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.grid(True)

    # final tweaks
    axes[-1].set_xlabel('Date')
    plt.tight_layout()
    plt.show()
