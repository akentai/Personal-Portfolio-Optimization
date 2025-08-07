# ​ Personal Portfolio Optimization

A modular Python framework to test and apply portfolio optimization strategies for **monthly investments** using **historical data**, **mathematical optimization**, and **backtesting**. Built for **individual investors** and **aspiring quants**.

---

## ​ Motivation

Many of us set aside money each month — often letting it sit idle in a bank account, slowly eroded by inflation. This project proposes a more conscious alternative: **systematic, low-effort monthly investing** based on **optimization techniques** used in quantitative finance.

We evaluate a range of **portfolio optimization strategies**, helping users:

- Determine **how much to allocate** to each instrument (stocks/ETFs) every month.
- Test the behavior of strategies over time using **real historical data**.
- Compare performance with popular **benchmarks**:
  - Saving cash  
  - Keeping money in a bank (risk-free rate)  
  - Investing fully in the S&P 500  
  - Applying the 3-ETF rule (e.g., S&P 50%, QQQ - Tech ETF 40%, VYM - Dividen ETF 10%)

### ​ Assumptions

1. **Fractional Shares**  
   The framework assumes you can buy fractional shares (e.g., via Interactive Brokers, not Degiro). This allows precise allocation without rounding.

2. **Monthly Deposits**  
   You invest a fixed amount every month (default: \$1,000–\$10,000). The simulator handles the compounding and injection of capital.

3. **Buy-Only Rebalancing**  
   We do **not re-allocate the full portfolio each month**. Instead, we **buy additional shares** of selected instruments — minimizing effort and transaction costs.  
   > Yearly full reallocation is a planned feature.

4. **Transaction Costs Ignored**  
   Transaction fees are excluded. Based on Interactive Brokers, costs are around **€1.75 per instrument**, which is negligible for larger monthly investments.

5. **Dollar Analysis**
   Everything is in dollars since that is what 'yfinance' api provides.

---

##  Literature & GitHub Repos

Here are some noteworthy open-source projects with complementary features or inspiration:

- **[PyPortfolioOpt](https://github.com/robertmartin8/PyPortfolioOpt)**  
  Implements portfolio optimization techniques including mean-variance, Black-Litterman, Hierarchical Risk Parity, shrinkage, and efficient frontiers.  
  **Pros**: Rich, well-documented. **Cons**: Focuses on static optimizations, fewer buy-only flow scenarios.  

- **[Riskfolio-Lib](https://github.com/dcajasn/Riskfolio-Lib)**  
  Built atop CVXPY and pandas, it offers strategic asset allocation, performance analysis, and risk decomposition.  
  **Pros**: Advanced analytics. **Cons**: More academic focus, less tailored to monthly incremental investing.  

- **[skfolio](https://github.com/skfolio/skfolio)**  
  Portfolio optimization and risk management framework compatible with scikit-learn.  
  **Pros**: Pipeline integration, cross-validation support. **Cons**: Less focus on buy-only or periodic injection logic.  

- **[PyroQuant/Portfolio‑Optimizer](https://github.com/PyroQuant/Portfolio-Optimizer)**  
  A simple script demonstrating Sharpe, CVaR, Sortino, and variance-based optimizations using Yahoo Finance data.  
  **Pros**: Lightweight and straightforward. **Cons**: Lacks modular strategy design and backtesting capabilities.  

Other useful resources for algorithms and tools include the **[awesome-quant](https://github.com/wilsonfreitas/awesome-quant)** list for many frameworks, and **FinRL** for reinforcement learning (though overkill for low-frequency investing).  

---

## ​​ Repository Structure

```
.
├── monthly_optimization.py       # To be used every month
├── strategies/                   # Individual strategy classes
├── backtesting/                  # Backtester class
├── data/                         # DataLoader (uses Yahoo Finance) and benchmarks
├── evaluation/                   # Visualization and performance (TBD)
├── requirements.txt
└── README.md
```

Strategies

| Strategy Type                 | Description                                 |
|------------------------------|----------------------------------------------|
| Mean-Variance Optimization   | Markowitz efficient frontier                 | 
| Risk Parity                  | Equalizes risk contributions                 | 
| Max Sharpe / Sortino         | Risk-adjusted return maximization            |
| CVaR Optimization            | Focuses on downside risk                     | 
| Black-Litterman              | Combines priors and investor views           | 
| Equal Weight                 | Equal allocation across assets               |
| Momentum-based               | Allocates to recent winners                  | 
| Value Averaging              | Targets specific growth paths                | 

---

## 4. 🚀 How to Use This Project

### A. 🧪 Compare Strategies — `run_multiple_strategies.py`

- Choose your tickers
- Backtest a list of strategies
- Compare results with benchmarks like:
  - Cash
  - Risk-free (bank interest)
  - SPY (S&P 500)
  - 3-ETF Rule (e.g., SPY, QQQ, VYM)

---

### B. 🔍 Analyze a Single Strategy — `run_single_strategy.py`

- Test a single strategy (e.g., Markowitz MVO)
- Visualize how it allocates capital
- Compare its performance to cash/SPY/ETF benchmarks

---

### C. 📅 Monthly Optimization Script — `monthly_optimization.py`

Use this lightweight script to **get your monthly allocation** using updated prices and your chosen strategy.

```python
from data import DataLoader
from strategies import MomentumStrategy

tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
loader = DataLoader(tickers, start='2010-01-01', interval='1mo')
prices = loader.fetch_prices()

strategy = MomentumStrategy(tickers)
df = strategy.optimize(
    current_portfolio=[0]*len(tickers),
    new_capital=2000,
    price_history=prices,
    returns_history=prices.pct_change().dropna()
)
print(df.round(1))
