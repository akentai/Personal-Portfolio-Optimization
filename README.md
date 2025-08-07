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
  - Applying the 3-ETF rule (e.g., SPY, QQQ, VYM)

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

---

## ​ Related Work

| Strategy Type                 | Description                                  | Pros                                | Cons                                |
|------------------------------|----------------------------------------------|-------------------------------------|-------------------------------------|
| Mean-Variance Optimization   | Markowitz efficient frontier                 | Simple, well-studied                | Sensitive to input estimation       |
| Risk Parity                  | Equalizes risk contributions                 | Robust to assumptions               | May underperform in bull markets    |
| Max Sharpe / Sortino         | Risk-adjusted return maximization            | Captures return/risk balance        | Requires accurate input estimation  |
| CVaR Optimization            | Focuses on downside risk                     | Tail-risk aware                     | Slower to compute                   |
| Black-Litterman              | Combines priors and investor views           | Flexible, realistic                 | Requires good view specification    |
| Equal Weight                 | Equal allocation across assets               | Simple and naive baseline           | Ignores return/risk differences     |
| Momentum-based               | Allocates to recent winners                  | Captures trends                     | Can overfit or chase noise          |
| Value Averaging              | Targets specific growth paths                | Disciplined contribution strategy   | Ignores recent market changes       |

---

##  Literature & GitHub Repos

Here are some noteworthy open-source projects with complementary features or inspiration:

- **[PyPortfolioOpt](https://github.com/robertmartin8/PyPortfolioOpt)**  
  Implements portfolio optimization techniques including mean-variance, Black-Litterman, Hierarchical Risk Parity, shrinkage, and efficient frontiers.  
  **Pros**: Rich, well-documented. **Cons**: Focuses on static optimizations, fewer buy-only flow scenarios.  
  :contentReference[oaicite:1]{index=1}

- **[Riskfolio-Lib](https://github.com/dcajasn/Riskfolio-Lib)**  
  Built atop CVXPY and pandas, it offers strategic asset allocation, performance analysis, and risk decomposition.  
  **Pros**: Advanced analytics. **Cons**: More academic focus, less tailored to monthly incremental investing.  
  :contentReference[oaicite:2]{index=2}

- **[skfolio](https://github.com/skfolio/skfolio)**  
  Portfolio optimization and risk management framework compatible with scikit-learn.  
  **Pros**: Pipeline integration, cross-validation support. **Cons**: Less focus on buy-only or periodic injection logic.  
  :contentReference[oaicite:3]{index=3}

- **[PyroQuant/Portfolio‑Optimizer](https://github.com/PyroQuant/Portfolio-Optimizer)**  
  A simple script demonstrating Sharpe, CVaR, Sortino, and variance-based optimizations using Yahoo Finance data.  
  **Pros**: Lightweight and straightforward. **Cons**: Lacks modular strategy design and backtesting capabilities.  
  :contentReference[oaicite:4]{index=4}

Other useful resources for algorithms and tools include the **[awesome-quant](https://github.com/wilsonfreitas/awesome-quant)** list for many frameworks, and **FinRL** for reinforcement learning (though overkill for low-frequency investing).  
:contentReference[oaicite:5]{index=5}

---

## ​​ Repository Structure

```text
.
├── main.py                        # Entry scripts
├── benchmarks/                   # Cash, SPY, Risk-Free, ETFs
├── strategies/                   # Individual strategy classes
├── backtesting/                  # Backtester class
├── data/                         # DataLoader (uses Yahoo Finance)
├── evaluation/                   # Visualization and performance (TBD)
├── monthly_optimization.py       # One-step allocation
├── requirements.txt
└── README.md
