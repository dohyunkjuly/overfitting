# Optimizing an SMA Crossover (Hourly BTC)
# ============
#
# `optimize` runs a strategy once for every combination of parameter values
# and reports the best one. This example tunes the two moving-average lengths
# of the SMA crossover from `sma_cross.py`.
#
# Steps:
# 1. Declare the parameters as class attributes.
# 2. Run a grid search and read the result.
# 3. Look at every combination tried, as a table and a heatmap.
# 4. Try a random search and a custom objective.
# 5. Inspect the winning strategy like any other backtest.

# +
import matplotlib.pyplot as plt
import pandas as pd
from overfitting import Strategy, optimize
from overfitting.indicators import sma, crossover, crossunder

df = pd.read_csv('./data/BTCUSDT.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
df = df.set_index('timestamp').loc['2023-01-01':]
data = {"BTC": df}
# -

# **1. Parameters are class attributes.**
#
# `fast = 20` and `slow = 50` are the defaults used when the strategy runs on
# its own. `optimize` replaces them for each combination it tries. Read them
# as `self.fast` and `self.slow`.

# +
class SmaCross(Strategy):
    fast = 20
    slow = 50

    def init(self):
        self.set_leverage("BTC", 1)
        close = self.data["BTC"].close
        self.fast_ma = self.I(sma, close, self.fast)
        self.slow_ma = self.I(sma, close, self.slow)

    def next(self):
        position = self.get_position("BTC")
        if crossover(self.fast_ma, self.slow_ma) and position.qty == 0:
            price = self.price("BTC")
            self.limit_order("BTC", self.get_balance() // price, price)
        elif crossunder(self.fast_ma, self.slow_ma) and position.qty > 0:
            self.market_order("BTC", -position.qty)
# -

# **2. Grid search.**
#
# Every combination of `fast` and `slow` is tried, except those the
# `constraint` rejects (a fast average longer than the slow one makes no
# sense). `train` picks the data the search uses: here 2023, given as a date
# range (the end date is not included). It can also be a share of the most
# recent data, such as `train=0.6`; by default the search uses all the data.
# The winner is then backtested over all the data. Any `Strategy` argument,
# such as `slippage` or `commission_rate`, can be passed and applies to every run.

result = optimize(
    SmaCross, data,
    params={"fast": range(5, 55, 5), "slow": range(20, 220, 20)},
    maximize="sharpe_ratio",
    constraint=lambda p: p.fast < p.slow,
    train=("2023-01-01", "2024-01-01"),
    slippage=0.001,
)
print(result)

# The fields of the result:

print(result.best_params)   # the winning combination
print(result.trials)        # how many combinations were tried
print(result.train_start, result.train_end)   # the training window
result.metrics              # the winner's metrics over all the data

# **3. Every combination tried.**
#
# `result.table` has one row per combination, best first, with its metrics on
# the training window. `objective` is the value that was maximized.

result.table.head(10)

# `result.heatmap(x, y)` turns the table into a grid. A good parameter choice
# sits in a broad area of good values; a lone bright cell surrounded by poor
# ones is more likely luck than an edge.

# +
grid = result.heatmap("fast", "slow", metric="sharpe_ratio")

fig, ax = plt.subplots(figsize=(8, 6))
image = ax.imshow(grid.values, cmap="RdBu", vmin=-abs(grid).max().max(),
                  vmax=abs(grid).max().max(), origin="lower", aspect="auto")
ax.set_xticks(range(len(grid.columns)), grid.columns)
ax.set_yticks(range(len(grid.index)), grid.index)
ax.set_xlabel("fast")
ax.set_ylabel("slow")
ax.set_title("Sharpe ratio on the training window")
fig.colorbar(image, ax=ax)
plt.show()
# -

# **4. Random search and a custom objective.**
#
# With many parameters the grid grows fast. `method="random"` tries `n`
# combinations picked at random (`seed` makes it repeatable).
#
# `maximize` can also be a function of the metrics. Here: return per unit of
# drawdown (CAGR divided by the maximum drawdown), but only for combinations
# that traded at least 20 times.

# +
def return_over_drawdown(m):
    if m["fills"] < 20:
        return None                     # never wins
    return m["cagr"] / abs(m["max_drawdown"])

random_result = optimize(
    SmaCross, data,
    params={"fast": range(5, 100), "slow": range(20, 400, 5)},
    constraint=lambda p: p.fast < p.slow,
    method="random", n=200, seed=42,
    maximize=return_over_drawdown,
    slippage=0.001,
)
print(random_result)
# -

# **5. The winning strategy.**
#
# `result.strategy` is the winner, run over all the data with recording on, so the usual report, charts and trade tables work.

result.strategy.plot(result.returns)

result.strategy.fetch_round_trips().head()
