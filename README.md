# ML Trading Strategy — Random Forest Signal Generation from Technical Indicators

A rule-based trading strategy and a Random Forest classification learner, both driven by five
hand-implemented technical indicators, evaluated against a buy-and-hold benchmark on JPM across
separate in-sample and out-of-sample periods.

> **Note on source code.** This repository contains a technical writeup only. The implementation was
> built for a graduate course, and the course's academic policy prohibits publishing the code
> publicly, so no source files, assignment text, or course materials are included here. Happy to
> walk through the implementation and design decisions in detail on a call.

**Stack:** Python · Pandas · NumPy · Matplotlib

---

## Problem

Given daily adjusted-close price data for a single equity, decide each trading day whether to hold
1,000 shares long, 1,000 shares short, or no position. Compare two approaches — a hand-tuned rule
system and a supervised learner — against a benchmark that simply buys 1,000 shares on day one and
sells on the last day.

**Constraints.** $100,000 starting cash, unlimited leverage, buy/sell only, exactly three allowable
positions (+1000, 0, −1000 shares). Position changes trade the full delta, so flipping from short to
long is a 2,000-share order.

**Periods.** In-sample 2008-01-01 to 2009-12-31. Out-of-sample 2010-01-01 to 2011-12-31. Ticker: JPM.
The in-sample window spans the 2008 financial crisis, which matters for interpreting the results
below — JPM was extraordinarily volatile in that window.

---

## Indicators

Five indicators were implemented from raw price data rather than pulled from a library. Each is
evaluated independently on its own natural scale — no normalization was applied, since each was
compared against thresholds meaningful in its own units.

| Indicator | Window | What it measures | Buy threshold | Sell threshold |
|---|---|---|---|---|
| **Bollinger %B** | 20-day SMA, ±2σ | Price position within the Bollinger band envelope; overbought/oversold | < 0.15 | > 0.85 |
| **Momentum** | 10-day | Rate of price change; trend direction and strength | > 0.02 | < −0.02 |
| **MACD histogram** | 12/26-day EMA, 9-day signal | Divergence between short- and long-term EMAs; reversal pressure | > 0.01 | < −0.01 |
| **RSI** | 14-day | Average gain vs. average loss; overbought/oversold on a 0–100 scale | < 35 | > 60 |
| **CCI** | 20-day | Price deviation from SMA scaled by mean deviation | < −150 | > 150 |

### Threshold design

The thresholds are not textbook defaults; they were tuned so the indicators complement rather than
fight each other.

- **%B and CCI are deliberately desensitized.** Both detect the same broad condition (overbought /
  oversold), so leaving both at conventional thresholds meant they fired together constantly and
  drowned out everything else. Widening them to 0.15/0.85 and ±150 means they only speak on genuinely
  extreme readings, where they act as confirmation rather than as the primary signal.
- **RSI is left comparatively sensitive** (35/60, versus the conventional 30/70) so that one
  mean-reversion voice is active often enough to push back on the trend-following indicators.
- **Momentum and MACD carry small dead zones** (±0.02, ±0.01) rather than triggering on any nonzero
  reading. Without the dead zone both fired on day-to-day noise; with it they fire on moves with some
  persistence behind them.

---

## Signal generation: indicator voting

Each indicator casts a vote of +1 (buy) or −1 (sell) when its previous-day value falls inside a
threshold, and abstains otherwise. Votes are summed:

| Vote sum | Target position |
|---|---|
| ≥ +1 | 1,000 shares long |
| ≤ −1 | 1,000 shares short |
| 0 | hold current position |

The trade executed is the difference between the target and current position, so a short-to-long flip
trades 2,000 shares. Signals are computed from the prior day's close and acted on the following day,
to avoid using same-day information the strategy would not actually have.

---

## Two strategies over the same signals

**Rule-based strategy.** Applies the vote-sum rule above directly, with no learning component.

**Random Forest learner.** A bagging ensemble of random decision trees, converted from regression to
classification by aggregating leaf predictions with the mode rather than the mean.

- **Features:** the five indicator values, each on its own scale
- **Labels:** the trading action produced by the voting system
- **Leaf size:** 10 — small enough to learn structure, large enough to limit overfitting on a
  two-year window
- **Bags:** 1,000 — enough for prediction stability without unreasonable training time

**Design note, stated plainly:** because the labels come from the voting system rather than from
realized forward returns, the learner is trained to approximate the rule-based strategy's decision
boundary, not to discover a mapping from indicators to future price movement. Its advantage over the
rules is therefore one of smoothing and generalization — bagging softens the hard threshold edges
into a probabilistic boundary — rather than the discovery of genuinely new signal. This is a real
limitation of the setup and is why the learner's edge, where it appears, is modest. A stronger
formulation would label on n-day forward return with a transaction-cost-aware threshold.

---

## Results

### Rule-based strategy vs. benchmark

| Metric | Rules (in-sample) | Benchmark (in-sample) | Rules (out-of-sample) | Benchmark (out-of-sample) |
|---|---|---|---|---|
| Cumulative return | **0.0360** | 0.0123 | **−0.0101** | −0.0836 |
| Mean daily return | 0.0002 | 0.0002 | 0.0000 | −0.0001 |
| Std. dev. daily return | 0.0164 | 0.0170 | 0.0078 | 0.0085 |

**In-sample:** the rules returned 3.6% against the benchmark's 1.2%, with marginally lower daily
volatility.

**Out-of-sample:** JPM fell hard and the strategy did not escape it — but it lost 1.0% where
buy-and-hold lost 8.4%, a 7.3 percentage-point improvement, again at lower volatility. Note carefully
that *mean daily returns were nearly identical* between the strategy and the benchmark in both
periods. The cumulative gap comes from avoiding a subset of the worst drawdowns, not from a superior
per-day edge. That is a meaningful distinction and the honest reading of this table.

### Random Forest learner vs. rules vs. benchmark

<p align="center">
  <img src="images/experiment1_in_sample.png" width="640" alt="In-sample portfolio comparison"/>
</p>

In-sample, the learner clearly outperforms both the rules and the benchmark. This is the expected and
largely uninformative result — it is being evaluated on data it trained on, so the gap here measures
fit, not skill.

<p align="center">
  <img src="images/experiment1_out_of_sample.png" width="640" alt="Out-of-sample portfolio comparison"/>
</p>

Out-of-sample is the result that matters. The learner and the rules track each other closely at the
start and end of the window. The learner's advantage opens up around April 2010, when it identifies a
steep, relatively low-volatility downward trend faster than the rules do, and it holds that lead until
volatility spikes late in the period and both strategies lose their edge. The learner is the only one
of the three to finish the out-of-sample period profitable, against a benchmark that lost 8.4% —
though the margin is thin, and on a single ticker over a single two-year window that is weak evidence
of a durable edge.

### Sensitivity to market impact

Market impact models the cost of moving the price against yourself when you trade. The learner was
re-evaluated in-sample across impact values from 0% to 4% of trade value.

<p align="center">
  <img src="images/experiment2_cumreturn.png" width="480" alt="Impact vs cumulative return"/>
  <img src="images/experiment2_sharpe.png" width="480" alt="Impact vs Sharpe ratio"/>
</p>

Cumulative return declines monotonically as impact rises, from roughly +0.5 at zero impact to about
−2.7 at 4%. Sharpe ratio degrades too, though not monotonically — the 3% case comes out better than
2% and 4%, which is a reminder that these are single runs on one ticker and the ordering at
intermediate impact levels is within noise.

The underlying point is structural and worth stating: this strategy trades often, and it flips between
full long and full short positions, so every position change moves 2,000 shares. A strategy with that
turnover profile is highly exposed to transaction costs, and a per-trade edge that looks adequate at
zero impact does not survive realistic friction. Any honest read of these charts is that the strategy
is not viable at meaningful impact levels without reducing turnover.

---

## Limitations

Stating these because they are the first things I would ask about:

1. **One ticker, one asset class, two windows.** Everything here is JPM. Nothing about these results
   generalizes without testing across many symbols and multiple regimes.
2. **The in-sample window is the 2008 crisis.** Thresholds tuned on the most volatile two years in
   recent market history are unlikely to transfer cleanly to calm markets.
3. **Labels are derived from the rule system, not from forward returns** — see the design note above.
   The learner cannot, by construction, discover signal the rules do not already encode.
4. **Thresholds were tuned against the in-sample period**, which means the in-sample results are
   optimistic by an unmeasured amount. The out-of-sample numbers are the only ones worth weighing.
5. **High turnover, cost-sensitive.** The impact experiment shows the strategy failing well before
   realistic institutional cost levels.
6. **No risk management layer.** No position sizing, no stop-losses, no volatility targeting, no
   exposure limits. The strategy is always either fully long, fully short, or flat.
7. **No statistical significance testing.** With one ticker and two two-year windows, there is not
   enough data to distinguish a real edge from luck.

---

## What I would do differently

- Label on n-day forward return net of an estimated cost threshold, so the learner is solving the
  actual prediction problem rather than imitating a rule set
- Walk-forward validation across rolling windows instead of a single fixed train/test split
- Test across 30+ tickers spanning sectors and volatility regimes
- Add a turnover penalty directly to the signal logic, given how badly impact degrades performance
- Check indicator correlation and drop or combine redundant features (%B, RSI, and CCI are all
  measuring closely related conditions)
- Compare against stronger baselines than buy-and-hold — a simple moving-average crossover, at minimum

---

## Contact

**Jason Betsargon** · M.S. Computer Science, Georgia Tech (Machine Learning)
[LinkedIn](https://www.linkedin.com/in/jason-betsargon/) · [Portfolio](https://jasonbet.github.io/homepage/) · jasonbetsargon@gmail.com
