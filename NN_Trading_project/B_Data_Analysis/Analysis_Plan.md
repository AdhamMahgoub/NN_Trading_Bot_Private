# Numbered Data Analysis Plan For Weekly 1.5% Strategy

## Summary
This plan updates `B_Data_Analysis` into a numbered, analysis-only workflow for studying whether a stock can reach a `+1.5%` target within 5 trading sessions while using a simple low-risk stop rule.

Scope stays inside `NN_Trading_project/B_Data_Analysis`. No prediction model, model training, `D_Models`, or dataset-building implementation is included until this plan is approved.

## 0. Investment Objective And Analysis Rules

### 0.1 Objective
- Target return: `+1.5%`.
- Holding period: 5 trading sessions.
- Entry mode: daily entries, where each row is treated as a possible entry at `Open[t]`.
- Exit rule: trade can be considered successful if the target is reached within the 5-session window.
- Risk profile: low risk.

### 0.2 Primary And Secondary Outcomes
- Primary label: `target_before_stop_5d`.
  - `1` if price reaches `Open[t] * 1.015` before touching `Open[t] * 0.985`.
  - `0` otherwise.
- Secondary outcomes:
  - `target_hit_5d`: price reaches `+1.5%` at any point in the 5-session window.
  - `max_forward_return_5d`: best high-based return during the window.
  - `min_forward_drawdown_5d`: worst low-based drawdown during the window.
  - `close_return_5d`: close-to-entry return at the end of the window.
  - `days_to_target`: number of sessions needed to reach the target, if reached.
- If target and stop are both touched in the same daily candle, count it as failure because intraday order is unknown from daily OHLCV data.

## 1. Exploratory Data Analysis (EDA)

### 1.1 Dataset Overview
- Load all CSV files from `A_Data_Gathering/dataset/stocks`.
- Handle the extra yfinance ticker row that appears after the header.
- Print:
  - Number of tickers loaded.
  - Total rows.
  - Rows per ticker.
  - Start and end date per ticker.
  - Overall date range.
  - Available OHLCV columns.

### 1.2 Data Quality Check
- Check missing values in `Date`, `Open`, `High`, `Low`, `Close`, `Adj Close`, and `Volume`.
- Check duplicate dates per ticker.
- Check invalid rows:
  - Non-positive prices.
  - Negative volume.
  - `High < Low`.
  - `Open`, `Close`, `High`, or `Low` outside the candle range.
- Report clean tickers and tickers with issues.

### 1.3 Ticker And Sector Mapping
- Load `B_Data_Analysis/sector_map.json`.
- Print which stock belongs to which sector.
- Check that every loaded ticker has a sector.
- Mark unknown or missing sector values as `Other`.
- Summarize number of tickers per sector.

### 1.4 Simple Return And Risk Statistics
- Calculate weekly returns for each stock.
- Keep this section simple and remove Sharpe, skewness, and kurtosis.
- Use:
  - Total return.
  - Mean weekly return.
  - Weekly return volatility.
  - Max drawdown.
  - Positive-week rate.
  - `target_hit_5d` rate.
  - `target_before_stop_5d` rate.
  - Median `max_forward_return_5d`.
  - Median `min_forward_drawdown_5d`.

## 2. Technical Analysis

### 2.1 Moving Average Trend View
- Plot close price with simple moving averages and exponential moving averages.
- Use easy-to-read windows such as 5, 20, and 50 trading days.
- Add simple trend fields:
  - Close above/below SMA 20.
  - Close above/below SMA 50.
  - Distance from SMA 20.
  - Distance from EMA 20.

### 2.2 Simple Momentum Indicators
- Keep the technical indicators understandable.
- Include:
  - RSI 14.
  - MACD summary only if presented simply.
  - Bollinger Band width.
  - Bollinger Band position.
- Avoid complicated support/resistance and MACD-heavy chart sections.

### 2.3 Simple Breakout Signals
- Identify simple recent-high breakouts:
  - Close above prior 20-day high.
  - Close near prior 20-day high.
  - Distance from prior 20-day high.
- Use these as analysis features and simple visual diagnostics.

## 3. Market And Sector Analysis

### 3.1 Sector Return And Risk Profile
- Group stocks by sector using `sector_map.json`.
- Keep the risk profile simple and sector-based.
- Compare each sector by:
  - Average total return.
  - Mean weekly return.
  - Weekly volatility.
  - Max drawdown.
  - `target_hit_5d` rate.
  - `target_before_stop_5d` rate.
  - Median forward drawdown.

### 3.2 Sector Leaders And Laggards
- Identify strongest and weakest sectors by:
  - Mean weekly return.
  - Target-before-stop success rate.
  - Drawdown risk.
- Identify strongest and weakest tickers within each sector.

### 3.3 Stock Correlation Matrix
- Calculate correlation between stocks using daily returns.
- Plot a stock-level correlation heatmap.
- Highlight highly correlated pairs.

### 3.4 Sector Correlation Matrix
- Aggregate daily returns by sector.
- Calculate correlation between sectors.
- Plot a sector-level correlation heatmap.
- Use this to understand market-wide comovement and diversification risk.

## 4. Pattern Recognition

### 4.1 Simple Breakout Pattern Analysis
- Keep only simple breakout analysis for now.
- Analyze:
  - Breakout above recent high.
  - Breakdown below recent low.
  - Breakout frequency by ticker.
  - Breakout frequency by sector.
  - Whether breakout days have better `target_before_stop_5d` rates.

### 4.2 Pattern Visualizations
- Plot a small number of readable examples.
- Show close price, recent high/low levels, and breakout markers.
- Avoid complex chart patterns such as head-and-shoulders or double tops/bottoms in this phase.

## 5. Target And Label Analysis

### 5.1 Label Creation
- Build the target labels for every valid ticker/date row.
- Use only future OHLCV data for labels, never for features.
- Drop rows near the end of each ticker where a full 5-session forward window is not available.
- Track `valid_forward_window` for rows with enough future data to judge the label.
- Track `valid_backward_window` for rows with enough historical data to compute all features.
- Track `valid_analysis_window` for rows that satisfy both conditions.

### 5.2 Label Prevalence
- Report:
  - Overall `target_hit_5d` rate.
  - Overall `target_before_stop_5d` rate.
  - Rates by ticker.
  - Rates by sector.
  - Rates by volatility bucket.
  - Rates by breakout condition.

### 5.3 Low-Risk Sector Summary
- Create a sector-level risk summary using:
  - Target-before-stop success rate.
  - Median forward drawdown.
  - Worst forward drawdown percentile.
  - Positive-week rate.
  - Number of valid samples.
- Use this as the main low-risk summary instead of a complicated portfolio risk model.

## 6. Feature Engineering And Correlation Analysis

### 6.1 Analysis Feature Table
- Build one analysis table with one row per `Date` plus `Ticker`.
- Include identity columns:
  - `Date`
  - `Ticker`
  - `Sector`
- Include outcome columns from section 0.2.

### 6.2 Feature Set
- Include these analysis features:
  - 5-day return.
  - 20-day return.
  - Moving-average distances.
  - RSI 14.
  - MACD summary if kept simple.
  - Bollinger Band width.
  - Bollinger Band position.
  - Volume change.
  - Relative volume.
  - Dollar volume.
  - Realized volatility.
  - ATR-style range metric.
  - Gap up/down behavior.
  - Recent-high breakout fields.

### 6.3 Correlation Maps
- Use `valid_analysis_window` rows for feature correlations, so each row has both complete feature history and a valid future label.
- Generate a feature-to-label correlation heatmap against `target_before_stop_5d`.
- Generate a secondary correlation table against:
  - `target_hit_5d`
  - `max_forward_return_5d`
  - `min_forward_drawdown_5d`
  - `close_return_5d`
- Generate a feature-to-feature correlation heatmap to identify redundant indicators.

### 6.4 Interpretation Notes
- Treat correlation as an exploratory signal, not proof of predictability.
- Highlight features with stronger relationship to success and risk.
- Highlight features that are highly redundant with each other.
- Keep final notes focused on data-driven decision-making, not profit promises.

## 7. Expected Outputs
- Dataset quality summary.
- Ticker-to-sector mapping summary.
- Primary target-before-stop outcome pie chart.
- Sector universe pie chart.
- Weekly return and simple risk summary.
- Ticker-level risk map showing target-before-stop rate vs median forward drawdown.
- Sector return and risk profile.
- Sector risk map showing target-before-stop rate vs weekly volatility.
- Stock correlation heatmap.
- Sector correlation heatmap.
- Simple moving-average and indicator visuals.
- Technical indicator distribution chart.
- Simple breakout analysis visuals.
- Breakout outcome chart by breakout flag.
- Target/stop label prevalence report.
- Label prevalence chart by volatility bucket.
- Feature-to-label correlation heatmap.
- Feature-to-feature correlation heatmap.
- Selected feature distribution chart.
- Top feature-correlation chart.

## 8. Validation Checks Before Implementation
- Confirm all CSV files load correctly despite the extra yfinance ticker row.
- Confirm `sector_map.json` covers all loaded tickers or reports missing sectors.
- Validate label logic for:
  - Target reached before stop.
  - Stop reached before target.
  - Neither target nor stop reached.
  - Target and stop touched in the same daily candle.
  - Not enough future rows.
- Confirm all features use only information available at entry time.
- Confirm volume pattern analysis is not reintroduced as a standalone section.
- Confirm Sharpe, skewness, and kurtosis are removed from section `1.4`.
- Confirm notebook/code implementation waits for approval after this plan update.

## 9. Assumptions
- First implementation targets the current US stock dataset.
- ETFs can be added later if matching CSV files are added to the same data structure.
- Earnings, news, and sentiment are optional future enrichments and are not part of the first implementation.
- The analysis is decision-support only and does not promise profit.
