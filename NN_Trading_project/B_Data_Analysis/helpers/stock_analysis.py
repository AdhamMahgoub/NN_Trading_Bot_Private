"""
Trading Data Analysis Implementation
Analyzing stock dataset with OHLCV data from 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Define paths
DATASET_PATH = "/scratch/temp_/NN_Trading_Bot_Private/NN_Trading_project/A_Data_Gathering/dataset/stocks"
OUTPUT_PATH = "/scratch/temp_/NN_Trading_Bot_Private/NN_Trading_project/B_Data_Analysis"

def load_stock_data():
    """Load all stock data from CSV files"""
    stock_data = {}

    print("Loading stock data...")
    for file in os.listdir(DATASET_PATH):
        if file.endswith('.csv'):
            symbol = file.split('.')[0]
            try:
                df = pd.read_csv(os.path.join(DATASET_PATH, file))
                # Remove header row (all 'AGCO' values)
                df = df.iloc[1:].copy()

                # Convert columns to appropriate types
                df['Date'] = pd.to_datetime(df['Date'])
                numeric_cols = ['Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']
                for col in numeric_cols:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

                # Set Date as index
                df.set_index('Date', inplace=True)
                df.sort_index(inplace=True)

                stock_data[symbol] = df
                print(f"Loaded {symbol}: {len(df)} days of data")

            except Exception as e:
                print(f"Error loading {symbol}: {e}")

    return stock_data

def check_data_quality(stock_data):
    """Check for missing values and data quality issues"""
    print("\n" + "="*50)
    print("DATA QUALITY CHECK")
    print("="*50)

    quality_report = {}

    for symbol, df in stock_data.items():
        # Check missing values
        missing_data = df.isnull().sum()

        # Check for duplicate dates
        duplicate_dates = df.index.duplicated().sum()

        # Check data range
        date_range = (df.index.min(), df.index.max())

        quality_report[symbol] = {
            'missing_values': missing_data.to_dict(),
            'duplicate_dates': duplicate_dates,
            'date_range': date_range,
            'total_days': len(df)
        }

        # Report any issues
        if missing_data.sum() > 0:
            print(f"\n{symbol}: Missing values found")
            for col, count in missing_data.items():
                if count > 0:
                    print(f"  {col}: {count} missing")

        if duplicate_dates > 0:
            print(f"{symbol}: {duplicate_dates} duplicate dates found")

        print(f"{symbol}: Data from {date_range[0].date()} to {date_range[1].date()} ({len(df)} days)")

    return quality_report

def calculate_daily_returns(stock_data):
    """Calculate daily returns for each stock"""
    print("\n" + "="*50)
    print("CALCULATING DAILY RETURNS")
    print("="*50)

    returns_data = {}

    for symbol, df in stock_data.items():
        # Use 'Close' price for returns
        df['Daily_Return'] = df['Close'].pct_change()
        df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))

        returns_data[symbol] = df

        # Print basic return statistics
        mean_return = df['Daily_Return'].mean() * 252  # Annualize
        std_return = df['Daily_Return'].std() * np.sqrt(252)  # Annualized volatility

        print(f"{symbol}:")
        print(f"  Mean Daily Return: {mean_return:.2%}")
        print(f"  Annualized Volatility: {std_return:.2%}")
        print(f"  Total Return: {(df['Close'].iloc[-1] / df['Close'].iloc[0] - 1):.2%}")

    return returns_data

def basic_statistics_analysis(stock_data):
    """Compute statistical measures for each stock"""
    print("\n" + "="*50)
    print("BASIC STATISTICS ANALYSIS")
    print("="*50)

    stats_summary = {}

    for symbol, df in stock_data.items():
        # Focus on price and volume data
        price_stats = df[['Close', 'High', 'Low', 'Open']].describe()
        volume_stats = df['Volume'].describe()

        stats_summary[symbol] = {
            'price_stats': price_stats,
            'volume_stats': volume_stats
        }

        print(f"\n{symbol} Price Statistics:")
        print(f"  Mean: ${price_stats.loc['mean', 'Close']:.2f}")
        print(f"  Std: ${price_stats.loc['std', 'Close']:.2f}")
        print(f"  Min: ${price_stats.loc['min', 'Close']:.2f}")
        print(f"  Max: ${price_stats.loc['max', 'Close']:.2f}")
        print(f"  25th Percentile: ${price_stats.loc['25%', 'Close']:.2f}")
        print(f"  75th Percentile: ${price_stats.loc['75%', 'Close']:.2f}")

        print(f"{symbol} Volume Statistics:")
        print(f"  Mean Volume: {volume_stats.loc['mean']:,.0f}")
        print(f"  Median Volume: {volume_stats.loc['50%']:,.0f}")
        print(f"  Max Volume: {volume_stats.loc['max']:,.0f}")

    return stats_summary

def volume_pattern_analysis(stock_data):
    """Analyze volume patterns and trading activity"""
    print("\n" + "="*50)
    print("VOLUME PATTERN ANALYSIS")
    print("="*50)

    volume_analysis = {}

    for symbol, df in stock_data.items():
        # Calculate volume metrics
        df['Volume_MA_10'] = df['Volume'].rolling(window=10).mean()
        df['Volume_MA_30'] = df['Volume'].rolling(window=30).mean()
        df['Volume_Spike'] = df['Volume'] > df['Volume_MA_10'] * 1.5

        # Calculate average volume on spike days
        spike_days = df['Volume_Spike'].sum()
        avg_spike_volume = df[df['Volume_Spike']]['Volume'].mean()
        avg_normal_volume = df[~df['Volume_Spike']]['Volume'].mean()

        volume_analysis[symbol] = {
            'spike_days': spike_days,
            'avg_spike_volume': avg_spike_volume,
            'avg_normal_volume': avg_normal_volume,
            'volume_spike_ratio': avg_spike_volume / avg_normal_volume if avg_normal_volume > 0 else 0
        }

        print(f"\n{symbol}:")
        print(f"  Volume spike days: {spike_days}")
        print(f"  Avg spike volume: {avg_spike_volume:,.0f}")
        print(f"  Avg normal volume: {avg_normal_volume:,.0f}")
        print(f"  Spike ratio: {volume_analysis[symbol]['volume_spike_ratio']:.2f}")

    return volume_analysis

def plot_price_trends(stock_data, top_n=10):
    """Plot price trends for top N stocks by market cap"""
    print("\n" + "="*50)
    print("PLOTTING PRICE TRENDS")
    print("="*50)

    # Calculate market cap (using Close price and volume as proxy)
    market_caps = {}
    for symbol, df in stock_data.items():
        # Use average price * average volume as market cap proxy
        avg_price = df['Close'].mean()
        avg_volume = df['Volume'].mean()
        market_caps[symbol] = avg_price * avg_volume

    # Get top N stocks by market cap
    top_stocks = sorted(market_caps.items(), key=lambda x: x[1], reverse=True)[:top_n]

    # Create subplot
    fig, axes = plt.subplots(2, 5, figsize=(20, 10))
    axes = axes.flatten()

    for i, (symbol, _) in enumerate(top_stocks):
        ax = axes[i]
        df = stock_data[symbol]

        # Plot closing price
        ax.plot(df.index, df['Close'], label='Close Price', linewidth=1.5)

        # Plot 20-day moving average
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        ax.plot(df.index, df['MA_20'], label='MA 20', alpha=0.7)

        ax.set_title(f'{symbol}', fontsize=10)
        ax.set_ylabel('Price ($)', fontsize=8)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=8)

        # Rotate x-axis labels
        plt.setp(ax.get_xticklabels(), rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, 'price_trends_top10.png'), dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Saved price trends plot for top {top_n} stocks")

def main():
    """Main analysis function"""
    print("Starting Trading Data Analysis...")

    # Load all stock data
    stock_data = load_stock_data()

    # Check data quality
    quality_report = check_data_quality(stock_data)

    # Calculate daily returns
    returns_data = calculate_daily_returns(stock_data)

    # Basic statistics
    stats_summary = basic_statistics_analysis(stock_data)

    # Volume pattern analysis
    volume_analysis = volume_pattern_analysis(stock_data)

    # Plot price trends
    plot_price_trends(stock_data)

    print("\n" + "="*50)
    print("ANALYSIS COMPLETE")
    print("="*50)
    print("Check the B_Data_Analysis folder for generated plots!")

if __name__ == "__main__":
    main()