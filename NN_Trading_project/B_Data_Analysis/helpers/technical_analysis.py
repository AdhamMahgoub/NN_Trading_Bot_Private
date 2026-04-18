"""
Technical Analysis Implementation
Moving Average Convergence Divergence (MACD) and other technical indicators
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

# Define paths
DATASET_PATH = "/scratch/temp_/NN_Trading_Bot_Private/NN_Trading_project/A_Data_Gathering/dataset/stocks"
OUTPUT_PATH = "/scratch/temp_/NN_Trading_Bot_Private/NN_Trading_project/B_Data_Analysis"

def load_stock_data():
    """Load stock data (reusing from previous script)"""
    stock_data = {}

    for file in os.listdir(DATASET_PATH):
        if file.endswith('.csv'):
            symbol = file.split('.')[0]
            df = pd.read_csv(os.path.join(DATASET_PATH, file))
            # Remove header row (all values are the symbol name)
            df = df.iloc[1:].copy()

            # Convert columns
            df['Date'] = pd.to_datetime(df['Date'])
            numeric_cols = ['Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            df.set_index('Date', inplace=True)
            df.sort_index(inplace=True)
            stock_data[symbol] = df

    return stock_data

def calculate_macd(df, fast_period=12, slow_period=26, signal_period=9):
    """Calculate MACD indicator"""
    # Calculate EMAs
    exp1 = df['Close'].ewm(span=fast_period, adjust=False).mean()
    exp2 = df['Close'].ewm(span=slow_period, adjust=False).mean()

    # MACD line
    macd_line = exp1 - exp2

    # Signal line
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()

    # Histogram
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram

def support_resistance_levels(df, window=20):
    """Identify support and resistance levels"""
    # Rolling max and min for resistance and support
    resistance = df['High'].rolling(window=window).max()
    support = df['Low'].rolling(window=window).min()

    # Find recent key levels
    recent_resistance = df['High'].tail(window).max()
    recent_support = df['Low'].tail(window).min()

    return resistance, support, recent_resistance, recent_support

def plot_macd_analysis(stock_data, symbols=None):
    """Plot MACD analysis for selected stocks"""
    if symbols is None:
        # Select top 5 performers and bottom 5 performers
        returns = {}
        for symbol, df in stock_data.items():
            total_return = (df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) * 100
            returns[symbol] = total_return

        symbols = sorted(returns.items(), key=lambda x: x[1], reverse=True)[:5]
        symbols = [s[0] for s in symbols]
        symbols.extend(sorted(returns.items(), key=lambda x: x[1])[:5])
        symbols = [s[0] for s in symbols]

    fig, axes = plt.subplots(5, 2, figsize=(16, 20))
    axes = axes.flatten()

    for i, symbol in enumerate(symbols):
        if i >= len(axes):
            break

        print(f"Processing symbol: {symbol}")
        if symbol not in stock_data:
            print(f"Warning: {symbol} not found in stock_data")
            continue

        df = stock_data[symbol]
        ax = axes[i]

        # Calculate MACD
        macd_line, signal_line, histogram = calculate_macd(df)

        # Plot price and moving averages
        ax2 = ax.twinx()

        # Price plot
        ax2.plot(df.index, df['Close'], label='Close Price', color='blue', alpha=0.7)
        ax2.plot(df.index, df['Close'].rolling(window=50).mean(), label='MA 50', color='orange', alpha=0.7)

        # MACD plot
        ax.plot(macd_line.index, macd_line, label='MACD', color='green')
        ax.plot(signal_line.index, signal_line, label='Signal', color='red')
        ax.bar(histogram.index, histogram * 0.5, label='Histogram', color='gray', alpha=0.3)

        ax.set_title(f'{symbol} - MACD Analysis')
        ax.set_ylabel('MACD')
        ax2.set_ylabel('Price ($)')
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        # Format dates
        plt.setp(ax.get_xticklabels(), rotation=45)
        plt.setp(ax2.get_xticklabels(), rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, 'macd_analysis.png'), dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Saved MACD analysis plot for {len(symbols)} stocks")

def plot_support_resistance(stock_data, top_n=5):
    """Plot support and resistance levels"""
    # Get top N stocks by recent performance
    returns = {}
    for symbol, df in stock_data.items():
        total_return = (df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) * 100
        returns[symbol] = total_return

    top_stocks = sorted(returns.items(), key=lambda x: x[1], reverse=True)[:top_n]

    fig, axes = plt.subplots(1, top_n, figsize=(5*top_n, 6))
    if top_n == 1:
        axes = [axes]

    for i, (symbol, _) in enumerate(top_stocks):
        df = stock_data[symbol]
        ax = axes[i]

        # Calculate support/resistance
        resistance, support, recent_resistance, recent_support = support_resistance_levels(df)

        # Plot price
        ax.plot(df.index, df['Close'], label='Close Price', color='black', linewidth=1)

        # Plot rolling support/resistance
        ax.plot(resistance.index, resistance, label='Resistance', color='red', alpha=0.5, linestyle='--')
        ax.plot(support.index, support, label='Support', color='green', alpha=0.5, linestyle='--')

        # Plot recent key levels
        ax.axhline(y=recent_resistance, color='red', linestyle='-', alpha=0.8, label=f'Resistance: ${recent_resistance:.2f}')
        ax.axhline(y=recent_support, color='green', linestyle='-', alpha=0.8, label=f'Support: ${recent_support:.2f}')

        ax.set_title(f'{symbol}')
        ax.set_ylabel('Price ($)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        plt.setp(ax.get_xticklabels(), rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, 'support_resistance.png'), dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Saved support/resistance analysis for top {top_n} stocks")

def main():
    """Main technical analysis function"""
    print("Starting Technical Analysis...")

    # Load data
    stock_data = load_stock_data()

    # Perform MACD analysis
    print("\nCalculating MACD indicators...")
    plot_macd_analysis(stock_data)

    # Perform support/resistance analysis
    print("\nIdentifying support and resistance levels...")
    plot_support_resistance(stock_data)

    print("\nTechnical Analysis complete!")

if __name__ == "__main__":
    main()