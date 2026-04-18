"""
Pattern Recognition Implementation
Identifying chart patterns, support/resistance levels, and breakout analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta

# Define paths
DATASET_PATH = "/scratch/temp_/NN_Trading_Bot_Private/NN_Trading_project/A_Data_Gathering/dataset/stocks"
OUTPUT_PATH = "/scratch/temp_/NN_Trading_Bot_Private/NN_Trading_project/B_Data_Analysis"

def load_stock_data():
    """Load stock data"""
    stock_data = {}

    for file in os.listdir(DATASET_PATH):
        if file.endswith('.csv'):
            symbol = file.split('.')[0]
            df = pd.read_csv(os.path.join(DATASET_PATH, file))
            df = df.iloc[1:].copy()

            df['Date'] = pd.to_datetime(df['Date'])
            numeric_cols = ['Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            df.set_index('Date', inplace=True)
            df.sort_index(inplace=True)
            stock_data[symbol] = df

    return stock_data

def detect_head_and_shoulders(df, window=20):
    """Detect Head and Shoulders pattern"""
    pattern_signals = []

    # Calculate rolling highs and lows
    rolling_high = df['High'].rolling(window=window).max()
    rolling_low = df['Low'].rolling(window=window).min()

    # Look for potential H&S pattern
    for i in range(window, len(df)-window):
        left_shoulder = df['High'].iloc[i-window*2:i-window].max()
        head = df['High'].iloc[i-window:i].max()
        right_shoulder = df['High'].iloc[i:i+window].max()

        # Check if it's an inverted H&S (for uptrend)
        if (left_shoulder > head * 1.01 and
            right_shoulder > head * 1.01 and
            head < left_shoulder * 0.98 and
            head < right_shoulder * 0.98):
            pattern_signals.append({
                'date': df.index[i],
                'pattern': 'Head and Shoulders (Top)',
                'price': head,
                'left_shoulder': left_shoulder,
                'right_shoulder': right_shoulder
            })

        # Check for inverted H&S (for downtrend)
        left_shoulder_low = df['Low'].iloc[i-window*2:i-window].min()
        head_low = df['Low'].iloc[i-window:i].min()
        right_shoulder_low = df['Low'].iloc[i:i+window].min()

        if (left_shoulder_low < head_low * 0.99 and
            right_shoulder_low < head_low * 0.99 and
            head_low > left_shoulder_low * 1.02 and
            head_low > right_shoulder_low * 1.02):
            pattern_signals.append({
                'date': df.index[i],
                'pattern': 'Head and Shoulders (Bottom)',
                'price': head_low,
                'left_shoulder': left_shoulder_low,
                'right_shoulder': right_shoulder_low
            })

    return pattern_signals

def detect_double_top_bottom(df, window=20):
    """Detect Double Top/Bottom patterns"""
    pattern_signals = []

    for i in range(window, len(df)-window):
        # Double top detection
        peak1 = df['High'].iloc[i-window*2:i-window].max()
        valley = df['Low'].iloc[i-window:i].min()
        peak2 = df['High'].iloc[i:i+window].max()

        if (abs(peak1 - peak2) / peak1 < 0.02 and  # Similar peaks
            valley < min(peak1, peak2) * 0.97 and  # Clear valley
            peak1 > df['High'].mean()):  # Significant high
            pattern_signals.append({
                'date': df.index[i],
                'pattern': 'Double Top',
                'peak1': peak1,
                'peak2': peak2,
                'valley': valley
            })

        # Double bottom detection
        valley1 = df['Low'].iloc[i-window*2:i-window].min()
        peak = df['High'].iloc[i-window:i].max()
        valley2 = df['Low'].iloc[i:i+window].min()

        if (abs(valley1 - valley2) / valley1 < 0.02 and  # Similar valleys
            peak > max(valley1, valley2) * 1.03 and  # Clear peak
            valley1 < df['Low'].mean()):  # Significant low
            pattern_signals.append({
                'date': df.index[i],
                'pattern': 'Double Bottom',
                'valley1': valley1,
                'valley2': valley2,
                'peak': peak
            })

    return pattern_signals

def detect_breakouts(df, window=20):
    """Detect price breakouts above resistance/below support"""
    breakout_signals = []

    # Calculate recent support and resistance
    resistance = df['High'].rolling(window=window).max()
    support = df['Low'].rolling(window=window).min()

    # Look for breakouts
    for i in range(1, len(df)):
        current_price = df['Close'].iloc[i]
        prev_close = df['Close'].iloc[i-1]
        current_resistance = resistance.iloc[i]
        current_support = support.iloc[i]

        # Breakout above resistance
        if (current_price > current_resistance and
            prev_close <= current_resistance):
            breakout_signals.append({
                'date': df.index[i],
                'type': 'Breakout Up',
                'price': current_price,
                'resistance': current_resistance
            })

        # Breakdown below support
        elif (current_price < current_support and
              prev_close >= current_support):
            breakout_signals.append({
                'date': df.index[i],
                'type': 'Breakdown Down',
                'price': current_price,
                'support': current_support
            })

    return breakout_signals

def plot_patterns_for_stock(stock_data, symbol):
    """Plot chart patterns for a specific stock"""
    df = stock_data[symbol]

    # Detect patterns
    hs_patterns = detect_head_and_shoulders(df)
    db_patterns = detect_double_top_bottom(df)
    breakouts = detect_breakouts(df)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

    # Price chart with patterns
    ax1.plot(df.index, df['Close'], label='Close Price', color='black', alpha=0.7)

    # Plot support/resistance
    window = 20
    resistance = df['High'].rolling(window=window).max()
    support = df['Low'].rolling(window=window).min()
    ax1.plot(resistance.index, resistance, label='Resistance', color='red', alpha=0.5, linestyle='--')
    ax1.plot(support.index, support, label='Support', color='green', alpha=0.5, linestyle='--')

    # Plot patterns
    for pattern in hs_patterns:
        if 'Top' in pattern['pattern']:
            ax1.scatter(pattern['date'], pattern['price'], color='red', s=100, marker='v', label='H&S Top')
        else:
            ax1.scatter(pattern['date'], pattern['price'], color='green', s=100, marker='^', label='H&S Bottom')

    for pattern in db_patterns:
        if 'Top' in pattern['pattern']:
            ax1.scatter(pattern['date'], pattern['peak'], color='orange', s=100, marker='v', label='Double Top')
        else:
            ax1.scatter(pattern['date'], pattern['valley'], color='blue', s=100, marker='^', label='Double Bottom')

    for breakout in breakouts:
        if breakout['type'] == 'Breakout Up':
            ax1.scatter(breakout['date'], breakout['price'], color='purple', s=100, marker='*', label='Breakout Up')
        else:
            ax1.scatter(breakout['date'], breakout['price'], color='brown', s=100, marker='*', label='Breakdown Down')

    ax1.set_title(f'{symbol} - Chart Pattern Analysis')
    ax1.set_ylabel('Price ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Volume chart
    ax2.bar(df.index, df['Volume'], color='gray', alpha=0.6)
    ax2.set_title('Volume')
    ax2.set_ylabel('Volume')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, f'{symbol}_patterns.png'), dpi=300, bbox_inches='tight')
    plt.close()

def analyze_all_patterns(stock_data):
    """Analyze patterns for all stocks"""
    print("Analyzing chart patterns for all stocks...")

    all_patterns = {}
    pattern_count = {'Head and Shoulders': 0, 'Double Top/Bottom': 0, 'Breakouts': 0}

    for symbol in stock_data.keys():
        print(f"Analyzing {symbol}...")

        df = stock_data[symbol]

        # Detect patterns
        hs_patterns = detect_head_and_shoulders(df)
        db_patterns = detect_double_top_bottom(df)
        breakouts = detect_breakouts(df)

        all_patterns[symbol] = {
            'head_shoulders': hs_patterns,
            'double_top_bottom': db_patterns,
            'breakouts': breakouts
        }

        pattern_count['Head and Shoulders'] += len(hs_patterns)
        pattern_count['Double Top/Bottom'] += len(db_patterns)
        pattern_count['Breakouts'] += len(breakouts)

        # Plot patterns for top stocks by volume
        df_volume_avg = df['Volume'].mean()
        if df_volume_avg > df['Volume'].mean():  # Above average volume
            plot_patterns_for_stock(stock_data, symbol)

    # Print pattern summary
    print("\nPattern Detection Summary:")
    for pattern_type, count in pattern_count.items():
        print(f"{pattern_type}: {count} occurrences")

    return all_patterns

def main():
    """Main pattern recognition function"""
    print("Starting Pattern Recognition Analysis...")

    # Load data
    stock_data = load_stock_data()

    # Analyze all patterns
    all_patterns = analyze_all_patterns(stock_data)

    print("\nPattern Recognition Analysis complete!")
    print("Check the generated pattern plots for visual analysis.")

if __name__ == "__main__":
    main()