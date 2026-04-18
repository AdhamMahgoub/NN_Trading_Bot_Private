"""
Market Analysis Implementation
Correlation analysis and sector performance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

# Define paths
DATASET_PATH = "/scratch/temp_/NN_Trading_Bot_Private/NN_Trading_project/A_Data_Gathering/dataset/stocks"
OUTPUT_PATH = "/scratch/temp_/NN_Trading_Bot_Private/NN_Trading_project/B_Data_Analysis"

# Define sector mapping (simplified)
SECTOR_MAPPING = {
    # Technology
    'LOGI': 'Technology',
    'NTES': 'Technology',
    'VRSN': 'Technology',
    'MSI': 'Technology',
    'PAYC': 'Technology',
    'CBOE': 'Financial',
    'SBAC': 'Technology',
    'AVB': 'Real Estate',
    'MAA': 'Real Estate',
    'REG': 'Real Estate',
    'SLG': 'Real Estate',
    # Industrial
    'ITW': 'Industrial',
    'DE': 'Industrial',  # Note: DE not in dataset, just for reference
    'CAT': 'Industrial',  # Note: CAT not in dataset
    'HON': 'Industrial',  # Note: HON not in dataset
    'ROK': 'Industrial',
    'DOV': 'Industrial',
    'PWR': 'Industrial',
    'CMI': 'Industrial',
    'WAB': 'Industrial',
    'BWXT': 'Industrial',
    # Financial
    'MTB': 'Financial',
    'UBSI': 'Financial',
    'CBOE': 'Financial',
    'LAZ': 'Financial',
    'BR': 'Financial',
    'BURL': 'Financial',
    'FVRR': 'Consumer',
    'YELP': 'Technology',
    'IT': 'Technology',
    'AGCO': 'Industrial',
    'ALLE': 'Industrial',
    'ALNY': 'Healthcare',
    'REGN': 'Healthcare',
    'MCK': 'Healthcare',
    'DGX': 'Healthcare',
    'CALM': 'Consumer',
    'HOPE': 'Financial',
    'NSA': 'Financial',
    'RPM': 'Industrial',
    'KLAC': 'Technology',
    'JBHT': 'Industrial',
    'EXR': 'Real Estate',
    'IWBC': 'Financial',  # Assuming EWBC is a bank
    'MCO': 'Financial',
    'TRIN': 'Financial',  # Assuming it's a financial indicator
    # More sectors would need to be defined
}

def load_stock_data():
    """Load stock data"""
    stock_data = {}
    returns_data = {}

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

            # Calculate returns
            df['Daily_Return'] = df['Close'].pct_change()
            returns_data[symbol] = df['Daily_Return']

            stock_data[symbol] = df

    return stock_data, returns_data

def correlation_analysis(returns_data):
    """Perform correlation analysis between stocks"""
    print("Performing correlation analysis...")

    # Create correlation matrix
    returns_df = pd.DataFrame(returns_data)
    corr_matrix = returns_df.corr()

    # Plot correlation heatmap
    plt.figure(figsize=(16, 12))
    sns.heatmap(corr_matrix, cmap='coolwarm', center=0,
                square=True, linewidths=0.5)
    plt.title('Stock Correlation Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, 'correlation_matrix.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # Find highly correlated pairs
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr = corr_matrix.iloc[i, j]
            if abs(corr) > 0.7:  # High correlation threshold
                high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr))

    # Sort by correlation strength
    high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)

    print("\nMost Highly Correlated Stock Pairs:")
    for stock1, stock2, corr in high_corr_pairs[:10]:
        print(f"{stock1} - {stock2}: {corr:.3f}")

    return corr_matrix, high_corr_pairs

def sector_performance_analysis(stock_data, returns_data):
    """Analyze performance by sector"""
    print("\nAnalyzing sector performance...")

    sector_returns = {}
    sector_volatility = {}

    # Calculate sector performance
    for symbol, returns in returns_data.items():
        sector = SECTOR_MAPPING.get(symbol, 'Other')
        total_return = (1 + returns).prod() - 1  # Total return

        if sector not in sector_returns:
            sector_returns[sector] = []
            sector_volatility[sector] = []

        sector_returns[sector].append(total_return)
        sector_volatility[sector].append(returns.std() * np.sqrt(252))  # Annualized volatility

    # Calculate average sector returns
    avg_sector_returns = {}
    avg_sector_volatility = {}
    for sector in sector_returns:
        avg_sector_returns[sector] = np.mean(sector_returns[sector])
        avg_sector_volatility[sector] = np.mean(sector_volatility[sector])

    # Plot sector performance
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Sector returns
    sectors = list(avg_sector_returns.keys())
    returns = list(avg_sector_returns.values())
    colors = ['green' if r > 0 else 'red' for r in returns]

    ax1.bar(sectors, returns, color=colors)
    ax1.set_title('Average Sector Returns')
    ax1.set_ylabel('Return')
    ax1.tick_params(axis='x', rotation=45)
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)

    # Sector volatility
    volatilities = list(avg_sector_volatility.values())
    ax2.bar(sectors, volatilities, color='orange')
    ax2.set_title('Average Sector Volatility')
    ax2.set_ylabel('Annualized Volatility')
    ax2.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, 'sector_performance.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # Print sector statistics
    print("\nSector Performance Summary:")
    for sector in sectors:
        print(f"{sector}:")
        print(f"  Avg Return: {avg_sector_returns[sector]:.2%}")
        print(f"  Avg Volatility: {avg_sector_volatility[sector]:.2%}")
        print(f"  Number of Stocks: {len(sector_returns[sector])}")

    return sector_returns, sector_volatility

def pairs_trading_opportunities(returns_data, high_corr_pairs):
    """Identify potential pairs trading opportunities"""
    print("\nIdentifying pairs trading opportunities...")

    opportunities = []
    for stock1, stock2, corr in high_corr_pairs[:20]:  # Top 20 correlated pairs
        returns1 = returns_data[stock1]
        returns2 = returns_data[stock2]

        # Calculate spread and mean reversion
        spread = returns1 - returns2
        spread_mean = spread.mean()
        spread_std = spread.std()

        # Check if spread has mean reverted recently
        recent_spread = spread.tail(20)  # Last 20 days
        recent_mean = recent_spread.mean()
        z_score = (recent_mean - spread_mean) / spread_std if spread_std > 0 else 0

        opportunities.append({
            'pair': (stock1, stock2),
            'correlation': corr,
            'z_score': z_score,
            'recent_spread': recent_mean,
            'spread_mean': spread_mean,
            'spread_std': spread_std
        })

    # Sort by z-score (most extreme deviations)
    opportunities.sort(key=lambda x: abs(x['z_score']), reverse=True)

    print("\nTop Pairs Trading Opportunities:")
    for opp in opportunities[:10]:
        pair = opp['pair']
        print(f"{pair[0]} - {pair[1]}:")
        print(f"  Correlation: {opp['correlation']:.3f}")
        print(f"  Z-Score: {opp['z_score']:.3f} (Current spread deviation)")
        print(f"  Recent Spread: {opp['recent_spread']:.4f}")
        print()

    return opportunities

def main():
    """Main market analysis function"""
    print("Starting Market Analysis...")

    # Load data
    stock_data, returns_data = load_stock_data()

    # Perform correlation analysis
    corr_matrix, high_corr_pairs = correlation_analysis(returns_data)

    # Sector performance analysis
    sector_returns, sector_volatility = sector_performance_analysis(stock_data, returns_data)

    # Pairs trading opportunities
    opportunities = pairs_trading_opportunities(returns_data, high_corr_pairs)

    print("\nMarket Analysis complete!")

if __name__ == "__main__":
    main()