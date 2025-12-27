import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

def calculate_monthly_median_quotes(symbols, data_dir):
    """
    Calculate median daily quote for each symbol over the past 3 months
    
    Parameters:
    symbols (list): List of symbol strings
    data_dir (str): Directory path containing parquet files
    
    Returns:
    pd.DataFrame: DataFrame with symbol column and month index, values are median quotes
    """
    
    # Calculate the date range for past 3 months
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)  # Approximate 3 months
    
    results = []
    
    for symbol in symbols:
        try:
            # Read the parquet file for this symbol
            file_path = Path(data_dir) / f"{symbol}.parquet"
            
            if not file_path.exists():
                print(f"Warning: File not found for symbol {symbol}")
                continue
                
            df = pd.read_parquet(file_path)
            
            # Ensure we have the required columns
            if 'quote' not in df.columns:
                print(f"Warning: 'quote' column not found for symbol {symbol}")
                continue
            
            # Convert index to datetime if it's not already
            if not isinstance(df.index, pd.DatetimeIndex):
                # Assuming the index or a column contains timestamp data
                # You might need to adjust this based on your data structure
                if 'end_tm' in df.columns:
                    df['end_tm'] = pd.to_datetime(df['end_tm'])
                    df.set_index('end_tm', inplace=True)
                else:
                    # Try to convert index to datetime
                    df.index = pd.to_datetime(df.index)
            
            # Filter data for the past 3 months
            df_filtered = df[df.index >= start_date]
            
            if df_filtered.empty:
                print(f"Warning: No data in the past 3 months for symbol {symbol}")
                continue
            
            # Group by month and calculate daily aggregations first, then median
            df_filtered['date'] = df_filtered.index.date
            df_filtered['month'] = df_filtered.index.to_period('M')
            
            # Calculate daily quote (you might want to adjust this - using mean of daily quotes)
            daily_quotes = df_filtered.groupby(['month', 'date'])['quote'].sum()
            
            # Calculate median of daily quotes for each month
            monthly_medians = daily_quotes.groupby('month').median()
            
            # Add results to our list
            for month, median_quote in monthly_medians.items():
                results.append({
                    'symbol': symbol,
                    'month': month,
                    'median_quote': median_quote
                })
                
        except Exception as e:
            print(f"Error processing symbol {symbol}: {str(e)}")
            continue
    
    # Create final DataFrame
    if results:
        result_df = pd.DataFrame(results)
        # Pivot to have symbols as columns and months as index
        pivot_df = result_df.pivot(index='month', columns='symbol', values='median_quote')
        return pivot_df
    else:
        print("No data processed successfully")
        return pd.DataFrame()

# Alternative version that keeps symbol as a column
def calculate_monthly_median_quotes_long_format(symbols, data_dir):
    """
    Same calculation but returns data in long format with symbol as a column
    """
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    
    results = []
    
    for symbol in symbols:
        try:
            file_path = Path(data_dir) / f"{symbol}.parquet"
            
            if not file_path.exists():
                print(f"Warning: File not found for symbol {symbol}")
                continue
                
            df = pd.read_parquet(file_path)
            
            if 'quote' not in df.columns:
                print(f"Warning: 'quote' column not found for symbol {symbol}")
                continue
            
            # Handle datetime index
            if not isinstance(df.index, pd.DatetimeIndex):
                if 'end_tm' in df.columns:
                    df['end_tm'] = pd.to_datetime(df['end_tm'])
                    df.set_index('end_tm', inplace=True)
                else:
                    df.index = pd.to_datetime(df.index)
            
            # Filter for past 3 months
            df_filtered = df[df.index >= start_date]
            
            if df_filtered.empty:
                continue
            
            # Calculate monthly medians
            df_filtered['date'] = df_filtered.index.date
            df_filtered['month'] = df_filtered.index.to_period('M')
            
            daily_quotes = df_filtered.groupby(['month', 'date'])['quote'].mean()
            monthly_medians = daily_quotes.groupby('month').median()
            
            for month, median_quote in monthly_medians.items():
                results.append({
                    'symbol': symbol,
                    'month': month,
                    'median_quote': median_quote
                })
                
        except Exception as e:
            print(f"Error processing symbol {symbol}: {str(e)}")
            continue
    
    if results:
        result_df = pd.DataFrame(results)
        result_df.set_index('month', inplace=True)
        return result_df
    else:
        return pd.DataFrame()

# Example usage:
if __name__ == "__main__":
    # Example symbols list

    futuniverse = pd.read_parquet('/data/crypto/universe/' + 'bn_future_universe.parquet')
    symbols = [i.upper() for i in futuniverse['id']]
    data_dir = "/data/crypto/bar1m/futures"  # Replace with your actual data directory
    
    # Get results in pivot format (symbols as columns)
    result_pivot = calculate_monthly_median_quotes(symbols, data_dir)
    print("Pivot format (symbols as columns):")
    print(result_pivot)
    print()
    """
    # Get results in long format (symbol as column)
    result_long = calculate_monthly_median_quotes_long_format(symbols, data_dir)
    print("Long format (symbol as column):")
    print(result_long)
    """