import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

def calculate_monthly_median_quotes(asofdate, symbols, data_dir):
    """
    Calculate median daily quote for each symbol over the past 3 months
    
    Parameters:
    symbols (list): List of symbol strings
    data_dir (str): Directory path containing parquet files
    
    Returns:
    pd.DataFrame: DataFrame with symbol column and month index, values are median quotes
    """
    
    # Calculate the date range for past 3 months
    start_date = pd.to_datetime('2022-01-01')#asofdate - timedelta(days=90)  # Approximate 3 months
    
    results = []
    
    for symbol in symbols:
        try:
            # Read the parquet file for this symbol
            file_path = Path(data_dir) / f"perp_{symbol}.parquet"
            
            if not file_path.exists():
                print(f"Warning: File not found for symbol {symbol}")
                continue
                
            df = pd.read_parquet(file_path)
            
            # Ensure we have the required columns
            if 'dvol' not in df.columns:
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
            daily_quotes = df_filtered.groupby(['month', 'date'])['dvol'].sum()
            daily_oi = df_filtered.groupby(['month', 'date'])['open_interest'].mean() * df_filtered.groupby(['month', 'date'])['close'].last()
            
            # Calculate median of daily quotes for each month
            monthly_medians = daily_quotes.groupby('month').median()
            monthly_oi_medians = daily_oi.groupby('month').median()
            
            # Add results to our list
            for month, median_quote in monthly_medians.items():
                results.append({
                    'symbol': symbol,
                    'month': month,
                    'median_quote': median_quote,
                    'median_oi': monthly_oi_medians.loc[month]
                })
                
        except Exception as e:
            print(f"Error processing symbol {symbol}: {str(e)}")
            continue
    
    # Create final DataFrame
    if results:
        result_df = pd.DataFrame(results)
        # Pivot to have symbols as columns and months as index
        #pivot_df = result_df.pivot(index='month', columns='symbol', values='median_quote')
        return result_df
    else:
        print("No data processed successfully")
        return pd.DataFrame()


# Example usage:
if __name__ == "__main__":
    # Example symbols list

    futuniverse = pd.read_parquet('/data/crypto/universe/' + 'bn_future_universe.parquet')
    symbols = [i.upper() for i in futuniverse['id']]
    data_dir = "/home/ray/projects/data/sync_folder/factor/norm_1h"  # Replace with your actual data directory
    dates = [pd.to_datetime('2025-05-31')]#pd.date_range('2022-01-01', '2025-06-01', freq = '1m')
    # Get results in pivot format (symbols as columns)
    rdf = pd.DataFrame()
    for date in dates:
        result_pivot = calculate_monthly_median_quotes(date, symbols, data_dir)
        rdf = pd.concat([rdf, result_pivot])
    rdf.to_parquet("/home/moneyking/projects/mlframework/mlneutral/data/universe.parquet")
    print("Pivot format (symbols as columns):")
    print(rdf)
    print()
    """
    import numpy as np
    import pandas as pd
    a = pd.read_parquet("data/universe.parquet")
    a['filter'] = np.logical_and(a['median_quote'] > 1.5e7, a['median_oi'] > 2e7)
    filters = pd.pivot_table(a, index='month', columns='symbol', values='filter').fillna(0)
    filters.index = filters.index.end_time
    filters.to_parquet("data/filters.parquet")

    """