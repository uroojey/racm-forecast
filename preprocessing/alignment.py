import logging
import pandas as pd
from typing import Dict

logger = logging.getLogger(__name__)

class TimeAligner:
    """Aligns multiple DataFrames to a common daily frequency and date range."""
    
    def __init__(self):
        pass

    def align(self, dataframes: Dict[str, pd.DataFrame], fill_method: str = 'ffill') -> Dict[str, pd.DataFrame]:
        """Resamples DataFrames to daily frequency, aligns them to a common date range, 
        fills missing values, and drops rows with remaining NaNs.
        
        Args:
            dataframes: Dictionary mapping data source names to DataFrames.
                        DataFrames are expected to have a datetime index.
            fill_method: Method to fill missing values ('ffill', 'bfill', etc.).
            
        Returns:
            Dictionary of aligned DataFrames.
        """
        aligned_dfs = {}
        
        if not dataframes:
            logger.warning("Empty dataframes dictionary provided.")
            return aligned_dfs
            
        # Resample all to daily frequency and fill
        resampled_dfs = {}
        for name, df in dataframes.items():
            if df.empty:
                logger.warning(f"DataFrame '{name}' is empty. Skipping.")
                continue
            
            # Make sure index is DatetimeIndex
            if not isinstance(df.index, pd.DatetimeIndex):
                try:
                    if 'date' in df.columns:
                        df = df.set_index('date')
                    df.index = pd.to_datetime(df.index)
                except Exception as e:
                    logger.error(f"Failed to convert index to DatetimeIndex for '{name}': {e}")
                    continue

            try:
                # Group by index and mean if duplicates exist
                if df.index.duplicated().any():
                    # Simplified duplicate handling: aggregate numeric by mean, others by first
                    df = df.groupby(level=0).first()
                
                # Resample to daily
                resampled = df.resample('D').mean(numeric_only=True) # or first/last depending on data
                
                # We need to preserve non-numeric columns if relevant. For simplicity, 
                # we just do forward fill on everything after resampling.
                resampled = df.resample('D').ffill()
                
                if fill_method == 'ffill':
                    resampled = resampled.ffill()
                elif fill_method == 'bfill':
                    resampled = resampled.bfill()
                    
                resampled_dfs[name] = resampled
            except Exception as e:
                logger.error(f"Error resampling DataFrame '{name}': {e}")

        if not resampled_dfs:
            return {}

        # Find the common date range (intersection)
        common_index = resampled_dfs[list(resampled_dfs.keys())[0]].index
        for name, df in resampled_dfs.items():
            common_index = common_index.intersection(df.index)

        if common_index.empty:
            logger.warning("No common date range found among DataFrames.")
            return {}

        # Reindex and drop NaNs
        for name, df in resampled_dfs.items():
            aligned = df.loc[common_index]
            # Drop rows where all columns are NaN, or dropna based on preference
            aligned = aligned.dropna(how='any')
            aligned_dfs[name] = aligned

        # Re-intersect after dropna to ensure identical lengths if needed
        final_index = aligned_dfs[list(aligned_dfs.keys())[0]].index
        for name, df in aligned_dfs.items():
            final_index = final_index.intersection(df.index)
            
        for name in aligned_dfs.keys():
            aligned_dfs[name] = aligned_dfs[name].loc[final_index]

        return aligned_dfs
