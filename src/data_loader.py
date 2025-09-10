"""Data loader module for retail pricing and demand data."""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, Any
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """Load and preprocess retail pricing and demand data."""
    
    def __init__(self, data_path: str = "retail_pricing_demand_2024.csv"):
        """Initialize DataLoader.
        
        Args:
            data_path: Path to the CSV file
        """
        self.data_path = data_path
        self.df = None
        self.train_df = None
        self.test_df = None
        
    def load_data(self) -> pd.DataFrame:
        """Load data from CSV file.
        
        Returns:
            DataFrame with loaded data
        """
        try:
            self.df = pd.read_csv(self.data_path)
            self.df['date'] = pd.to_datetime(self.df['date'])
            logger.info(f"Loaded {len(self.df)} rows from {self.data_path}")
            
            # Sort by date, store, and SKU for time series
            self.df = self.df.sort_values(['store_id', 'sku_id', 'date'])
            
            # Calculate derived features if not present
            if 'margin' not in self.df.columns:
                # Estimate cost as 60% of base price (40% margin assumption)
                self.df['cost'] = self.df['base_price'] * 0.6
                self.df['margin'] = self.df['units_sold'] * (self.df['final_price'] - self.df['cost'])
            
            return self.df
            
        except FileNotFoundError:
            # Use sample data if main file not found
            logger.warning(f"File {self.data_path} not found, using sample data")
            self.data_path = "retail_pricing_demand_2024_sample.csv"
            return self.load_data()
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def split_data(self, 
                   split_date: str = "2023-10-01",
                   train_start: Optional[str] = None,
                   train_end: Optional[str] = None,
                   test_start: Optional[str] = None,
                   test_end: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into train and test sets.
        
        Args:
            split_date: Date to split train/test (default from requirements)
            train_start: Optional start date for training
            train_end: Optional end date for training
            test_start: Optional start date for testing
            test_end: Optional end date for testing
            
        Returns:
            Tuple of (train_df, test_df)
        """
        if self.df is None:
            self.load_data()
        
        # Use provided dates or defaults
        if train_start:
            train_mask = self.df['date'] >= pd.to_datetime(train_start)
        else:
            train_mask = self.df['date'] >= self.df['date'].min()
            
        if train_end:
            train_mask &= self.df['date'] <= pd.to_datetime(train_end)
        else:
            train_mask &= self.df['date'] < pd.to_datetime(split_date)
            
        if test_start:
            test_mask = self.df['date'] >= pd.to_datetime(test_start)
        else:
            test_mask = self.df['date'] >= pd.to_datetime(split_date)
            
        if test_end:
            test_mask &= self.df['date'] <= pd.to_datetime(test_end)
        else:
            test_mask &= self.df['date'] <= self.df['date'].max()
        
        self.train_df = self.df[train_mask].copy()
        self.test_df = self.df[test_mask].copy()
        
        logger.info(f"Train set: {len(self.train_df)} rows ({self.train_df['date'].min()} to {self.train_df['date'].max()})")
        logger.info(f"Test set: {len(self.test_df)} rows ({self.test_df['date'].min()} to {self.test_df['date'].max()})")
        
        return self.train_df, self.test_df
    
    def get_store_sku_combinations(self) -> list:
        """Get all store-SKU combinations.
        
        Returns:
            List of (store_id, sku_id) tuples
        """
        if self.df is None:
            self.load_data()
            
        combinations = self.df[['store_id', 'sku_id']].drop_duplicates()
        return list(combinations.itertuples(index=False, name=None))
    
    def get_data_for_store_sku(self, 
                               store_id: str, 
                               sku_id: str,
                               dataset: str = 'train') -> pd.DataFrame:
        """Get data for a specific store-SKU combination.
        
        Args:
            store_id: Store identifier
            sku_id: SKU identifier
            dataset: 'train', 'test', or 'all'
            
        Returns:
            Filtered DataFrame
        """
        if dataset == 'train':
            df = self.train_df
        elif dataset == 'test':
            df = self.test_df
        else:
            df = self.df
            
        if df is None:
            raise ValueError(f"Dataset {dataset} not loaded. Run split_data() first.")
            
        mask = (df['store_id'] == store_id) & (df['sku_id'] == sku_id)
        return df[mask].copy()
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics of the dataset.
        
        Returns:
            Dictionary with summary statistics
        """
        if self.df is None:
            self.load_data()
            
        stats = {
            'n_rows': len(self.df),
            'n_stores': self.df['store_id'].nunique(),
            'n_skus': self.df['sku_id'].nunique(),
            'date_range': f"{self.df['date'].min()} to {self.df['date'].max()}",
            'avg_units_sold': self.df['units_sold'].mean(),
            'avg_revenue': self.df['revenue'].mean(),
            'avg_price': self.df['final_price'].mean(),
            'promo_rate': self.df['promo_flag'].mean(),
            'stockout_rate': self.df['stockout_flag'].mean() if 'stockout_flag' in self.df.columns else 0,
            'price_elasticity_estimate': self._estimate_elasticity()
        }
        
        return stats
    
    def _estimate_elasticity(self) -> float:
        """Estimate average price elasticity.
        
        Returns:
            Average price elasticity coefficient
        """
        # Simple elasticity calculation: % change in quantity / % change in price
        elasticities = []
        
        for store_id, sku_id in self.get_store_sku_combinations():
            data = self.get_data_for_store_sku(store_id, sku_id, 'all')
            if len(data) > 10:
                # Calculate period-over-period changes
                data['price_pct_change'] = data['final_price'].pct_change()
                data['quantity_pct_change'] = data['units_sold'].pct_change()
                
                # Filter out extreme values and calculate elasticity
                mask = (data['price_pct_change'].abs() > 0.01) & (data['price_pct_change'].abs() < 0.5)
                if mask.sum() > 0:
                    elasticity = (data.loc[mask, 'quantity_pct_change'] / 
                                data.loc[mask, 'price_pct_change']).median()
                    if not np.isnan(elasticity) and abs(elasticity) < 10:
                        elasticities.append(elasticity)
        
        return np.median(elasticities) if elasticities else -1.5
    
    def handle_stockouts(self, method: str = 'interpolate') -> pd.DataFrame:
        """Handle stockout periods in the data.
        
        Args:
            method: 'interpolate', 'forward_fill', or 'flag_only'
            
        Returns:
            DataFrame with stockouts handled
        """
        if self.df is None:
            self.load_data()
            
        df = self.df.copy()
        
        if 'stockout_flag' in df.columns:
            stockout_mask = df['stockout_flag'] == 1
            
            if method == 'interpolate':
                # Interpolate demand during stockouts
                df.loc[stockout_mask, 'units_sold'] = np.nan
                df['units_sold'] = df.groupby(['store_id', 'sku_id'])['units_sold'].transform(
                    lambda x: x.interpolate(method='linear', limit=3)
                )
            elif method == 'forward_fill':
                # Use last known demand
                df.loc[stockout_mask, 'units_sold'] = np.nan
                df['units_sold'] = df.groupby(['store_id', 'sku_id'])['units_sold'].transform(
                    lambda x: x.fillna(method='ffill', limit=3)
                )
            elif method == 'flag_only':
                # Keep stockout flag for modeling
                df['stockout_adjusted'] = df['units_sold'].copy()
                df.loc[stockout_mask, 'stockout_adjusted'] = np.nan
        
        return df


if __name__ == "__main__":
    # Test the data loader
    loader = DataLoader()
    df = loader.load_data()
    train_df, test_df = loader.split_data()
    stats = loader.get_summary_stats()
    
    print("\n=== Data Summary ===")
    for key, value in stats.items():
        print(f"{key}: {value}")