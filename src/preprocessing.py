"""
Comprehensive preprocessing pipeline for consistent feature engineering
between training and test data.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional, Dict, Any
import joblib
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Comprehensive preprocessing pipeline that ensures consistency
    between training and test data.
    """
    
    def __init__(self):
        """Initialize the preprocessor with empty state."""
        self.feature_columns = None
        self.categorical_encoders = {}
        self.scaler = None
        self.feature_stats = {}
        self.is_fitted = False
        
    def fit(self, df: pd.DataFrame, target_col: str = 'units_sold') -> 'DataPreprocessor':
        """
        Fit the preprocessor on training data.
        
        Args:
            df: Training DataFrame with all features
            target_col: Name of the target column
            
        Returns:
            Fitted preprocessor
        """
        logger.info("Fitting preprocessor on training data...")
        
        # Store all columns except target and metadata
        metadata_cols = ['date', 'store_id', 'sku_id', 'revenue', 'margin', 'set', 'stock_cap', 'stockout_flag']
        self.feature_columns = [col for col in df.columns 
                               if col not in metadata_cols and col != target_col]
        
        # Store statistics for each numeric feature
        numeric_cols = df[self.feature_columns].select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            self.feature_stats[col] = {
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'median': df[col].median()
            }
        
        # Fit categorical encoders
        categorical_cols = ['store_id', 'sku_id', 'day_of_week', 'month', 'quarter']
        for col in categorical_cols:
            if col in df.columns:
                self.categorical_encoders[col] = LabelEncoder()
                # Handle unseen categories by fitting on unique values
                unique_vals = df[col].dropna().unique()
                self.categorical_encoders[col].fit(unique_vals)
        
        # Fit scaler on numeric features
        self.scaler = StandardScaler()
        numeric_features = df[numeric_cols].fillna(0)
        self.scaler.fit(numeric_features)
        
        self.is_fitted = True
        logger.info(f"Preprocessor fitted with {len(self.feature_columns)} features")
        
        return self
    
    def transform(self, df: pd.DataFrame, is_training: bool = False) -> pd.DataFrame:
        """
        Transform data ensuring all expected features are present.
        
        Args:
            df: DataFrame to transform
            is_training: Whether this is training data
            
        Returns:
            Transformed DataFrame with consistent features
        """
        if not self.is_fitted and not is_training:
            raise ValueError("Preprocessor must be fitted before transform")
        
        df_processed = df.copy()
        
        # Ensure datetime
        if 'date' in df_processed.columns:
            df_processed['date'] = pd.to_datetime(df_processed['date'])
        
        # Add missing features with appropriate defaults
        for col in self.feature_columns:
            if col not in df_processed.columns:
                logger.warning(f"Feature '{col}' missing, filling with default value")
                
                # Use appropriate default based on feature type
                if col in self.feature_stats:
                    # Use median for numeric features
                    df_processed[col] = self.feature_stats[col]['median']
                elif 'lag' in col:
                    # Use 0 for lag features when missing
                    df_processed[col] = 0
                elif 'roll' in col:
                    # Use 0 for rolling features when missing
                    df_processed[col] = 0
                else:
                    # Default to 0
                    df_processed[col] = 0
        
        # Handle categorical encoding
        for col, encoder in self.categorical_encoders.items():
            if col in df_processed.columns:
                # Handle unseen categories
                df_processed[col] = df_processed[col].apply(
                    lambda x: encoder.transform([x])[0] 
                    if x in encoder.classes_ else -1
                )
        
        # Fill missing values in numeric features
        numeric_cols = df_processed[self.feature_columns].select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in df_processed.columns:
                # Use forward fill, then backward fill, then median
                if col in self.feature_stats:
                    df_processed[col] = df_processed[col].fillna(self.feature_stats[col]['median'])
                else:
                    df_processed[col] = df_processed[col].fillna(0)
        
        # Remove any features not in the fitted feature set
        extra_cols = [col for col in df_processed.columns 
                      if col not in self.feature_columns and col not in ['date', 'store_id', 'sku_id', 'units_sold']]
        if extra_cols:
            logger.info(f"Removing extra columns: {extra_cols}")
            df_processed = df_processed.drop(columns=extra_cols, errors='ignore')
        
        # Ensure column order is consistent
        if self.feature_columns:
            available_features = [col for col in self.feature_columns if col in df_processed.columns]
            other_cols = [col for col in df_processed.columns if col not in self.feature_columns]
            df_processed = df_processed[other_cols + available_features]
        
        return df_processed
    
    def _handle_stockouts(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle stockout bias in demand data.
        
        Args:
            df: DataFrame with stockout information
            
        Returns:
            DataFrame with corrected demand estimates
        """
        def correct_stockout_group(group):
            """Correct stockouts for a specific store-SKU combination."""
            stockout_mask = group['stockout_flag'] == 1
            if stockout_mask.any() and (~stockout_mask).any():
                # Calculate average demand during non-stockout periods
                non_stockout_demand = group[~stockout_mask]['units_sold'].mean()
                non_stockout_std = group[~stockout_mask]['units_sold'].std()
                
                # Estimate true demand during stockouts (assume higher than observed)
                # Use max of observed demand or 120% of average non-stockout demand
                stockout_correction = np.maximum(
                    group.loc[stockout_mask, 'units_sold'],
                    non_stockout_demand * 1.2 + non_stockout_std * 0.5
                )
                
                # Update units_sold for stockout periods
                group.loc[stockout_mask, 'units_sold'] = stockout_correction
                
                # Add a stockout correction feature
                group['stockout_corrected'] = stockout_mask.astype(int)
                
            return group
        
        # Apply stockout correction by store-SKU combination
        if 'stockout_flag' in df.columns and 'units_sold' in df.columns:
            df = df.groupby(['store_id', 'sku_id']).apply(correct_stockout_group).reset_index(drop=True)
            logger.info("Applied stockout bias correction")
        
        return df
    
    def fit_transform(self, df: pd.DataFrame, target_col: str = 'units_sold') -> pd.DataFrame:
        """
        Fit and transform in one step.
        
        Args:
            df: Training DataFrame
            target_col: Target column name
            
        Returns:
            Transformed DataFrame
        """
        self.fit(df, target_col)
        return self.transform(df, is_training=True)
    
    def get_feature_matrix(self, df: pd.DataFrame, 
                          target_col: Optional[str] = 'units_sold') -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Get feature matrix X and target y from preprocessed data.
        
        Args:
            df: Preprocessed DataFrame
            target_col: Target column name
            
        Returns:
            Tuple of (X, y) where y is None if target not in df
        """
        # Ensure we have the expected features
        X = df[self.feature_columns].copy()
        
        # Fill any remaining NaN
        X = X.fillna(0)
        
        # Get target if available
        y = None
        if target_col and target_col in df.columns:
            y = df[target_col].copy()
            # Remove rows where target is NaN
            valid_idx = y.notna()
            X = X[valid_idx]
            y = y[valid_idx]
        
        return X, y
    
    def save(self, filepath: str):
        """Save preprocessor to disk."""
        joblib.dump({
            'feature_columns': self.feature_columns,
            'categorical_encoders': self.categorical_encoders,
            'scaler': self.scaler,
            'feature_stats': self.feature_stats,
            'is_fitted': self.is_fitted
        }, filepath)
        logger.info(f"Preprocessor saved to {filepath}")
    
    def load(self, filepath: str):
        """Load preprocessor from disk."""
        data = joblib.load(filepath)
        self.feature_columns = data['feature_columns']
        self.categorical_encoders = data['categorical_encoders']
        self.scaler = data['scaler']
        self.feature_stats = data['feature_stats']
        self.is_fitted = data['is_fitted']
        logger.info(f"Preprocessor loaded from {filepath}")
        return self


class FeaturePipeline:
    """
    Complete feature engineering and preprocessing pipeline.
    """
    
    def __init__(self):
        """Initialize the pipeline."""
        from features import FeatureEngineer
        self.feature_engineer = FeatureEngineer()
        self.preprocessor = DataPreprocessor()
        self.is_fitted = False
        
    def fit(self, df: pd.DataFrame, target_col: str = 'units_sold') -> 'FeaturePipeline':
        """
        Fit the complete pipeline on training data.
        
        Args:
            df: Raw training DataFrame
            target_col: Target column name
            
        Returns:
            Fitted pipeline
        """
        logger.info("Fitting feature pipeline...")
        
        # Handle stockouts first
        df_corrected = self.preprocessor._handle_stockouts(df)
        
        # Create features
        df_features = self.feature_engineer.create_features(df_corrected)
        
        # Fit preprocessor
        self.preprocessor.fit(df_features, target_col)
        
        self.is_fitted = True
        logger.info("Feature pipeline fitted successfully")
        
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform raw data through the complete pipeline.
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Fully processed DataFrame
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before transform")
        
        # Handle stockouts first
        df_corrected = self.preprocessor._handle_stockouts(df)
        
        # Create features
        df_features = self.feature_engineer.create_features(df_corrected)
        
        # Apply preprocessing
        df_processed = self.preprocessor.transform(df_features)
        
        return df_processed
    
    def fit_transform(self, df: pd.DataFrame, target_col: str = 'units_sold') -> pd.DataFrame:
        """
        Fit and transform in one step.
        
        Args:
            df: Raw training DataFrame
            target_col: Target column name
            
        Returns:
            Fully processed DataFrame
        """
        self.fit(df, target_col)
        return self.transform(df)
    
    def get_train_test_data(self, 
                           train_df: pd.DataFrame, 
                           test_df: pd.DataFrame,
                           target_col: str = 'units_sold') -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Process train and test data ensuring consistency.
        
        Args:
            train_df: Raw training DataFrame
            test_df: Raw test DataFrame
            target_col: Target column name
            
        Returns:
            Tuple of (X_train, y_train, X_test, y_test)
        """
        # Fit on training data
        train_processed = self.fit_transform(train_df, target_col)
        
        # Transform test data
        test_processed = self.transform(test_df)
        
        # Get feature matrices
        X_train, y_train = self.preprocessor.get_feature_matrix(train_processed, target_col)
        X_test, y_test = self.preprocessor.get_feature_matrix(test_processed, target_col)
        
        # Ensure same columns
        common_cols = list(set(X_train.columns) & set(X_test.columns))
        X_train = X_train[common_cols]
        X_test = X_test[common_cols]
        
        logger.info(f"Training shape: {X_train.shape}, Test shape: {X_test.shape}")
        
        return X_train, y_train, X_test, y_test
    
    def save(self, filepath: str):
        """Save complete pipeline to disk."""
        joblib.dump({
            'feature_engineer': self.feature_engineer,
            'preprocessor': self.preprocessor,
            'is_fitted': self.is_fitted
        }, filepath)
        logger.info(f"Pipeline saved to {filepath}")
    
    def load(self, filepath: str):
        """Load complete pipeline from disk."""
        data = joblib.load(filepath)
        self.feature_engineer = data['feature_engineer']
        self.preprocessor = data['preprocessor']
        self.is_fitted = data['is_fitted']
        logger.info(f"Pipeline loaded from {filepath}")
        return self


def create_consistent_features(train_df: pd.DataFrame, 
                              test_df: pd.DataFrame,
                              target_col: str = 'units_sold') -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Convenience function to create consistent features for train and test data.
    
    Args:
        train_df: Raw training DataFrame
        test_df: Raw test DataFrame
        target_col: Target column name
        
    Returns:
        Tuple of (X_train, y_train, X_test, y_test)
    """
    pipeline = FeaturePipeline()
    return pipeline.get_train_test_data(train_df, test_df, target_col)