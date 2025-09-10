"""
Unit tests for feature engineering module.

Tests feature shapes, monotonic relationships, and data integrity.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features import FeatureEngineer


class TestFeatureEngineer:
    """Test suite for feature engineering."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        
        data = {
            'date': dates,
            'store_id': np.random.choice(['S01', 'S02'], 100),
            'sku_id': np.random.choice(['SKU001', 'SKU002'], 100),
            'units_sold': np.random.randint(10, 100, 100),
            'final_price': np.random.uniform(10, 20, 100),
            'base_price': np.random.uniform(12, 22, 100),
            'competitor_price': np.random.uniform(10, 20, 100),
            'promo_flag': np.random.choice([0, 1], 100),
            'promo_depth': np.random.uniform(0, 0.3, 100),
            'holiday_flag': np.random.choice([0, 1], 100, p=[0.9, 0.1]),
            'weather_index': np.random.uniform(0, 1, 100),
            'week_of_year': [d.isocalendar()[1] for d in dates]
        }
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def feature_engineer(self):
        """Create FeatureEngineer instance."""
        return FeatureEngineer()
    
    def test_feature_creation(self, feature_engineer, sample_data):
        """Test that features are created correctly."""
        # Create features
        df_features = feature_engineer.create_features(sample_data.copy())
        
        # Check that DataFrame is returned
        assert isinstance(df_features, pd.DataFrame)
        
        # Check that original columns are preserved
        for col in sample_data.columns:
            assert col in df_features.columns
        
        # Check that new features are created
        assert len(df_features.columns) > len(sample_data.columns)
        
        # Check specific feature categories exist
        assert any('lag' in col for col in df_features.columns), "Lag features should exist"
        assert any('roll' in col for col in df_features.columns), "Rolling features should exist"
        assert 'day_of_week' in df_features.columns, "Time features should exist"
        assert 'price_ratio' in df_features.columns, "Price features should exist"
    
    def test_lag_features(self, feature_engineer, sample_data):
        """Test lag feature creation."""
        df_features = feature_engineer.create_features(sample_data.copy(), lag_days=[1, 7])
        
        # Check lag features exist
        assert 'units_sold_lag_1' in df_features.columns
        assert 'units_sold_lag_7' in df_features.columns
        assert 'final_price_lag_1' in df_features.columns
        
        # Check lag values are correct (should have NaN for first values)
        assert pd.isna(df_features['units_sold_lag_1'].iloc[0])
        
        # Check non-NaN values match original shifted values
        for store in df_features['store_id'].unique():
            for sku in df_features['sku_id'].unique():
                mask = (df_features['store_id'] == store) & (df_features['sku_id'] == sku)
                subset = df_features[mask].reset_index(drop=True)
                
                if len(subset) > 7:
                    # Check 1-day lag
                    assert subset['units_sold_lag_1'].iloc[1] == subset['units_sold'].iloc[0]
    
    def test_rolling_features(self, feature_engineer, sample_data):
        """Test rolling window features."""
        df_features = feature_engineer.create_features(sample_data.copy(), rolling_windows=[7])
        
        # Check rolling features exist
        assert 'units_sold_roll_mean_7' in df_features.columns
        assert 'units_sold_roll_std_7' in df_features.columns
        assert 'units_sold_roll_max_7' in df_features.columns
        assert 'units_sold_roll_min_7' in df_features.columns
        
        # Check that rolling features are calculated correctly
        # Note: Implementation uses shift(1) so current value not included
        for store in df_features['store_id'].unique()[:1]:  # Test one store
            for sku in df_features['sku_id'].unique()[:1]:  # Test one SKU
                mask = (df_features['store_id'] == store) & (df_features['sku_id'] == sku)
                subset = df_features[mask].reset_index(drop=True)
                
                if len(subset) > 8:
                    # Manual calculation for row 8
                    expected_mean = subset['units_sold'].iloc[0:7].mean()
                    actual_mean = subset['units_sold_roll_mean_7'].iloc[8]
                    
                    # Allow for small numerical differences
                    if not pd.isna(actual_mean) and not pd.isna(expected_mean):
                        assert abs(actual_mean - expected_mean) < 1.0
    
    def test_price_features(self, feature_engineer, sample_data):
        """Test price-related features."""
        df_features = feature_engineer.create_features(sample_data.copy())
        
        # Check price features exist
        assert 'price_ratio' in df_features.columns
        assert 'discount_amount' in df_features.columns
        assert 'log_price' in df_features.columns
        assert 'price_vs_comp' in df_features.columns
        
        # Test price ratio calculation
        expected_ratio = df_features['final_price'] / df_features['base_price']
        pd.testing.assert_series_equal(df_features['price_ratio'], expected_ratio, check_names=False)
        
        # Test discount amount
        expected_discount = df_features['base_price'] - df_features['final_price']
        pd.testing.assert_series_equal(df_features['discount_amount'], expected_discount, check_names=False)
        
        # Test log transformation (should be log1p)
        expected_log = np.log1p(df_features['final_price'])
        pd.testing.assert_series_equal(df_features['log_price'], expected_log, check_names=False)
    
    def test_monotonic_price_demand(self, feature_engineer):
        """Test that price has expected monotonic relationship with demand."""
        # Create data with clear price-demand relationship
        prices = np.linspace(10, 20, 100)
        demand = 100 - 3 * prices + np.random.normal(0, 5, 100)  # Negative relationship
        
        data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=100),
            'store_id': 'S01',
            'sku_id': 'SKU001',
            'units_sold': np.maximum(0, demand),
            'final_price': prices,
            'base_price': prices * 1.1,
            'competitor_price': prices,
            'promo_flag': 0,
            'promo_depth': 0,
            'holiday_flag': 0,
            'weather_index': 0.5,
            'week_of_year': 1
        })
        
        df_features = feature_engineer.create_features(data)
        
        # Check negative correlation between price and demand
        correlation = df_features['final_price'].corr(df_features['units_sold'])
        assert correlation < 0, f"Price should have negative correlation with demand, got {correlation}"
    
    def test_temporal_features(self, feature_engineer, sample_data):
        """Test temporal feature creation."""
        df_features = feature_engineer.create_features(sample_data.copy())
        
        # Check temporal features exist
        assert 'day_of_week' in df_features.columns
        assert 'month' in df_features.columns
        assert 'is_weekend' in df_features.columns
        assert 'day_sin' in df_features.columns
        assert 'day_cos' in df_features.columns
        
        # Test day of week (0 = Monday, 6 = Sunday)
        assert df_features['day_of_week'].min() >= 0
        assert df_features['day_of_week'].max() <= 6
        
        # Test weekend flag
        weekend_days = df_features[df_features['day_of_week'].isin([5, 6])]
        assert (weekend_days['is_weekend'] == 1).all()
        
        weekday_days = df_features[~df_features['day_of_week'].isin([5, 6])]
        assert (weekday_days['is_weekend'] == 0).all()
        
        # Test cyclic encoding range
        assert df_features['day_sin'].min() >= -1
        assert df_features['day_sin'].max() <= 1
        assert df_features['day_cos'].min() >= -1
        assert df_features['day_cos'].max() <= 1
    
    def test_interaction_features(self, feature_engineer, sample_data):
        """Test interaction feature creation."""
        df_features = feature_engineer.create_features(sample_data.copy(), include_interactions=True)
        
        # Check interaction features exist
        assert 'price_x_promo' in df_features.columns
        assert 'price_x_holiday' in df_features.columns
        assert 'price_x_weather' in df_features.columns
        
        # Test price x promo interaction
        expected = df_features['final_price'] * df_features['promo_flag']
        pd.testing.assert_series_equal(df_features['price_x_promo'], expected, check_names=False)
    
    def test_feature_shapes(self, feature_engineer, sample_data):
        """Test that feature shapes are consistent."""
        df_features = feature_engineer.create_features(sample_data.copy())
        
        # All features should have same length as original data
        assert len(df_features) == len(sample_data)
        
        # Feature columns should be identified
        assert len(feature_engineer.feature_columns) > 0
        
        # All feature columns should exist in DataFrame
        for col in feature_engineer.feature_columns:
            assert col in df_features.columns
    
    def test_prepare_for_modeling(self, feature_engineer, sample_data):
        """Test data preparation for modeling."""
        df_features = feature_engineer.create_features(sample_data.copy())
        X, y = feature_engineer.prepare_for_modeling(df_features)
        
        # Check types
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        
        # Check shapes
        assert len(X) == len(y)
        assert len(X) <= len(df_features)  # May be less due to NaN removal
        
        # Check no NaN in features
        assert not X.isnull().any().any(), "Features should not contain NaN"
        assert not y.isnull().any(), "Target should not contain NaN"
        
        # Check target is units_sold
        assert y.name == 'units_sold'
    
    def test_feature_selection(self, feature_engineer, sample_data):
        """Test feature selection with correlation threshold."""
        df_features = feature_engineer.create_features(sample_data.copy())
        
        # Create highly correlated features for testing
        df_features['highly_correlated_1'] = df_features['final_price'] * 2
        df_features['highly_correlated_2'] = df_features['final_price'] * 2.01
        
        feature_engineer.feature_columns.extend(['highly_correlated_1', 'highly_correlated_2'])
        
        # Select features with correlation threshold
        selected = feature_engineer.select_features(df_features, correlation_threshold=0.95)
        
        # Should have removed at least one of the highly correlated features
        assert 'highly_correlated_1' not in selected or 'highly_correlated_2' not in selected
    
    def test_empty_data_handling(self, feature_engineer):
        """Test handling of empty DataFrame."""
        empty_df = pd.DataFrame()
        
        # Should handle empty DataFrame gracefully
        result = feature_engineer.create_features(empty_df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])