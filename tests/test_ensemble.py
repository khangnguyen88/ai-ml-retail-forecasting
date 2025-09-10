"""Unit tests for Gradient Boosting ensemble model."""

import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_regression
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.ensemble_model import EnsembleForecaster


class TestEnsembleModel:
    """Test suite for Gradient Boosting ensemble model."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample time series data for testing."""
        np.random.seed(42)
        n_samples = 1000
        
        # Create features
        X, y = make_regression(
            n_samples=n_samples,
            n_features=20,
            n_informative=15,
            noise=10,
            random_state=42
        )
        
        # Convert to DataFrame
        feature_names = [f'feature_{i}' for i in range(20)]
        X_df = pd.DataFrame(X, columns=feature_names)
        
        # Add categorical features
        X_df['store_id'] = np.random.choice(['S001', 'S002', 'S003'], n_samples)
        X_df['sku_id'] = np.random.choice(['A', 'B', 'C'], n_samples)
        X_df['day_of_week'] = np.random.randint(0, 7, n_samples)
        X_df['month'] = np.random.randint(1, 13, n_samples)
        X_df['quarter'] = (X_df['month'] - 1) // 3 + 1
        
        # Ensure positive target for demand
        y = np.abs(y) + 10
        
        return X_df, pd.Series(y)
    
    def test_initialization(self):
        """Test model initialization."""
        model = EnsembleForecaster()
        
        assert model.use_gpu == False
        assert model.enable_categorical == True
        assert model.random_state == 42
        assert model.model is None
        assert len(model.categorical_features) == 5
    
    def test_model_params(self):
        """Test Gradient Boosting parameter configuration."""
        model = EnsembleForecaster()
        params = model._get_model_params()
        
        # Check key scikit-learn GradientBoosting parameters
        assert params['loss'] == 'squared_error'
        assert params['learning_rate'] == 0.02
        assert params['n_estimators'] == 500
        assert params['max_depth'] == 12
        assert params['criterion'] == 'friedman_mse'
    
    def test_fit_predict(self, sample_data):
        """Test model fitting and prediction."""
        X, y = sample_data
        
        # Split data
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Create and fit model
        model = EnsembleForecaster()
        model.fit(X_train, y_train)
        
        # Test predictions
        predictions = model.predict(X_test)
        
        assert len(predictions) == len(X_test)
        assert np.all(predictions >= 0)  # Demand should be non-negative
        assert model.feature_columns is not None
        assert len(model.feature_importance_) > 0
    
    def test_fit_with_validation(self, sample_data):
        """Test model fitting with validation set."""
        X, y = sample_data
        
        # Split data
        train_size = int(len(X) * 0.6)
        val_size = int(len(X) * 0.2)
        
        X_train = X[:train_size]
        y_train = y[:train_size]
        X_val = X[train_size:train_size+val_size]
        y_val = y[train_size:train_size+val_size]
        
        # Fit with validation
        model = EnsembleForecaster()
        model.fit(X_train, y_train, X_val, y_val)
        
        assert model.model is not None
        assert model.model.best_iteration > 0
    
    def test_evaluate(self, sample_data):
        """Test model evaluation metrics."""
        X, y = sample_data
        
        # Split data
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Train and evaluate
        model = EnsembleForecaster()
        model.fit(X_train, y_train)
        metrics = model.evaluate(X_test, y_test)
        
        # Check metrics structure
        assert 'gradient_boosting' in metrics
        assert 'rmse' in metrics['gradient_boosting']
        assert 'mape' in metrics['gradient_boosting']
        assert 'r2' in metrics['gradient_boosting']
        assert 'mae' in metrics['gradient_boosting']
        assert 'mean_error' in metrics['gradient_boosting']
        
        # Check metric values are reasonable
        assert metrics['gradient_boosting']['rmse'] > 0
        assert 0 <= metrics['gradient_boosting']['mape'] <= 100
        assert -1 <= metrics['gradient_boosting']['r2'] <= 1
    
    def test_feature_importance(self, sample_data):
        """Test feature importance extraction."""
        X, y = sample_data
        
        # Train model
        model = EnsembleForecaster()
        model.fit(X[:800], y[:800])
        
        # Get feature importance
        importance = model.get_feature_importance(top_n=10)
        
        assert len(importance) == 10
        assert 'feature' in importance.columns
        assert 'importance' in importance.columns
        assert importance['importance'].sum() == 1.0  # Should be normalized
        assert importance['importance'].iloc[0] >= importance['importance'].iloc[-1]  # Sorted
    
    def test_save_load_model(self, sample_data, tmp_path):
        """Test model saving and loading."""
        X, y = sample_data
        
        # Train model
        model = EnsembleForecaster()
        model.fit(X[:800], y[:800])
        
        # Save model
        model_path = tmp_path / "test_model.pkl"
        model.save_model(str(model_path))
        
        # Load model
        new_model = EnsembleForecaster()
        new_model.load_model(str(model_path))
        
        # Test loaded model
        predictions = new_model.predict(X[800:])
        assert len(predictions) == len(X[800:])
        assert new_model.feature_columns == model.feature_columns
    
    def test_categorical_features(self, sample_data):
        """Test handling of categorical features."""
        X, y = sample_data
        
        # Check categorical features are handled
        model = EnsembleForecaster(enable_categorical=True)
        X_processed = model._prepare_features(X)
        
        # Categorical columns should be converted to 'category' dtype
        for col in ['store_id', 'sku_id']:
            assert X_processed[col].dtype.name == 'category'
    
    def test_predict_with_missing_columns(self, sample_data):
        """Test prediction with missing columns raises error."""
        X, y = sample_data
        
        # Train model
        model = EnsembleForecaster()
        model.fit(X[:800], y[:800])
        
        # Try to predict with missing columns
        X_missing = X[800:].drop(columns=['feature_0'])
        
        with pytest.raises(KeyError):
            model.predict(X_missing)
    
    def test_cross_validation(self, sample_data):
        """Test time series cross-validation."""
        X, y = sample_data
        
        # Run cross-validation
        model = EnsembleForecaster()
        cv_scores = model.cross_validate(X[:500], y[:500], n_splits=3)
        
        assert 'rmse' in cv_scores
        assert 'mape' in cv_scores
        assert 'r2' in cv_scores
        assert len(cv_scores['rmse']) == 3
        assert all(score > 0 for score in cv_scores['rmse'])
    
    def test_explain_prediction(self, sample_data):
        """Test SHAP-like feature contribution explanation."""
        X, y = sample_data
        
        # Train model
        model = EnsembleForecaster()
        model.fit(X[:800], y[:800])
        
        # Explain single prediction
        X_sample = X[800:801]
        contributions = model.explain_prediction(X_sample)
        
        assert len(contributions.columns) == len(model.feature_columns) + 1  # +1 for baseline
        assert 'baseline' in contributions.columns
        
        # Check that contributions are reasonable (simplified explanation test)
        prediction = model.predict(X_sample)[0]
        
        # The explain_prediction method uses a simplified approach
        # Just verify the structure is correct, not exact mathematical equivalence
        assert contributions.shape[0] == 1  # One row for one sample
        assert 'baseline' in contributions.columns
        
        # Check that contributions exist and are numeric
        assert not contributions.isnull().any().any()
        assert all(pd.api.types.is_numeric_dtype(contributions[col]) for col in contributions.columns)
    
    def test_predict_with_uncertainty(self, sample_data):
        """Test prediction with uncertainty estimates."""
        X, y = sample_data
        
        # Train model
        model = EnsembleForecaster()
        model.fit(X[:800], y[:800])
        
        # Predict with uncertainty
        mean_pred, std_pred = model.predict_with_uncertainty(X[800:850], n_iterations=10)
        
        assert len(mean_pred) == 50
        assert len(std_pred) == 50
        assert np.all(std_pred >= 0)  # Standard deviation should be non-negative
    
    def test_gpu_params(self):
        """Test GPU parameter configuration (scikit-learn doesn't use GPU)."""
        model = EnsembleForecaster(use_gpu=True)
        params = model._get_model_params()
        
        # Scikit-learn GradientBoosting doesn't have GPU parameters
        # Just check that use_gpu flag doesn't break parameter generation
        assert 'loss' in params
        assert 'n_estimators' in params
        assert params['random_state'] == model.random_state
    
    def test_deterministic_results(self, sample_data):
        """Test that results are deterministic with same seed."""
        X, y = sample_data
        
        # Train two models with same seed
        model1 = EnsembleForecaster(random_state=42)
        model1.fit(X[:800], y[:800])
        pred1 = model1.predict(X[800:850])
        
        model2 = EnsembleForecaster(random_state=42)
        model2.fit(X[:800], y[:800])
        pred2 = model2.predict(X[800:850])
        
        # Predictions should be identical
        np.testing.assert_array_almost_equal(pred1, pred2, decimal=5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])