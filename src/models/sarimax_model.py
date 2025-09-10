"""SARIMAX model for time series demand forecasting."""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, Any
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
import warnings
import logging
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SARIMAXForecaster:
    """Classical time series model using SARIMAX."""
    
    def __init__(self, 
                 order: Tuple[int, int, int] = (2, 1, 2),
                 seasonal_order: Tuple[int, int, int, int] = (1, 0, 1, 7),
                 exog_features: Optional[list] = None):
        """Initialize SARIMAX model with improved defaults.
        
        Args:
            order: (p, d, q) order of the model
            seasonal_order: (P, D, Q, s) seasonal order
            exog_features: List of exogenous feature names
        """
        self.order = order
        self.seasonal_order = seasonal_order
        # Use no exogenous features by default for better baseline performance
        self.exog_features = exog_features or []
        self.models = {}
        self.model_params = {}
        
    def check_stationarity(self, series: pd.Series) -> Dict[str, Any]:
        """Check if time series is stationary using Augmented Dickey-Fuller test.
        
        Args:
            series: Time series to test
            
        Returns:
            Dictionary with test results
        """
        result = adfuller(series.dropna())
        return {
            'adf_statistic': result[0],
            'p_value': result[1],
            'is_stationary': result[1] < 0.05,
            'critical_values': result[4]
        }
    
    def auto_select_order(self, 
                         series: pd.Series,
                         exog: Optional[pd.DataFrame] = None,
                         max_p: int = 3,
                         max_q: int = 3) -> Tuple[Tuple, float]:
        """Automatically select best ARIMA order using AIC.
        
        Args:
            series: Time series
            exog: Exogenous variables
            max_p: Maximum p value to test
            max_q: Maximum q value to test
            
        Returns:
            Tuple of (best_order, best_aic)
        """
        best_aic = np.inf
        best_order = (1, 1, 1)
        
        for p in range(max_p + 1):
            for q in range(max_q + 1):
                if p == 0 and q == 0:
                    continue
                try:
                    model = SARIMAX(series, 
                                  exog=exog,
                                  order=(p, 1, q),
                                  seasonal_order=self.seasonal_order,
                                  enforce_stationarity=False,
                                  enforce_invertibility=False)
                    results = model.fit(disp=False)
                    
                    if results.aic < best_aic:
                        best_aic = results.aic
                        best_order = (p, 1, q)
                except:
                    continue
        
        logger.info(f"Best order: {best_order} with AIC: {best_aic:.2f}")
        return best_order, best_aic
    
    def fit(self, 
            train_data: pd.DataFrame,
            store_id: str,
            sku_id: str,
            auto_order: bool = False) -> Dict[str, Any]:
        """Fit SARIMAX model for a specific store-SKU combination.
        
        Args:
            train_data: Training data
            store_id: Store identifier
            sku_id: SKU identifier
            auto_order: Whether to automatically select order
            
        Returns:
            Dictionary with model results
        """
        # Filter data for store-SKU
        data = train_data[(train_data['store_id'] == store_id) & 
                         (train_data['sku_id'] == sku_id)].copy()
        
        if len(data) < 30:
            logger.warning(f"Insufficient data for {store_id}-{sku_id}")
            return None
        
        # Sort by date
        data = data.sort_values('date')
        data = data.set_index('date')
        
        # Prepare endogenous and exogenous variables
        endog = data['units_sold']
        
        # Improved data preprocessing for better SARIMAX performance
        # Remove outliers (values beyond 3 std deviations)
        mean_demand = endog.mean()
        std_demand = endog.std()
        outlier_mask = np.abs(endog - mean_demand) <= 3 * std_demand
        
        endog = endog[outlier_mask]
        data_filtered = data[outlier_mask]
        
        # Select available exogenous features and ensure stationarity
        available_exog = [f for f in self.exog_features if f in data_filtered.columns]
        if available_exog:
            exog = data_filtered[available_exog].copy()
            
            # Transform exogenous variables for better SARIMAX performance
            for col in exog.columns:
                if col == 'final_price':
                    # Use price changes instead of levels for stationarity
                    exog[col] = exog[col].pct_change().fillna(0)
                elif col in ['promo_flag', 'holiday_flag']:
                    # Keep binary variables as-is
                    pass
                else:
                    # Standardize other variables
                    exog[col] = (exog[col] - exog[col].mean()) / (exog[col].std() + 1e-8)
        else:
            exog = None
        
        # Check stationarity
        stationarity = self.check_stationarity(endog)
        logger.info(f"Stationarity test for {store_id}-{sku_id}: p-value={stationarity['p_value']:.4f}")
        
        # Auto-select order if requested
        if auto_order:
            self.order, _ = self.auto_select_order(endog, exog)
        
        try:
            # Apply log transformation for better performance
            endog_transformed = np.log1p(endog)  # log(1 + x) transformation
            
            # Try multiple model configurations for robustness
            # Add a naive forecast approach as baseline that should achieve positive R²
            model_configs = [
                # Configuration 1: Naive approaches (often work better on cross-sectional data)
                {'order': (0, 0, 0), 'seasonal_order': (0, 0, 0, 0), 'name': 'Naive Mean', 'special': 'mean'},
                {'order': (1, 0, 0), 'seasonal_order': (0, 0, 0, 0), 'name': 'Naive AR(1)', 'special': 'ar1'},
                # Configuration 2: Simple ARIMA models
                {'order': (1, 1, 0), 'seasonal_order': (0, 0, 0, 0), 'name': 'AR(1) + Diff'},
                {'order': (0, 1, 1), 'seasonal_order': (0, 0, 0, 0), 'name': 'MA(1) + Diff'},
                {'order': (1, 1, 1), 'seasonal_order': (0, 0, 0, 0), 'name': 'ARIMA(1,1,1)'},
                # Configuration 3: More complex models
                {'order': self.order, 'seasonal_order': self.seasonal_order, 'name': 'Original Config'}
            ]
            
            best_model = None
            best_aic = np.inf
            
            for config in model_configs:
                try:
                    # Handle special naive approaches
                    if config.get('special') == 'mean':
                        # Create a simple mean-based "model"
                        class NaiveMeanModel:
                            def __init__(self, mean_val):
                                self.mean_val = mean_val
                                self.aic = 999  # High AIC
                                self.mle_retvals = {'converged': True}
                            def forecast(self, steps):
                                return pd.Series([self.mean_val] * steps)
                                
                        results = NaiveMeanModel(endog_transformed.mean())
                        
                    elif config.get('special') == 'ar1':
                        # Simple AR(1) without differencing
                        model = SARIMAX(endog_transformed,
                                      order=(1, 0, 0),
                                      enforce_stationarity=False)
                        results = model.fit(disp=False, maxiter=50)
                    else:
                        # Regular SARIMAX models
                        model = SARIMAX(endog_transformed,
                                      order=config['order'],
                                      seasonal_order=config['seasonal_order'],
                                      enforce_stationarity=False,
                                      enforce_invertibility=False)
                        
                        results = model.fit(disp=False, maxiter=100)
                    
                    # Select model with best AIC that converged
                    converged = results.mle_retvals.get('converged', True)
                    if converged and results.aic < best_aic:
                        best_model = results
                        best_aic = results.aic
                        self.order = config['order']
                        self.seasonal_order = config['seasonal_order']
                        logger.info(f"  Selected {config.get('name', 'config')} for {store_id}-{sku_id}")
                        
                except Exception as e:
                    continue
            
            if best_model is None:
                logger.warning(f"All SARIMAX configurations failed for {store_id}-{sku_id}")
                return None
                
            results = best_model
            
            # Store model
            model_key = f"{store_id}_{sku_id}"
            self.models[model_key] = results
            
            # Perform diagnostic tests
            residuals = results.resid
            ljung_box = acorr_ljungbox(residuals, lags=10, return_df=True)
            
            # Store parameters
            self.model_params[model_key] = {
                'aic': results.aic,
                'bic': results.bic,
                'llf': results.llf,
                'params': results.params.to_dict(),
                'pvalues': results.pvalues.to_dict(),
                'ljung_box_pvalue': ljung_box['lb_pvalue'].mean(),
                'order': self.order,
                'seasonal_order': self.seasonal_order,
                'exog_features': [],  # No exog features for improved model
                'log_transformed': True  # Track that we used log transformation
            }
            
            logger.info(f"Successfully fitted SARIMAX for {store_id}-{sku_id}, AIC={results.aic:.2f}")
            
            return self.model_params[model_key]
            
        except Exception as e:
            logger.error(f"Error fitting SARIMAX for {store_id}-{sku_id}: {e}")
            return None
    
    def predict(self,
               store_id: str,
               sku_id: str,
               steps: int,
               exog_future: Optional[pd.DataFrame] = None) -> pd.Series:
        """Generate predictions for a specific store-SKU.
        
        Args:
            store_id: Store identifier
            sku_id: SKU identifier
            steps: Number of steps to predict
            exog_future: Future exogenous variables
            
        Returns:
            Series with predictions
        """
        model_key = f"{store_id}_{sku_id}"
        
        if model_key not in self.models:
            logger.warning(f"No model found for {store_id}-{sku_id}")
            return pd.Series()
        
        model = self.models[model_key]
        
        try:
            # Generate forecast (no exogenous features in improved version)
            is_log_transformed = self.model_params[model_key].get('log_transformed', False)
            
            # Generate forecast
            forecast = model.forecast(steps=steps)
            
            # Transform back from log space if needed
            if is_log_transformed:
                forecast = np.expm1(forecast)  # Inverse of log1p
            
            # Ensure non-negative predictions
            forecast = forecast.clip(lower=0)
            
            return forecast
            
        except Exception as e:
            logger.error(f"Error predicting for {store_id}-{sku_id}: {e}")
            return pd.Series()
    
    def evaluate(self,
                test_data: pd.DataFrame,
                store_id: str,
                sku_id: str) -> Dict[str, float]:
        """Evaluate model performance on test data.
        
        Args:
            test_data: Test dataset
            store_id: Store identifier
            sku_id: SKU identifier
            
        Returns:
            Dictionary with evaluation metrics
        """
        model_key = f"{store_id}_{sku_id}"
        
        if model_key not in self.models:
            return {}
        
        # Filter test data
        test = test_data[(test_data['store_id'] == store_id) & 
                        (test_data['sku_id'] == sku_id)].copy()
        test = test.sort_values('date')
        
        if len(test) == 0:
            return {}
        
        # Prepare exogenous variables
        exog_features = self.model_params[model_key].get('exog_features', [])
        exog_test = test[exog_features] if exog_features else None
        
        # Generate predictions
        predictions = self.predict(store_id, sku_id, len(test), exog_test)
        
        if len(predictions) == 0:
            return {}
        
        # Calculate metrics
        y_true = test['units_sold'].values
        y_pred = predictions.values
        
        # Align arrays
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]
        
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mape': mean_absolute_percentage_error(y_true, y_pred) * 100,
            'mae': np.mean(np.abs(y_true - y_pred)),
            'bias': np.mean(y_pred - y_true),
            'r2': 1 - (np.sum((y_true - y_pred)**2) / np.sum((y_true - y_true.mean())**2))
        }
        
        return metrics
    
    def fit_all(self,
               train_data: pd.DataFrame,
               store_sku_combinations: list,
               auto_order: bool = False) -> Dict[str, Dict]:
        """Fit models for all store-SKU combinations.
        
        Args:
            train_data: Training data
            store_sku_combinations: List of (store_id, sku_id) tuples
            auto_order: Whether to auto-select order
            
        Returns:
            Dictionary with all model results
        """
        results = {}
        
        for store_id, sku_id in store_sku_combinations:
            logger.info(f"Fitting SARIMAX for {store_id}-{sku_id}")
            result = self.fit(train_data, store_id, sku_id, auto_order)
            if result:
                results[f"{store_id}_{sku_id}"] = result
        
        logger.info(f"Fitted {len(results)} SARIMAX models")
        return results
    
    def predict_all(self,
                   store_sku_combinations: list,
                   steps: int,
                   exog_future: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Generate predictions for all store-SKU combinations.
        
        Args:
            store_sku_combinations: List of (store_id, sku_id) tuples
            steps: Number of steps to predict
            exog_future: Future exogenous variables
            
        Returns:
            DataFrame with all predictions
        """
        predictions = []
        
        for store_id, sku_id in store_sku_combinations:
            # Filter exog for this combination if provided
            if exog_future is not None:
                exog_combo = exog_future[(exog_future['store_id'] == store_id) & 
                                        (exog_future['sku_id'] == sku_id)]
            else:
                exog_combo = None
            
            preds = self.predict(store_id, sku_id, steps, exog_combo)
            
            if len(preds) > 0:
                pred_df = pd.DataFrame({
                    'store_id': store_id,
                    'sku_id': sku_id,
                    'prediction': preds.values,
                    'date': preds.index if hasattr(preds, 'index') else range(len(preds))
                })
                predictions.append(pred_df)
        
        if predictions:
            return pd.concat(predictions, ignore_index=True)
        else:
            return pd.DataFrame()


if __name__ == "__main__":
    # Test SARIMAX model
    import sys
    sys.path.append('..')
    from data_loader import DataLoader
    
    # Load data
    loader = DataLoader()
    df = loader.load_data()
    train_df, test_df = loader.split_data()
    
    # Get store-SKU combinations
    combos = loader.get_store_sku_combinations()[:2]  # Test with 2 combinations
    
    # Initialize and fit model
    sarimax = SARIMAXForecaster()
    results = sarimax.fit_all(train_df, combos, auto_order=True)
    
    # Evaluate on test set
    for store_id, sku_id in combos:
        metrics = sarimax.evaluate(test_df, store_id, sku_id)
        if metrics:
            print(f"\nMetrics for {store_id}-{sku_id}:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.2f}")