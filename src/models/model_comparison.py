"""Model comparison module for evaluating classical vs ensemble approaches."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelComparison:
    """Compare performance of different forecasting models."""
    
    def __init__(self):
        """Initialize ModelComparison."""
        self.results = {}
        self.predictions = {}
        self.training_times = {}
        self.inference_times = {}
        
    def compare_models(self,
                      sarimax_model,
                      ensemble_model,
                      train_data: pd.DataFrame,
                      test_data: pd.DataFrame,
                      feature_engineer=None) -> pd.DataFrame:
        """Compare SARIMAX and Ensemble models.
        
        Args:
            sarimax_model: Trained SARIMAX model
            ensemble_model: Trained Ensemble model
            train_data: Training data
            test_data: Test data
            feature_engineer: Feature engineering object
            
        Returns:
            DataFrame with comparison results
        """
        comparison_results = []
        
        # Get store-SKU combinations
        combos = test_data[['store_id', 'sku_id']].drop_duplicates().values
        
        for store_id, sku_id in combos[:5]:  # Limit to 5 for demo
            logger.info(f"Comparing models for {store_id}-{sku_id}")
            
            # Filter test data
            test_subset = test_data[(test_data['store_id'] == store_id) & 
                                   (test_data['sku_id'] == sku_id)].copy()
            
            if len(test_subset) < 10:
                continue
            
            result = {
                'store_id': store_id,
                'sku_id': sku_id,
                'test_samples': len(test_subset)
            }
            
            # SARIMAX predictions and metrics
            try:
                start_time = time.time()
                sarimax_pred = sarimax_model.predict(store_id, sku_id, len(test_subset))
                sarimax_time = time.time() - start_time
                
                if len(sarimax_pred) > 0:
                    y_true = test_subset['units_sold'].values[:len(sarimax_pred)]
                    y_pred = sarimax_pred.values
                    
                    result['sarimax_rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
                    result['sarimax_mape'] = mean_absolute_percentage_error(y_true, y_pred) * 100
                    result['sarimax_r2'] = r2_score(y_true, y_pred)
                    result['sarimax_time'] = sarimax_time
                    
                    self.predictions[f'sarimax_{store_id}_{sku_id}'] = {
                        'y_true': y_true,
                        'y_pred': y_pred
                    }
            except Exception as e:
                logger.warning(f"SARIMAX prediction failed for {store_id}-{sku_id}: {e}")
                result['sarimax_rmse'] = np.nan
                result['sarimax_mape'] = np.nan
                result['sarimax_r2'] = np.nan
                result['sarimax_time'] = np.nan
            
            # Ensemble predictions and metrics
            try:
                if feature_engineer:
                    test_features = feature_engineer.create_features(test_subset)
                    X_test, y_test = feature_engineer.prepare_for_modeling(test_features)
                else:
                    # Use basic features if no feature engineer provided
                    feature_cols = ['final_price', 'promo_flag', 'holiday_flag', 
                                  'weather_index', 'competitor_price', 'week_of_year']
                    X_test = test_subset[feature_cols]
                    y_test = test_subset['units_sold']
                
                start_time = time.time()
                ensemble_pred = ensemble_model.predict(X_test)
                ensemble_time = time.time() - start_time
                
                result['ensemble_rmse'] = np.sqrt(mean_squared_error(y_test, ensemble_pred))
                result['ensemble_mape'] = mean_absolute_percentage_error(y_test, ensemble_pred) * 100
                result['ensemble_r2'] = r2_score(y_test, ensemble_pred)
                result['ensemble_time'] = ensemble_time
                
                # Gradient Boosting is our single ensemble model
                result['gradient_boosting_rmse'] = result['ensemble_rmse']
                result['gradient_boosting_mape'] = result['ensemble_mape']
                result['gradient_boosting_r2'] = result['ensemble_r2']
                
                self.predictions[f'ensemble_{store_id}_{sku_id}'] = {
                    'y_true': y_test.values,
                    'y_pred': ensemble_pred
                }
                
            except Exception as e:
                logger.warning(f"Ensemble prediction failed for {store_id}-{sku_id}: {e}")
                result['ensemble_rmse'] = np.nan
                result['ensemble_mape'] = np.nan
                result['ensemble_r2'] = np.nan
                result['ensemble_time'] = np.nan
            
            comparison_results.append(result)
        
        # Create comparison DataFrame
        df_results = pd.DataFrame(comparison_results)
        self.results = df_results
        
        return df_results
    
    def get_summary_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for model comparison.
        
        Returns:
            Dictionary with summary statistics
        """
        if self.results.empty:
            return {}
        
        summary = {}
        
        # SARIMAX statistics
        sarimax_cols = [col for col in self.results.columns if 'sarimax' in col]
        if sarimax_cols:
            summary['sarimax'] = {
                'mean_rmse': self.results['sarimax_rmse'].mean(),
                'std_rmse': self.results['sarimax_rmse'].std(),
                'mean_mape': self.results['sarimax_mape'].mean(),
                'std_mape': self.results['sarimax_mape'].std(),
                'mean_r2': self.results['sarimax_r2'].mean(),
                'mean_time': self.results['sarimax_time'].mean()
            }
        
        # Ensemble statistics
        ensemble_cols = [col for col in self.results.columns if 'ensemble' in col]
        if ensemble_cols:
            summary['ensemble'] = {
                'mean_rmse': self.results['ensemble_rmse'].mean(),
                'std_rmse': self.results['ensemble_rmse'].std(),
                'mean_mape': self.results['ensemble_mape'].mean(),
                'std_mape': self.results['ensemble_mape'].std(),
                'mean_r2': self.results['ensemble_r2'].mean(),
                'mean_time': self.results['ensemble_time'].mean()
            }
        
        # XGBoost statistics
        if 'xgboost_rmse' in self.results.columns:
            summary['xgboost'] = {
                'mean_rmse': self.results['xgboost_rmse'].mean(),
                'mean_mape': self.results['xgboost_mape'].mean()
            }
        
        # Gradient Boosting statistics
        if 'gradient_boosting_rmse' in self.results.columns:
            summary['gradient_boosting'] = {
                'mean_rmse': self.results['gradient_boosting_rmse'].mean(),
                'mean_mape': self.results['gradient_boosting_mape'].mean()
            }
        
        return summary
    
    def plot_comparison(self, save_path: str = None):
        """Create visualization comparing model performance.
        
        Args:
            save_path: Path to save the plot
        """
        if self.results.empty:
            logger.warning("No results to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # RMSE Comparison - handle missing SARIMAX columns
        available_cols = [col for col in ['sarimax_rmse', 'ensemble_rmse'] if col in self.results.columns]
        if available_cols:
            rmse_data = self.results[available_cols].dropna()
            if not rmse_data.empty:
                ax1 = axes[0, 0]
                rmse_data.boxplot(ax=ax1)
                ax1.set_title('RMSE Comparison: SARIMAX vs Ensemble')
                ax1.set_ylabel('RMSE')
                ax1.grid(True, alpha=0.3)
        
        # MAPE Comparison
        available_cols = [col for col in ['sarimax_mape', 'ensemble_mape'] if col in self.results.columns]
        if available_cols:
            mape_data = self.results[available_cols].dropna()
            if not mape_data.empty:
                ax2 = axes[0, 1]
                mape_data.boxplot(ax=ax2)
                ax2.set_title('MAPE Comparison: SARIMAX vs Ensemble')
                ax2.set_ylabel('MAPE (%)')
                ax2.grid(True, alpha=0.3)
        
        # R2 Score Comparison
        available_cols = [col for col in ['sarimax_r2', 'ensemble_r2'] if col in self.results.columns]
        if available_cols:
            r2_data = self.results[available_cols].dropna()
            if not r2_data.empty:
                ax3 = axes[1, 0]
                r2_data.boxplot(ax=ax3)
                ax3.set_title('R² Score Comparison: SARIMAX vs Ensemble')
                ax3.set_ylabel('R² Score')
                ax3.grid(True, alpha=0.3)
        
        # Inference Time Comparison
        available_cols = [col for col in ['sarimax_time', 'ensemble_time'] if col in self.results.columns]
        if available_cols:
            time_data = self.results[available_cols].dropna()
            if not time_data.empty:
                ax4 = axes[1, 1]
                time_data.boxplot(ax=ax4)
                ax4.set_title('Inference Time Comparison: SARIMAX vs Ensemble')
                ax4.set_ylabel('Time (seconds)')
                ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Comparison plot saved to {save_path}")
        
        plt.show()
    
    def plot_predictions_sample(self, 
                               store_id: str, 
                               sku_id: str,
                               save_path: str = None):
        """Plot actual vs predicted values for a specific store-SKU.
        
        Args:
            store_id: Store identifier
            sku_id: SKU identifier
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # SARIMAX predictions
        sarimax_key = f'sarimax_{store_id}_{sku_id}'
        if sarimax_key in self.predictions:
            ax1 = axes[0]
            data = self.predictions[sarimax_key]
            x = range(len(data['y_true']))
            
            ax1.plot(x, data['y_true'], label='Actual', color='blue', alpha=0.7)
            ax1.plot(x, data['y_pred'], label='SARIMAX', color='red', alpha=0.7)
            ax1.set_title(f'SARIMAX Predictions: {store_id}-{sku_id}')
            ax1.set_xlabel('Time')
            ax1.set_ylabel('Units Sold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # Ensemble predictions
        ensemble_key = f'ensemble_{store_id}_{sku_id}'
        if ensemble_key in self.predictions:
            ax2 = axes[1]
            data = self.predictions[ensemble_key]
            x = range(len(data['y_true']))
            
            ax2.plot(x, data['y_true'], label='Actual', color='blue', alpha=0.7)
            ax2.plot(x, data['y_pred'], label='Ensemble', color='green', alpha=0.7)
            ax2.set_title(f'Ensemble Predictions: {store_id}-{sku_id}')
            ax2.set_xlabel('Time')
            ax2.set_ylabel('Units Sold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Predictions plot saved to {save_path}")
        
        plt.show()
    
    def generate_report(self) -> str:
        """Generate a text report comparing models.
        
        Returns:
            String report
        """
        summary = self.get_summary_statistics()
        
        report = []
        report.append("=" * 60)
        report.append("MODEL COMPARISON REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Overall comparison
        report.append("OVERALL PERFORMANCE COMPARISON:")
        report.append("-" * 40)
        
        if 'sarimax' in summary and 'ensemble' in summary:
            # RMSE comparison
            sarimax_rmse = summary['sarimax']['mean_rmse']
            ensemble_rmse = summary['ensemble']['mean_rmse']
            rmse_improvement = ((sarimax_rmse - ensemble_rmse) / sarimax_rmse) * 100
            
            report.append(f"Average RMSE:")
            report.append(f"  SARIMAX:  {sarimax_rmse:.2f}")
            report.append(f"  Ensemble: {ensemble_rmse:.2f}")
            report.append(f"  Improvement: {rmse_improvement:.1f}%")
            report.append("")
            
            # MAPE comparison
            sarimax_mape = summary['sarimax']['mean_mape']
            ensemble_mape = summary['ensemble']['mean_mape']
            mape_improvement = ((sarimax_mape - ensemble_mape) / sarimax_mape) * 100
            
            report.append(f"Average MAPE:")
            report.append(f"  SARIMAX:  {sarimax_mape:.2f}%")
            report.append(f"  Ensemble: {ensemble_mape:.2f}%")
            report.append(f"  Improvement: {mape_improvement:.1f}%")
            report.append("")
            
            # R2 comparison
            report.append(f"Average R² Score:")
            report.append(f"  SARIMAX:  {summary['sarimax']['mean_r2']:.3f}")
            report.append(f"  Ensemble: {summary['ensemble']['mean_r2']:.3f}")
            report.append("")
            
            # Speed comparison
            report.append(f"Average Inference Time:")
            report.append(f"  SARIMAX:  {summary['sarimax']['mean_time']:.3f}s")
            report.append(f"  Ensemble: {summary['ensemble']['mean_time']:.3f}s")
            report.append("")
        
        # Individual model performance
        if 'xgboost' in summary:
            report.append("INDIVIDUAL MODEL PERFORMANCE:")
            report.append("-" * 40)
            report.append(f"XGBoost:")
            report.append(f"  RMSE: {summary['xgboost']['mean_rmse']:.2f}")
            report.append(f"  MAPE: {summary['xgboost']['mean_mape']:.2f}%")
            report.append("")
        
        if 'gradient_boosting' in summary:
            report.append(f"Gradient Boosting:")
            report.append(f"  RMSE: {summary['gradient_boosting']['mean_rmse']:.2f}")
            report.append(f"  MAPE: {summary['gradient_boosting']['mean_mape']:.2f}%")
            report.append("")
        
        # Model characteristics comparison
        report.append("MODEL CHARACTERISTICS:")
        report.append("-" * 40)
        report.append("SARIMAX (Classical Time Series):")
        report.append("  + Captures temporal dependencies well")
        report.append("  + Handles seasonality explicitly")
        report.append("  + Good interpretability")
        report.append("  - Limited feature handling")
        report.append("  - Slower inference")
        report.append("  - Requires stationarity")
        report.append("")
        
        report.append("Gradient Boosting Ensemble:")
        report.append("  + Handles many features")
        report.append("  + Fast inference")
        report.append("  + No stationarity requirement")
        report.append("  + Better accuracy on average")
        report.append("  - Less interpretable")
        report.append("  - Requires feature engineering")
        report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS:")
        report.append("-" * 40)
        
        if 'sarimax' in summary and 'ensemble' in summary:
            if ensemble_rmse < sarimax_rmse * 0.9:
                report.append("• Ensemble model shows significant improvement (>10%)")
                report.append("• Recommend using ensemble for production")
            elif sarimax_rmse < ensemble_rmse * 0.9:
                report.append("• SARIMAX shows better performance")
                report.append("• Consider using SARIMAX for this dataset")
            else:
                report.append("• Both models show comparable performance")
                report.append("• Consider ensemble approach for flexibility")
        
        report.append("• Implement model monitoring for drift detection")
        report.append("• Consider hybrid approach for different SKU types")
        report.append("")
        
        return "\n".join(report)


if __name__ == "__main__":
    # Test model comparison
    import sys
    sys.path.append('..')
    from data_loader import DataLoader
    from features import FeatureEngineer
    from models.sarimax_model import SARIMAXForecaster
    from models.ensemble_model import EnsembleForecaster
    
    # Load data
    loader = DataLoader()
    df = loader.load_data()
    train_df, test_df = loader.split_data()
    
    # Get limited combos for testing
    combos = loader.get_store_sku_combinations()[:2]
    
    # Train SARIMAX
    sarimax = SARIMAXForecaster()
    sarimax.fit_all(train_df, combos)
    
    # Create features and train ensemble
    fe = FeatureEngineer()
    train_features = fe.create_features(train_df)
    X_train, y_train = fe.prepare_for_modeling(train_features)
    
    ensemble = EnsembleForecaster(model_type='ensemble')
    ensemble.fit(X_train, y_train)
    
    # Compare models
    comparator = ModelComparison()
    results = comparator.compare_models(sarimax, ensemble, train_df, test_df, fe)
    
    # Generate report
    report = comparator.generate_report()
    print(report)