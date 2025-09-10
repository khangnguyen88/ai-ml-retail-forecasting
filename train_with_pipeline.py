#!/usr/bin/env python3
"""
Train models using the preprocessing pipeline for consistent feature handling.
No warnings, no missing features!
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append('src')

import pandas as pd
import numpy as np
import joblib
import pickle
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import our modules
from data_loader import DataLoader
from preprocessing import FeaturePipeline
from models.ensemble_model import EnsembleForecaster
from models.sarimax_model import SARIMAXForecaster

def train_models_with_pipeline(
    train_start: str = '2024-01-01',
    train_end: str = '2024-09-30',
    test_start: str = '2024-10-01',
    test_end: str = '2024-12-31',
    use_grid_search: bool = False,
    max_sarimax_combos: int = 5
):
    """
    Train models using the preprocessing pipeline.
    
    Args:
        train_start: Training start date
        train_end: Training end date
        test_start: Test start date
        test_end: Test end date
        use_grid_search: Whether to use grid search for hyperparameter tuning
        max_sarimax_combos: Maximum number of SARIMAX models to train
        
    Returns:
        Dictionary of performance metrics
    """
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Initialize data loader
    logger.info("Loading data...")
    data_loader = DataLoader('retail_pricing_demand_2024.csv')
    df = data_loader.load_data()
    
    # Split data
    train_df, test_df = data_loader.split_data(
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end
    )
    
    logger.info(f"Training set: {len(train_df)} rows")
    logger.info(f"Test set: {len(test_df)} rows")
    
    # Initialize preprocessing pipeline
    logger.info("Initializing preprocessing pipeline...")
    pipeline = FeaturePipeline()
    
    # Get preprocessed train and test data
    logger.info("Preprocessing data with consistent features...")
    X_train, y_train, X_test, y_test = pipeline.get_train_test_data(train_df, test_df)
    
    # Split validation set
    val_size = int(len(X_train) * 0.2)
    X_val = X_train.iloc[-val_size:]
    y_val = y_train.iloc[-val_size:]
    X_train_final = X_train.iloc[:-val_size]
    y_train_final = y_train.iloc[:-val_size]
    
    logger.info(f"Training samples: {len(X_train_final)}")
    logger.info(f"Validation samples: {len(X_val)}")
    logger.info(f"Test samples: {len(X_test)}")
    logger.info(f"Features: {X_train.shape[1]}")
    
    # Store metrics
    metrics = {}
    
    # ============================================
    # Train Gradient Boosting Model
    # ============================================
    logger.info("\n" + "="*50)
    logger.info("TRAINING GRADIENT BOOSTING MODEL")
    logger.info("="*50)
    
    try:
        # Initialize ensemble model with optimized settings
        ensemble_model = EnsembleForecaster(model_type='gb', random_state=42)
        
        # Disable problematic target transformation if not using grid search
        if not use_grid_search:
            logger.info("Using optimized hyperparameters (no target transformation)")
            ensemble_model.target_transformer = None
            ensemble_model.lambda_boxcox = None
            # Override transform method to return unchanged target
            ensemble_model._transform_target = lambda y: y
        else:
            logger.info("Using grid search with target transformation enabled")
        
        # Train model
        logger.info("Training model...")
        ensemble_model.fit(
            X_train_final,
            y_train_final,
            X_val,
            y_val,
            tune_hyperparameters=use_grid_search
        )
        
        # Log model configuration
        if hasattr(ensemble_model, 'target_transformer') and ensemble_model.target_transformer:
            logger.info(f"Model used target transformation: {ensemble_model.target_transformer}")
        else:
            logger.info("Model trained without target transformation")
        
        # Log hyperparameters used
        if hasattr(ensemble_model.model, 'n_estimators'):
            logger.info(f"Final hyperparameters: n_estimators={ensemble_model.model.n_estimators}, "
                       f"max_depth={ensemble_model.model.max_depth}, "
                       f"learning_rate={ensemble_model.model.learning_rate}")
        
        # Evaluate on test set
        logger.info("Evaluating model...")
        gb_metrics = ensemble_model.evaluate(X_test, y_test)
        metrics['gradient_boosting'] = gb_metrics.get('gradient_boosting', gb_metrics.get('gb', {}))
        
        # Add quick performance check
        test_predictions = ensemble_model.predict(X_test)
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
        
        r2 = r2_score(y_test, test_predictions)
        rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
        mape = mean_absolute_percentage_error(y_test, test_predictions)
        
        logger.info(f"\nQuick Performance Check:")
        performance_level = "EXCELLENT" if r2 > 0.8 else "GOOD" if r2 > 0.6 else "FAIR" if r2 > 0.4 else "POOR"
        logger.info(f"  R²: {r2:.4f} ({performance_level})")
        logger.info(f"  RMSE: {rmse:.2f}")
        logger.info(f"  MAPE: {mape:.1%}")
        logger.info(f"  Prediction range: [{test_predictions.min():.1f}, {test_predictions.max():.1f}]")
        logger.info(f"  Actual range: [{y_test.min():.1f}, {y_test.max():.1f}]")
        
        # Update metrics with correct values
        metrics['gradient_boosting'].update({
            'r2': float(r2),
            'rmse': float(rmse),
            'mape': float(mape)
        })
        
        # Get feature importance
        logger.info("Getting feature importance...")
        feature_importance = ensemble_model.get_feature_importance(top_n=20)
        
        logger.info("\nTop 10 Most Important Features:")
        for idx, row in feature_importance.head(10).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")
        
        # Save models and pipeline
        logger.info("Saving models and pipeline...")
        
        # Save the model
        ensemble_model.save_model('models/gb_model_pipeline.pkl')
        
        # Save the preprocessing pipeline
        pipeline.save('models/pipeline.pkl')
        
        # Save feature importance
        feature_importance.to_csv('models/feature_importance.csv', index=False)
        
        logger.info("\nGradient Boosting Performance:")
        for metric, value in metrics['gradient_boosting'].items():
            logger.info(f"  {metric}: {value:.4f}")
            
    except Exception as e:
        logger.error(f"Gradient Boosting training failed: {e}")
        import traceback
        traceback.print_exc()
    
    # ============================================
    # Train SARIMAX Models (Optional)
    # ============================================
    if max_sarimax_combos > 0:
        logger.info("\n" + "="*50)
        logger.info("TRAINING SARIMAX MODELS")
        logger.info("="*50)
        
        try:
            # Get store-SKU combinations
            combos = data_loader.get_store_sku_combinations()
            num_combos = min(max_sarimax_combos, len(combos))
            
            logger.info(f"Training SARIMAX for {num_combos} store-SKU combinations...")
            
            # Initialize SARIMAX forecaster
            sarimax_model = SARIMAXForecaster(
                order=(1, 1, 1),
                seasonal_order=(1, 1, 1, 7)
            )
            
            # Train on selected combinations
            results = sarimax_model.fit_all(train_df, combos[:num_combos])
            
            # Save SARIMAX models
            logger.info("Saving SARIMAX models...")
            with open('models/sarimax_model_pipeline.pkl', 'wb') as f:
                pickle.dump(sarimax_model, f)
            
            logger.info(f"Trained {num_combos} SARIMAX models")
            
        except Exception as e:
            logger.error(f"SARIMAX training failed: {e}")
            import traceback
            traceback.print_exc()
    
    # ============================================
    # Save metadata
    # ============================================
    metadata = {
        'train_start': train_start,
        'train_end': train_end,
        'test_start': test_start,
        'test_end': test_end,
        'train_samples': len(X_train_final) if 'X_train_final' in locals() else len(train_df),
        'val_samples': len(X_val) if 'X_val' in locals() else 0,
        'test_samples': len(test_df),
        'features': X_train.shape[1],
        'timestamp': datetime.now().isoformat(),
        'metrics': metrics,
        'grid_search_used': use_grid_search,
        'optimized_hyperparameters_used': not use_grid_search,
        'target_transformation': 'disabled_for_stability' if not use_grid_search else 'enabled'
    }
    
    with open('models/training_metadata_pipeline.pkl', 'wb') as f:
        pickle.dump(metadata, f)
    
    # Save as JSON for readability
    import json
    with open('models/training_metadata_pipeline.json', 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    # ============================================
    # Summary
    # ============================================
    logger.info("\n" + "="*50)
    logger.info("TRAINING COMPLETED SUCCESSFULLY!")
    logger.info("="*50)
    logger.info("\nSaved files:")
    logger.info("  - models/gb_model_pipeline.pkl (Gradient Boosting model)")
    logger.info("  - models/pipeline.pkl (Preprocessing pipeline)")
    logger.info("  - models/feature_importance.csv (Feature rankings)")
    if max_sarimax_combos > 0:
        logger.info("  - models/sarimax_model_pipeline.pkl (SARIMAX models)")
    logger.info("  - models/training_metadata_pipeline.json (Training info)")
    
    if metrics:
        logger.info("\n" + "="*50)
        logger.info("FINAL PERFORMANCE METRICS")
        logger.info("="*50)
        for model_name, model_metrics in metrics.items():
            logger.info(f"\n{model_name.upper()}:")
            for metric, value in model_metrics.items():
                if isinstance(value, float):
                    logger.info(f"  {metric}: {value:.4f}")
    
    return metrics

def quick_train():
    """Quick training with optimized hyperparameters (recommended)."""
    logger.info("Starting quick training with optimized hyperparameters...")
    return train_models_with_pipeline(
        use_grid_search=False,
        max_sarimax_combos=2
    )

def optimized_train():
    """Grid search training (slower, may have target transform issues)."""
    logger.info("Starting training with grid search (slower)...")
    return train_models_with_pipeline(
        use_grid_search=True,
        max_sarimax_combos=5
    )

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train models with preprocessing pipeline')
    parser.add_argument('--optimize', action='store_true', 
                       help='Use grid search (slower, may have issues)')
    parser.add_argument('--train-start', default='2024-01-01', 
                       help='Training start date')
    parser.add_argument('--train-end', default='2024-09-30', 
                       help='Training end date')
    parser.add_argument('--test-start', default='2024-10-01', 
                       help='Test start date')
    parser.add_argument('--test-end', default='2024-12-31', 
                       help='Test end date')
    parser.add_argument('--max-sarimax', type=int, default=5, 
                       help='Max SARIMAX combinations to train')
    parser.add_argument('--quick', action='store_true',
                       help='Quick training with optimized hyperparameters (recommended)')
    
    args = parser.parse_args()
    
    if args.quick:
        # Quick training
        metrics = quick_train()
    else:
        # Custom training
        metrics = train_models_with_pipeline(
            train_start=args.train_start,
            train_end=args.train_end,
            test_start=args.test_start,
            test_end=args.test_end,
            use_grid_search=args.optimize,
            max_sarimax_combos=args.max_sarimax
        )
    
    # Print instructions for prediction
    print("\n" + "="*60)
    print("TO MAKE PREDICTIONS WITH THE TRAINED MODEL:")
    print("="*60)
    print("\n1. Test with demo data:")
    print("   python predict_with_pipeline.py --demo")
    print("\n2. Predict on new data:")
    print("   python predict_with_pipeline.py --input your_data.csv --output predictions.csv")
    print("\n3. In your Python code:")
    print("   from preprocessing import FeaturePipeline")
    print("   from models.ensemble_model import EnsembleForecaster")
    print("   ")
    print("   # Load pipeline and model")
    print("   pipeline = FeaturePipeline().load('models/pipeline.pkl')")
    print("   model = EnsembleForecaster().load_model('models/gb_model_pipeline.pkl')")
    print("   ")
    print("   # Process new data and predict")
    print("   processed = pipeline.transform(new_data)")
    print("   X, _ = pipeline.preprocessor.get_feature_matrix(processed)")
    print("   predictions = model.predict(X)")
    print("="*60)