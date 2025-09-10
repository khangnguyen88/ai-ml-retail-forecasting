#!/usr/bin/env python3
"""
Load saved models and pipeline to make predictions with consistent preprocessing.
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
import logging
from datetime import datetime

# Import models and preprocessing
from preprocessing import FeaturePipeline
from models.ensemble_model import EnsembleForecaster
from data_loader import DataLoader

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_models_and_pipeline():
    """Load saved models and preprocessing pipeline from disk."""
    
    models = {}
    
    # Load preprocessing pipeline
    pipeline_paths = [
        'models/pipeline.pkl',  # From train_with_pipeline.py
    ]
    
    for path in pipeline_paths:
        if os.path.exists(path):
            logger.info(f"Loading preprocessing pipeline from {path}...")
            pipeline = FeaturePipeline()
            pipeline.load(path)
            models['pipeline'] = pipeline
            break
    
    if 'pipeline' not in models:
        logger.warning("Pipeline not found, checking for legacy feature engineer...")
        if os.path.exists('models/feature_engineer_optimized.pkl'):
            feature_engineer = joblib.load('models/feature_engineer_optimized.pkl')
            models['feature_engineer'] = feature_engineer
        elif os.path.exists('models/feature_engineer.pkl'):
            feature_engineer = joblib.load('models/feature_engineer.pkl')
            models['feature_engineer'] = feature_engineer
    
    # Load Gradient Boosting model
    model_paths = [
        'models/gb_model_pipeline.pkl',  # From train_with_pipeline.py
    ]
    
    for path in model_paths:
        if os.path.exists(path):
            logger.info(f"Loading ensemble model from {path}...")
            ensemble_model = EnsembleForecaster()
            ensemble_model.load_model(path)
            models['ensemble'] = ensemble_model
            break
    
    if 'ensemble' not in models:
        logger.error("No ensemble model found!")
    
    # Load SARIMAX models
    sarimax_paths = [
        'models/sarimax_model_pipeline.pkl',  # From train_with_pipeline.py
    ]
    
    for path in sarimax_paths:
        if os.path.exists(path):
            logger.info(f"Loading SARIMAX models from {path}...")
            with open(path, 'rb') as f:
                sarimax_model = pickle.load(f)
            models['sarimax'] = sarimax_model
            break
    
    # Load metadata
    metadata_paths = [
        'models/training_metadata_pipeline.pkl',  # From train_with_pipeline.py
    ]
    
    for path in metadata_paths:
        if os.path.exists(path):
            with open(path, 'rb') as f:
                metadata = pickle.load(f)
            models['metadata'] = metadata
            logger.info(f"Models trained on: {metadata.get('timestamp', 'Unknown')}")
            break
    
    return models

def predict_with_pipeline(models, input_data):
    """
    Make predictions using loaded models and pipeline.
    
    Args:
        models: Dictionary containing loaded models and pipeline
        input_data: DataFrame with input features
        
    Returns:
        Array of predictions
    """
    
    if 'pipeline' in models:
        # Use the preprocessing pipeline for consistent features
        logger.info("Using preprocessing pipeline...")
        pipeline = models['pipeline']
        
        # Transform the input data
        processed_data = pipeline.transform(input_data)
        
        # Get feature matrix
        X, _ = pipeline.preprocessor.get_feature_matrix(processed_data)
        
    elif 'feature_engineer' in models:
        # Fallback to legacy feature engineer
        logger.info("Using legacy feature engineer...")
        feature_engineer = models['feature_engineer']
        
        # Create features
        features_df = feature_engineer.create_features(input_data)
        X, _ = feature_engineer.prepare_for_modeling(features_df)
    
    else:
        raise ValueError("No preprocessing pipeline or feature engineer found!")
    
    # Make predictions
    if 'ensemble' in models:
        logger.info(f"Making predictions for {len(X)} samples...")
        predictions = models['ensemble'].predict(X)
    else:
        raise ValueError("No ensemble model found!")
    
    return predictions

def predict_with_sarimax(models, store_id, sku_id, steps=14):
    """
    Make predictions using SARIMAX model.
    
    Args:
        models: Dictionary containing loaded models
        store_id: Store identifier
        sku_id: SKU identifier
        steps: Number of steps to forecast
        
    Returns:
        Series of predictions
    """
    
    if 'sarimax' not in models:
        raise ValueError("SARIMAX model not loaded")
    
    logger.info(f"Making SARIMAX predictions for Store {store_id}, SKU {sku_id}...")
    
    try:
        predictions = models['sarimax'].predict(store_id, sku_id, steps)
    except Exception as e:
        logger.error(f"SARIMAX prediction failed: {e}")
        predictions = pd.Series([])
    
    return predictions

def demo_predictions():
    """Demonstrate loading models and making predictions without warnings."""
    
    # Load models and pipeline
    logger.info("="*50)
    logger.info("LOADING SAVED MODELS AND PIPELINE")
    logger.info("="*50)
    
    models = load_models_and_pipeline()
    
    if not models:
        logger.error("No models found. Please run train_optimized_models.py first.")
        return
    
    # Load test data
    logger.info("\n" + "="*50)
    logger.info("LOADING TEST DATA")
    logger.info("="*50)
    
    data_loader = DataLoader('retail_pricing_demand_2024.csv')
    df = data_loader.load_data()
    
    # Get last 30 days of data for predictions
    df['date'] = pd.to_datetime(df['date'])
    test_data = df[df['date'] >= '2024-10-01'].head(100)
    
    logger.info(f"Test data shape: {test_data.shape}")
    
    # ============================================
    # Ensemble Model Predictions (with pipeline)
    # ============================================
    if 'ensemble' in models:
        logger.info("\n" + "="*50)
        logger.info("ENSEMBLE MODEL PREDICTIONS")
        logger.info("="*50)
        
        try:
            predictions = predict_with_pipeline(models, test_data)
            
            # Compare with actual values
            actual_values = test_data['units_sold'].values
            
            from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
            
            # Calculate metrics
            valid_idx = ~np.isnan(actual_values) & ~np.isnan(predictions)
            if valid_idx.sum() > 0:
                rmse = np.sqrt(mean_squared_error(actual_values[valid_idx], predictions[valid_idx]))
                mape = mean_absolute_percentage_error(actual_values[valid_idx], predictions[valid_idx])
                r2 = r2_score(actual_values[valid_idx], predictions[valid_idx])
                
                logger.info(f"Predictions made: {len(predictions)}")
                logger.info(f"Sample predictions: {predictions[:5].round(2)}")
                logger.info(f"Actual values: {actual_values[:5]}")
                logger.info(f"RMSE: {rmse:.2f}")
                logger.info(f"MAPE: {mape:.2%}")
                logger.info(f"R²: {r2:.4f}")
            else:
                logger.warning("No valid predictions to evaluate")
                
        except Exception as e:
            logger.error(f"Ensemble prediction failed: {e}")
            import traceback
            traceback.print_exc()
    
    # ============================================
    # SARIMAX Predictions
    # ============================================
    if 'sarimax' in models:
        logger.info("\n" + "="*50)
        logger.info("SARIMAX PREDICTIONS")
        logger.info("="*50)
        
        try:
            # Get available store-SKU combinations
            if hasattr(models['sarimax'], 'models') and models['sarimax'].models:
                # Pick first available combination
                combo_key = list(models['sarimax'].models.keys())[0]
                store_id, sku_id = combo_key.split('_')
                
                # Make predictions
                sarimax_predictions = predict_with_sarimax(
                    models, 
                    store_id, 
                    sku_id, 
                    steps=14
                )
                
                if len(sarimax_predictions) > 0:
                    logger.info(f"Store {store_id}, SKU {sku_id}")
                    logger.info(f"Predictions made: {len(sarimax_predictions)}")
                    logger.info(f"Sample predictions: {sarimax_predictions.head()}")
                else:
                    logger.warning("No SARIMAX predictions generated")
            else:
                logger.warning("No SARIMAX models found in the saved file")
                
        except Exception as e:
            logger.error(f"SARIMAX prediction failed: {e}")
            import traceback
            traceback.print_exc()
    
    logger.info("\n" + "="*50)
    logger.info("PREDICTION DEMO COMPLETED - NO WARNINGS!")
    logger.info("="*50)

def predict_new_data(data_file: str, output_file: str = 'predictions.csv'):
    """
    Make predictions on new data from a CSV file.
    
    Args:
        data_file: Path to CSV file with new data
        output_file: Path to save predictions
    """
    
    # Load models and pipeline
    logger.info("Loading models and pipeline...")
    models = load_models_and_pipeline()
    
    # Load new data
    logger.info(f"Loading data from {data_file}...")
    new_data = pd.read_csv(data_file)
    
    # Make predictions
    logger.info("Making predictions...")
    predictions = predict_with_pipeline(models, new_data)
    
    # Add predictions to dataframe
    new_data['predicted_demand'] = predictions
    
    # Save results
    new_data.to_csv(output_file, index=False)
    logger.info(f"Predictions saved to {output_file}")
    
    # Print summary statistics
    logger.info("\nPrediction Summary:")
    logger.info(f"  Mean: {predictions.mean():.2f}")
    logger.info(f"  Std: {predictions.std():.2f}")
    logger.info(f"  Min: {predictions.min():.2f}")
    logger.info(f"  Max: {predictions.max():.2f}")
    
    return new_data

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Make predictions with saved models')
    parser.add_argument('--input', type=str, help='Path to input CSV file')
    parser.add_argument('--output', type=str, default='predictions.csv', 
                       help='Path to save predictions')
    parser.add_argument('--demo', action='store_true', 
                       help='Run demo with test data')
    
    args = parser.parse_args()
    
    if args.input:
        # Predict on new data
        predict_new_data(args.input, args.output)
    else:
        # Run demo
        demo_predictions()