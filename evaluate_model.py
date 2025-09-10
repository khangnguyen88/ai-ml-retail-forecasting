#!/usr/bin/env python3
"""
Load and evaluate the trained ensemble model performance.
"""

import sys
import os
import pandas as pd
import numpy as np
import json
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score

# Add src to path
sys.path.append('src')

from preprocessing import FeaturePipeline
from models.ensemble_model import EnsembleForecaster
from data_loader import DataLoader

def load_and_evaluate_model():
    """Load the trained model and evaluate its performance."""
    
    print("Loading trained ensemble model...")
    
    # Load the preprocessing pipeline
    try:
        pipeline = FeaturePipeline()
        pipeline.load('models/pipeline.pkl')
        print("SUCCESS: Preprocessing pipeline loaded successfully")
    except Exception as e:
        print(f"ERROR: Error loading pipeline: {e}")
        return
    
    # Load the ensemble model
    try:
        model = EnsembleForecaster()
        model.load_model('models/gb_model_pipeline.pkl')
        print("SUCCESS: Gradient Boosting model loaded successfully")
    except Exception as e:
        print(f"ERROR: Error loading model: {e}")
        return
    
    # Load training metadata
    try:
        with open('models/training_metadata_pipeline.json', 'r') as f:
            metadata = json.load(f)
        print("SUCCESS: Training metadata loaded successfully")
    except Exception as e:
        print(f"ERROR: Error loading metadata: {e}")
        return
    
    # Load and prepare test data
    print("\nLoading test data...")
    try:
        data_loader = DataLoader()
        df = data_loader.load_data()
        
        # Split data using same dates as training
        train_df, test_df = data_loader.split_data(
            train_start=metadata['train_start'],
            train_end=metadata['train_end'],
            test_start=metadata['test_start'],
            test_end=metadata['test_end']
        )
        
        print(f"SUCCESS: Test data loaded: {len(test_df)} samples")
        
    except Exception as e:
        print(f"ERROR: Error loading data: {e}")
        return
    
    # Process test data through pipeline
    print("\nProcessing test data...")
    try:
        # Transform test data
        test_processed = pipeline.transform(test_df)
        
        # Get feature matrix
        X_test, y_test = pipeline.preprocessor.get_feature_matrix(test_processed, 'units_sold')
        
        print(f"SUCCESS: Test data processed: {X_test.shape}")
        
    except Exception as e:
        print(f"ERROR: Error processing test data: {e}")
        return
    
    # Make predictions
    print("\nMaking predictions...")
    try:
        predictions = model.predict(X_test)
        print(f"SUCCESS: Predictions generated: {len(predictions)} values")
        
    except Exception as e:
        print(f"ERROR: Error making predictions: {e}")
        return
    
    # Calculate performance metrics
    print("\nCalculating performance metrics...")
    
    try:
        # Core metrics
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mape = mean_absolute_percentage_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        mae = np.mean(np.abs(y_test - predictions))
        
        # Additional metrics
        mean_error = np.mean(predictions - y_test)  # Bias
        median_error = np.median(predictions - y_test)
        coverage_90 = np.mean((predictions > 0.9 * y_test) & (predictions < 1.1 * y_test))
        
        # Prediction statistics
        pred_mean = np.mean(predictions)
        pred_std = np.std(predictions)
        actual_mean = np.mean(y_test)
        actual_std = np.std(y_test)
        
    except Exception as e:
        print(f"ERROR: Error calculating metrics: {e}")
        return
    
    # Display results
    print("\n" + "="*60)
    print("MODEL PERFORMANCE RESULTS")
    print("="*60)
    
    print(f"\nDataset Information:")
    print(f"   Training samples: {metadata['train_samples']:,}")
    print(f"   Test samples: {metadata['test_samples']:,}")
    print(f"   Features used: {metadata['features']}")
    print(f"   Training period: {metadata['train_start']} to {metadata['train_end']}")
    print(f"   Test period: {metadata['test_start']} to {metadata['test_end']}")
    
    print(f"\nCore Performance Metrics:")
    performance_level = "EXCELLENT" if r2 > 0.8 else "GOOD" if r2 > 0.6 else "FAIR"
    print(f"   R² Score: {r2:.4f} ({performance_level})")
    print(f"   RMSE: {rmse:.2f}")
    print(f"   MAE: {mae:.2f}")
    print(f"   MAPE: {mape:.1%}")
    
    print(f"\nError Analysis:")
    print(f"   Mean Error (Bias): {mean_error:+.2f}")
    print(f"   Median Error: {median_error:+.2f}")
    print(f"   90% Coverage: {coverage_90:.1%}")
    
    print(f"\nPrediction vs Actual:")
    print(f"   Predicted Mean: {pred_mean:.1f} (σ={pred_std:.1f})")
    print(f"   Actual Mean: {actual_mean:.1f} (σ={actual_std:.1f})")
    
    # Model details
    print(f"\nModel Configuration:")
    print(f"   Model Type: Gradient Boosting")
    print(f"   Hyperparameter Tuning: {'Yes' if metadata.get('grid_search_used', False) else 'No'}")
    if hasattr(model, 'target_transformer') and model.target_transformer:
        print(f"   Target Transformation: {model.target_transformer}")
    
    # Feature importance top 10
    print(f"\nTop 10 Most Important Features:")
    try:
        feature_importance_df = pd.read_csv('models/feature_importance.csv')
        for i, (_, row) in enumerate(feature_importance_df.head(10).iterrows(), 1):
            print(f"   {i:2d}. {row['feature']:25s} {row['importance']:6.3f}")
    except:
        print("   Feature importance data not available")
    
    # Performance interpretation
    print(f"\nPerformance Assessment:")
    if r2 > 0.85:
        print("   EXCELLENT - Model explains >85% of variance")
    elif r2 > 0.70:
        print("   VERY GOOD - Model explains >70% of variance")
    elif r2 > 0.60:
        print("   GOOD - Model explains >60% of variance")
    elif r2 > 0.40:
        print("   FAIR - Model explains >40% of variance")
    else:
        print("   POOR - Model explains <40% of variance")
    
    if mape < 0.15:
        print("   EXCELLENT - MAPE < 15%")
    elif mape < 0.25:
        print("   GOOD - MAPE < 25%")
    elif mape < 0.35:
        print("   FAIR - MAPE < 35%")
    else:
        print("   POOR - MAPE > 35%")
        
    
    return {
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'mean_error': mean_error,
        'coverage_90': coverage_90
    }

if __name__ == '__main__':
    load_and_evaluate_model()