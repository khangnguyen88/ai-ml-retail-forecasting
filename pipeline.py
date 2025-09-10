#!/usr/bin/env python3
"""
Main CLI pipeline for pricing optimization and model comparison.

Note: For training, use train_with_pipeline.py instead

Usage:
    python pipeline.py simulate --horizon 14 --price-plan price_plan.csv --out results.csv
    python pipeline.py optimize --horizon 14 --objective revenue --constraints constraints.yaml --out plan.csv
    python pipeline.py compare --output comparison_report.txt
"""

import argparse
import pandas as pd
import numpy as np
import yaml
import json
import sys
import os
from datetime import datetime, timedelta
import logging
import warnings

# Add src to path
sys.path.append('src')

from data_loader import DataLoader
from features import FeatureEngineer
from preprocessing import FeaturePipeline
from models.sarimax_model import SARIMAXForecaster
from models.ensemble_model import EnsembleForecaster
from models.model_comparison import ModelComparison
from pricing.optimizer import PricingOptimizer, PricingConstraints
# Create mock classes for when MLflow is not available
class MockMLflowTracker:
    def start_run(self, run_name=None): pass
    def log_params(self, params): pass
    def log_metrics(self, metrics): pass
    def log_artifact(self, path): pass
    def end_run(self, status=None): pass

class MockExperimentRunner:
    pass

# MLflow is optional - disable if not available
try:
    from ml_tracking import MLflowTracker as RealMLflowTracker, ExperimentRunner as RealExperimentRunner
    MLFLOW_AVAILABLE = True
    MLflowTracker = RealMLflowTracker
    ExperimentRunner = RealExperimentRunner
except ImportError:
    MLFLOW_AVAILABLE = False
    MLflowTracker = MockMLflowTracker
    ExperimentRunner = MockExperimentRunner

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Pipeline:
    """Main pipeline for simulation, optimization, and comparison."""
    
    def __init__(self, config_path: str = None, enable_mlflow: bool = True):
        """Initialize pipeline.
        
        Args:
            config_path: Path to configuration file
            enable_mlflow: Whether to enable MLflow tracking
        """
        self.config = self._load_config(config_path)
        self.data_loader = DataLoader()
        self.feature_engineer = FeatureEngineer()
        self.feature_pipeline = FeaturePipeline()  # New preprocessing pipeline
        self.sarimax_model = None
        self.ensemble_model = None
        self.pricing_optimizer = None
        
        # Initialize MLflow tracking
        if enable_mlflow and MLFLOW_AVAILABLE:
            self.enable_mlflow = True
            self.mlflow_tracker = MLflowTracker(
                tracking_uri=self.config.get('mlflow_tracking_uri', 'mlruns'),
                experiment_name=self.config.get('mlflow_experiment', 'demand-forecasting')
            )
            self.experiment_runner = ExperimentRunner(self.mlflow_tracker)
        else:
            self.enable_mlflow = False
            self.mlflow_tracker = MockMLflowTracker()
            self.experiment_runner = MockExperimentRunner()
            if enable_mlflow and not MLFLOW_AVAILABLE:
                logger.warning("MLflow not available, running without tracking")
        
    def _load_config(self, config_path: str = None) -> dict:
        """Load configuration from file or use defaults.
        
        Args:
            config_path: Path to config file
            
        Returns:
            Configuration dictionary
        """
        default_config = {
            'data_path': 'retail_pricing_demand_2024.csv',
            'model_type': 'ensemble',  # 'sarimax', 'ensemble', or 'both'
            'ensemble_type': 'gb',  # 'gb' for GradientBoosting, 'rf' for RandomForest
            'elasticity': -1.5,
            'cost_ratio': 0.6,  # Cost as ratio of base price
            'random_state': 42
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                if config_path.endswith('.yaml'):
                    config = yaml.safe_load(f)
                else:
                    config = json.load(f)
            default_config.update(config)
        
        return default_config
    
    def simulate(self,
                horizon: int,
                price_plan_path: str = None,
                output_path: str = 'simulation_results.csv'):
        """Simulate demand for given price plan.
        
        Args:
            horizon: Forecast horizon in days
            price_plan_path: Path to price plan CSV
            output_path: Path to save results
        """
        logger.info(f"Simulating demand for {horizon} days...")
        
        # Load models if not already loaded
        if self.ensemble_model is None:
            self.ensemble_model = EnsembleForecaster()
            self.ensemble_model.load_model('models/gb_model_pipeline.pkl')
        
        # Load price plan or use current prices
        if price_plan_path:
            price_plan = pd.read_csv(price_plan_path)
        else:
            # Use current prices from data
            df = self.data_loader.load_data()
            recent_data = df.groupby(['store_id', 'sku_id']).tail(1)
            price_plan = recent_data[['store_id', 'sku_id', 'final_price', 'base_price']]
        
        # Initialize pricing optimizer
        self.pricing_optimizer = PricingOptimizer(
            demand_model=self.ensemble_model,
            elasticity=self.config['elasticity']
        )
        
        # Load recent data for base demand estimates
        df = self.data_loader.load_data()
        recent_data = df.groupby(['store_id', 'sku_id']).tail(30).groupby(['store_id', 'sku_id'])['units_sold'].mean().reset_index()
        base_demand_dict = dict(zip(zip(recent_data['store_id'], recent_data['sku_id']), recent_data['units_sold']))
        
        # Simulate for each row in the price plan (each store-SKU-day combination)
        results = []
        for _, row in price_plan.iterrows():
            store_sku_key = (row.get('store_id', 'S01'), row.get('sku_id', 'SKU001'))
            base_demand = base_demand_dict.get(store_sku_key, 30)  # Use actual base demand
            
            # Create features for this specific day
            features = {
                'store_id': row.get('store_id', 'S01'),
                'sku_id': row.get('sku_id', 'SKU001'),
                'promo_flag': row.get('promo_flag', 0),
                'holiday_flag': row.get('holiday_flag', 0),
                'weather_index': row.get('weather_index', 0.5),
                'day_of_week': row.get('day_of_week', 0),
                'is_weekend': row.get('is_weekend', 0),
                'day': row.get('day', 1)
            }
            
            # Simulate demand with day-specific features
            demand = self.pricing_optimizer.simulate_demand(
                price=row.get('optimal_price', row.get('final_price', 10)),
                base_price=row.get('base_price', 10),
                base_demand=base_demand,
                features=features
            )
            
            # Calculate KPIs
            cost = row.get('base_price', 10) * self.config['cost_ratio']
            kpis = self.pricing_optimizer.calculate_kpis(
                price=row.get('optimal_price', row.get('final_price', 10)),
                demand=demand,
                cost=cost
            )
            
            # Add to results with all the plan data
            result = {
                'store_id': row.get('store_id', 'S01'),
                'sku_id': row.get('sku_id', 'SKU001'),
                'day': row.get('day', 1),
                'date': datetime.now() + timedelta(days=row.get('day', 1)-1),
                'price': row.get('optimal_price', row.get('final_price', 10)),
                'base_price': row.get('base_price', 10),
                'promo_flag': features['promo_flag'],
                'holiday_flag': features['holiday_flag'], 
                'weather_index': features['weather_index'],
                **kpis
            }
            results.append(result)
        
        # Create results DataFrame (no need to duplicate - already has all day-specific data)
        final_results = pd.DataFrame(results)
        
        # Save results
        final_results.to_csv(output_path, index=False)
        logger.info(f"Simulation results saved to {output_path}")
        
        # Print summary
        total_revenue = final_results['revenue'].sum()
        total_units = final_results['units_sold'].sum()
        avg_price = final_results['price'].mean()
        
        logger.info("\n=== Simulation Summary ===")
        logger.info(f"Total Revenue: ${total_revenue:,.2f}")
        logger.info(f"Total Units: {total_units:,.0f}")
        logger.info(f"Average Price: ${avg_price:.2f}")
        
        return final_results
    
    def optimize(self,
                horizon: int,
                objective: str,
                constraints_path: str = None,
                output_path: str = 'optimized_plan.csv'):
        """Optimize pricing plan.
        
        Args:
            horizon: Planning horizon in days
            objective: Optimization objective ('revenue', 'units', 'multi')
            constraints_path: Path to constraints file
            output_path: Path to save optimized plan
        """
        logger.info(f"Optimizing prices for {horizon} days with objective: {objective}")
        
        # Start MLflow run for optimization
        if self.enable_mlflow:
            run_name = f"optimization_{objective}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.mlflow_tracker.start_run(run_name=run_name)
            
            # Log parameters
            self.mlflow_tracker.log_params({
                'horizon': horizon,
                'objective': objective,
                'elasticity': self.config['elasticity']
            })
        
        # Load data
        df = self.data_loader.load_data()
        recent_data = df.groupby(['store_id', 'sku_id']).tail(30)
        
        # Load constraints
        if constraints_path and os.path.exists(constraints_path):
            with open(constraints_path, 'r') as f:
                constraints_dict = yaml.safe_load(f)
            constraints = PricingConstraints(**constraints_dict)
        else:
            constraints = PricingConstraints()
        
        # Initialize optimizer
        self.pricing_optimizer = PricingOptimizer(
            demand_model=self.ensemble_model,
            elasticity=self.config['elasticity'],
            constraints=constraints
        )
        
        # Optimize price plan
        optimized_plan = self.pricing_optimizer.optimize_price_plan(
            data=recent_data,
            horizon=horizon,
            objective=objective
        )
        
        # Validate constraints
        is_valid, violations = self.pricing_optimizer.validate_constraints(optimized_plan)
        
        # Calculate optimization metrics
        baseline_revenue = recent_data['revenue'].sum() * horizon / 30  # Scale to horizon
        optimized_revenue = optimized_plan['expected_revenue'].sum()
        revenue_lift = (optimized_revenue / baseline_revenue - 1) * 100
        
        baseline_units = recent_data['units_sold'].sum() * horizon / 30
        optimized_units = optimized_plan['expected_demand'].sum()
        units_lift = (optimized_units / baseline_units - 1) * 100
        
        # Log metrics to MLflow
        if self.enable_mlflow:
            self.mlflow_tracker.log_metrics({
                'baseline_revenue': baseline_revenue,
                'optimized_revenue': optimized_revenue,
                'revenue_lift_pct': revenue_lift,
                'baseline_units': baseline_units,
                'optimized_units': optimized_units,
                'units_lift_pct': units_lift,
                'constraints_valid': int(is_valid),
                'num_violations': len(violations) if violations else 0
            })
            
            # Log constraints to MLflow
            if constraints_path:
                self.mlflow_tracker.log_artifact(constraints_path)
        
        # Save plan
        optimized_plan.to_csv(output_path, index=False)
        logger.info(f"Optimized price plan saved to {output_path}")
        
        # Log plan to MLflow
        if self.enable_mlflow:
            self.mlflow_tracker.log_artifact(output_path)
        
        # Print summary
        logger.info("\n=== Optimization Summary ===")
        logger.info(f"Objective: {objective}")
        logger.info(f"Horizon: {horizon} days")
        logger.info(f"Constraints valid: {is_valid}")
        logger.info(f"Revenue Lift: {revenue_lift:.2f}%")
        logger.info(f"Units Lift: {units_lift:.2f}%")
        
        # End MLflow run
        if self.enable_mlflow:
            self.mlflow_tracker.end_run(status="FINISHED")
        if violations:
            logger.warning("Constraint violations:")
            for v in violations:
                logger.warning(f"  - {v}")
        
        # Calculate expected lift
        avg_lift = optimized_plan['revenue_lift'].mean()
        total_revenue = optimized_plan['expected_revenue'].sum()
        
        logger.info(f"Average Revenue Lift: {avg_lift:.1f}%")
        logger.info(f"Total Expected Revenue: ${total_revenue:,.2f}")
        
        return optimized_plan
    
    def compare(self, output_path: str = 'comparison_report.txt'):
        """Compare classical and ensemble models.
        
        Args:
            output_path: Path to save comparison report
        """
        logger.info("Comparing models...")
        
        # Load data
        df = self.data_loader.load_data()
        train_df, test_df = self.data_loader.split_data()
        
        # Load models
        import joblib
        if os.path.exists('models/sarimax_model.pkl'):
            self.sarimax_model = joblib.load('models/sarimax_model.pkl')
        else:
            logger.warning("SARIMAX model not found, training new one...")
            self.sarimax_model = SARIMAXForecaster()
            combos = self.data_loader.get_store_sku_combinations()[:5]
            self.sarimax_model.fit_all(train_df, combos)
        
        if os.path.exists('models/gb_model_pipeline.pkl'):
            self.ensemble_model = EnsembleForecaster()
            self.ensemble_model.load_model('models/gb_model_pipeline.pkl')
            # Load preprocessing pipeline if available
            if os.path.exists('models/pipeline.pkl'):
                self.feature_pipeline.load('models/pipeline.pkl')
        else:
            logger.warning("Ensemble model not found, training new one...")
            # Use preprocessing pipeline for consistent features
            X_train, y_train, X_test, y_test = self.feature_pipeline.get_train_test_data(train_df, test_df)
            self.ensemble_model = EnsembleForecaster(model_type='gb')
            self.ensemble_model.fit(X_train, y_train)
        
        # Compare models
        comparator = ModelComparison()
        results = comparator.compare_models(
            self.sarimax_model,
            self.ensemble_model,
            train_df,
            test_df,
            self.feature_engineer
        )
        
        # Generate report
        report = comparator.generate_report()
        
        # Save report
        with open(output_path, 'w') as f:
            f.write(report)
        
        logger.info(f"Comparison report saved to {output_path}")
        print("\n" + report)
        
        # Create visualizations
        comparator.plot_comparison(save_path='comparison_plot.png')
        
        return results


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description='Price Optimization and Model Comparison Pipeline (use train_with_pipeline.py for training)')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Simulate command
    simulate_parser = subparsers.add_parser('simulate', help='Simulate demand for price plan')
    simulate_parser.add_argument('--horizon', type=int, required=True, help='Forecast horizon (days)')
    simulate_parser.add_argument('--price-plan', help='Path to price plan CSV')
    simulate_parser.add_argument('--out', default='simulation_results.csv', help='Output path')
    
    # Optimize command
    optimize_parser = subparsers.add_parser('optimize', help='Optimize pricing')
    optimize_parser.add_argument('--horizon', type=int, required=True, help='Planning horizon (days)')
    optimize_parser.add_argument('--objective', choices=['revenue', 'units', 'multi'], 
                                required=True, help='Optimization objective')
    optimize_parser.add_argument('--constraints', help='Path to constraints YAML')
    optimize_parser.add_argument('--out', default='optimized_plan.csv', help='Output path')
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare models')
    compare_parser.add_argument('--output', default='comparison_report.txt', help='Report output path')
    
    # Config option for all commands
    parser.add_argument('--config', help='Path to configuration file')
    
    # MLflow options
    parser.add_argument('--mlflow', action='store_true', default=False,
                       help='Enable MLflow tracking (requires MLflow installation)')
    parser.add_argument('--mlflow-uri', default='mlruns',
                       help='MLflow tracking URI (default: mlruns)')
    parser.add_argument('--mlflow-experiment', default='demand-forecasting',
                       help='MLflow experiment name')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Update config with MLflow settings if provided
    config = {}
    if args.mlflow:
        config['mlflow_tracking_uri'] = args.mlflow_uri
        config['mlflow_experiment'] = args.mlflow_experiment
    
    # Initialize pipeline
    pipeline = Pipeline(config_path=args.config, enable_mlflow=args.mlflow)
    
    # Override config with CLI args
    if args.mlflow:
        pipeline.config.update(config)
    
    # Execute command
    if args.command == 'simulate':
        pipeline.simulate(
            horizon=args.horizon,
            price_plan_path=args.price_plan,
            output_path=args.out
        )
    elif args.command == 'optimize':
        pipeline.optimize(
            horizon=args.horizon,
            objective=args.objective,
            constraints_path=args.constraints,
            output_path=args.out
        )
    elif args.command == 'compare':
        pipeline.compare(output_path=args.output)


if __name__ == '__main__':
    main()