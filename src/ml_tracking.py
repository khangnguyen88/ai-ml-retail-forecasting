"""
MLflow Tracking Module for Demand Forecasting & Pricing Optimization

This module provides experiment tracking, model registry, and metric logging
capabilities for the entire ML pipeline.
"""

# Make MLflow imports optional
try:
    import mlflow
    import mlflow.sklearn
    import mlflow.lightgbm
    import mlflow.statsmodels
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    mlflow = None
    MlflowClient = None

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Union
import json
import yaml
import logging
from pathlib import Path
import hashlib
from datetime import datetime

logger = logging.getLogger(__name__)


class MLflowTracker:
    """
    MLflow tracking wrapper for experiment management
    """
    
    def __init__(self, 
                 tracking_uri: str = "mlruns",
                 experiment_name: str = "demand-forecasting",
                 artifact_location: Optional[str] = None):
        """
        Initialize MLflow tracking
        
        Args:
            tracking_uri: MLflow tracking server URI or local path
            experiment_name: Name of the experiment
            artifact_location: Optional custom artifact storage location
        """
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        
        # Set tracking URI
        mlflow.set_tracking_uri(tracking_uri)
        
        # Create or get experiment
        self.experiment = self._setup_experiment(experiment_name, artifact_location)
        self.experiment_id = self.experiment.experiment_id
        
        # Set experiment as active
        mlflow.set_experiment(experiment_name)
        
        # Initialize client for advanced operations
        self.client = MlflowClient(tracking_uri=tracking_uri)
        
        logger.info(f"MLflow tracking initialized: {tracking_uri}")
        logger.info(f"Experiment: {experiment_name} (ID: {self.experiment_id})")
    
    def _setup_experiment(self, name: str, artifact_location: Optional[str] = None):
        """Setup or get existing experiment"""
        try:
            experiment = mlflow.get_experiment_by_name(name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(
                    name, 
                    artifact_location=artifact_location
                )
                experiment = mlflow.get_experiment(experiment_id)
            return experiment
        except Exception as e:
            logger.error(f"Error setting up experiment: {e}")
            raise
    
    def start_run(self, 
                  run_name: Optional[str] = None,
                  tags: Optional[Dict[str, str]] = None) -> str:
        """
        Start a new MLflow run
        
        Args:
            run_name: Optional name for the run
            tags: Optional tags for the run
            
        Returns:
            Run ID
        """
        if run_name is None:
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        mlflow.start_run(run_name=run_name)
        
        # Set tags
        if tags:
            for key, value in tags.items():
                mlflow.set_tag(key, value)
        
        # Set default tags
        mlflow.set_tag("timestamp", datetime.now().isoformat())
        mlflow.set_tag("user", "demand-forecasting-pipeline")
        
        run = mlflow.active_run()
        logger.info(f"Started MLflow run: {run.info.run_id}")
        
        return run.info.run_id
    
    def log_params(self, params: Dict[str, Any]):
        """Log parameters to current run"""
        for key, value in params.items():
            # MLflow params must be strings or numbers
            if isinstance(value, (list, dict)):
                mlflow.log_param(key, json.dumps(value))
            else:
                mlflow.log_param(key, value)
    
    def log_metrics(self, 
                    metrics: Dict[str, float], 
                    step: Optional[int] = None):
        """Log metrics to current run"""
        for key, value in metrics.items():
            if value is not None and not np.isnan(value):
                mlflow.log_metric(key, value, step=step)
    
    def log_model(self, 
                  model: Any,
                  artifact_path: str,
                  model_type: str = "sklearn",
                  registered_model_name: Optional[str] = None,
                  **kwargs):
        """
        Log model to MLflow
        
        Args:
            model: Model object to log
            artifact_path: Path within artifacts to save model
            model_type: Type of model (sklearn, lightgbm, statsmodels)
            registered_model_name: Optional name for model registry
            **kwargs: Additional model logging parameters
        """
        try:
            if model_type == "lightgbm":
                mlflow.lightgbm.log_model(
                    model,
                    artifact_path,
                    registered_model_name=registered_model_name,
                    **kwargs
                )
            elif model_type == "statsmodels":
                mlflow.statsmodels.log_model(
                    model,
                    artifact_path,
                    registered_model_name=registered_model_name,
                    **kwargs
                )
            else:  # Default to sklearn
                mlflow.sklearn.log_model(
                    model,
                    artifact_path,
                    registered_model_name=registered_model_name,
                    **kwargs
                )
            
            logger.info(f"Model logged: {artifact_path}")
            
        except Exception as e:
            logger.error(f"Error logging model: {e}")
            raise
    
    def log_artifact(self, local_path: str):
        """Log a local file or directory as artifact"""
        mlflow.log_artifact(local_path)
    
    def log_artifacts(self, local_dir: str, artifact_path: Optional[str] = None):
        """Log all files in a directory as artifacts"""
        mlflow.log_artifacts(local_dir, artifact_path=artifact_path)
    
    def log_figure(self, figure, artifact_file: str):
        """Log matplotlib figure"""
        mlflow.log_figure(figure, artifact_file)
    
    def log_dict(self, dictionary: Dict, artifact_file: str):
        """Log dictionary as JSON artifact"""
        mlflow.log_dict(dictionary, artifact_file)
    
    def log_text(self, text: str, artifact_file: str):
        """Log text as artifact"""
        mlflow.log_text(text, artifact_file)
    
    def end_run(self, status: str = "FINISHED"):
        """
        End current MLflow run
        
        Args:
            status: Run status (FINISHED, FAILED, KILLED)
        """
        mlflow.end_run(status=status)
        logger.info(f"MLflow run ended with status: {status}")
    
    def log_dataset_info(self, 
                        df: pd.DataFrame,
                        dataset_name: str = "training_data"):
        """Log dataset statistics and sample"""
        dataset_info = {
            "name": dataset_name,
            "shape": list(df.shape),
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "null_counts": df.isnull().sum().to_dict(),
            "statistics": df.describe().to_dict()
        }
        
        # Log as artifact
        self.log_dict(dataset_info, f"{dataset_name}_info.json")
        
        # Log key metrics
        mlflow.log_param(f"{dataset_name}_rows", df.shape[0])
        mlflow.log_param(f"{dataset_name}_cols", df.shape[1])
        
        # Create data hash for versioning
        data_hash = hashlib.md5(pd.util.hash_pandas_object(df).values).hexdigest()
        mlflow.log_param(f"{dataset_name}_hash", data_hash[:8])
    
    def log_feature_importance(self, 
                              importance_dict: Dict[str, float],
                              top_n: int = 20):
        """Log feature importance"""
        # Sort by importance
        sorted_features = sorted(
            importance_dict.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:top_n]
        
        # Log as metrics
        for i, (feature, importance) in enumerate(sorted_features):
            mlflow.log_metric(f"feature_importance_{i:02d}_{feature}", importance)
        
        # Log full dict as artifact
        self.log_dict(importance_dict, "feature_importance.json")
    
    def log_optimization_results(self, 
                                 results: Dict[str, Any],
                                 baseline_results: Optional[Dict[str, Any]] = None):
        """Log price optimization results"""
        # Log optimized metrics
        if "revenue" in results:
            mlflow.log_metric("optimized_revenue", results["revenue"])
        if "units" in results:
            mlflow.log_metric("optimized_units", results["units"])
        if "margin" in results:
            mlflow.log_metric("optimized_margin", results["margin"])
        
        # Log lift vs baseline
        if baseline_results:
            for key in ["revenue", "units", "margin"]:
                if key in results and key in baseline_results:
                    lift = (results[key] / baseline_results[key] - 1) * 100
                    mlflow.log_metric(f"{key}_lift_pct", lift)
        
        # Log full results as artifact
        self.log_dict(results, "optimization_results.json")
    
    def search_runs(self, 
                   filter_string: str = "",
                   order_by: List[str] = None,
                   max_results: int = 100) -> pd.DataFrame:
        """
        Search for runs in the experiment
        
        Args:
            filter_string: Filter query (e.g., "metrics.rmse < 10")
            order_by: List of columns to order by
            max_results: Maximum number of results
            
        Returns:
            DataFrame of runs
        """
        runs = mlflow.search_runs(
            experiment_ids=[self.experiment_id],
            filter_string=filter_string,
            order_by=order_by,
            max_results=max_results
        )
        return runs
    
    def get_best_model(self, 
                      metric: str = "rmse",
                      ascending: bool = True) -> Dict[str, Any]:
        """
        Get best model based on metric
        
        Args:
            metric: Metric to optimize
            ascending: Whether lower is better
            
        Returns:
            Dictionary with run_id and model info
        """
        order_by = [f"metrics.{metric} {'ASC' if ascending else 'DESC'}"]
        runs = self.search_runs(order_by=order_by, max_results=1)
        
        if runs.empty:
            return None
        
        best_run = runs.iloc[0]
        return {
            "run_id": best_run["run_id"],
            "metric_value": best_run[f"metrics.{metric}"],
            "params": {col.replace("params.", ""): val 
                      for col, val in best_run.items() 
                      if col.startswith("params.")},
            "metrics": {col.replace("metrics.", ""): val 
                       for col, val in best_run.items() 
                       if col.startswith("metrics.")}
        }
    
    def load_model(self, run_id: str, artifact_path: str = "model"):
        """Load model from a specific run"""
        model_uri = f"runs:/{run_id}/{artifact_path}"
        return mlflow.pyfunc.load_model(model_uri)
    
    def compare_models(self, run_ids: List[str]) -> pd.DataFrame:
        """Compare multiple model runs"""
        runs_data = []
        
        for run_id in run_ids:
            run = self.client.get_run(run_id)
            run_data = {
                "run_id": run_id,
                "run_name": run.data.tags.get("mlflow.runName", ""),
                **run.data.params,
                **run.data.metrics
            }
            runs_data.append(run_data)
        
        return pd.DataFrame(runs_data)
    
    def register_model(self, 
                      run_id: str,
                      artifact_path: str = "model",
                      name: str = "demand-forecasting-model"):
        """Register model in model registry"""
        model_uri = f"runs:/{run_id}/{artifact_path}"
        
        model_details = mlflow.register_model(
            model_uri=model_uri,
            name=name
        )
        
        logger.info(f"Model registered: {name} v{model_details.version}")
        return model_details
    
    def transition_model_stage(self,
                              name: str,
                              version: int,
                              stage: str = "Staging"):
        """
        Transition model to different stage
        
        Args:
            name: Registered model name
            version: Model version
            stage: Target stage (Staging, Production, Archived)
        """
        self.client.transition_model_version_stage(
            name=name,
            version=version,
            stage=stage
        )
        logger.info(f"Model {name} v{version} transitioned to {stage}")


class ExperimentRunner:
    """
    High-level experiment runner with MLflow integration
    """
    
    def __init__(self, tracker: MLflowTracker):
        self.tracker = tracker
    
    def run_training_experiment(self,
                               model_type: str,
                               train_data: pd.DataFrame,
                               test_data: pd.DataFrame,
                               params: Dict[str, Any],
                               model_builder_func,
                               evaluator_func) -> Dict[str, Any]:
        """
        Run a complete training experiment with tracking
        
        Args:
            model_type: Type of model (ensemble, sarimax, etc.)
            train_data: Training DataFrame
            test_data: Test DataFrame  
            params: Model parameters
            model_builder_func: Function to build and train model
            evaluator_func: Function to evaluate model
            
        Returns:
            Experiment results
        """
        # Start MLflow run
        run_name = f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        tags = {
            "model_type": model_type,
            "data_split": "train_test",
            "environment": "development"
        }
        
        run_id = self.tracker.start_run(run_name=run_name, tags=tags)
        
        try:
            # Log parameters
            self.tracker.log_params(params)
            
            # Log dataset info
            self.tracker.log_dataset_info(train_data, "training_data")
            self.tracker.log_dataset_info(test_data, "test_data")
            
            # Train model
            logger.info(f"Training {model_type} model...")
            model = model_builder_func(train_data, params)
            
            # Evaluate model
            logger.info("Evaluating model...")
            metrics = evaluator_func(model, test_data)
            
            # Log metrics
            self.tracker.log_metrics(metrics)
            
            # Log model
            if model_type == "ensemble":
                # For ensemble, log component models
                # Log the LightGBM model
                if hasattr(model, 'model'):
                    self.tracker.log_model(
                        model,
                        "lightgbm_model", 
                        model_type="lightgbm"
                    )
            elif model_type == "sarimax":
                self.tracker.log_model(
                    model,
                    "sarimax_model",
                    model_type="statsmodels"
                )
            else:
                self.tracker.log_model(
                    model,
                    "model",
                    model_type="sklearn"
                )
            
            # Log feature importance if available
            if hasattr(model, 'feature_importance_'):
                self.tracker.log_feature_importance(model.feature_importance_)
            
            # End run successfully
            self.tracker.end_run(status="FINISHED")
            
            return {
                "run_id": run_id,
                "model": model,
                "metrics": metrics,
                "params": params
            }
            
        except Exception as e:
            logger.error(f"Experiment failed: {e}")
            self.tracker.end_run(status="FAILED")
            raise
    
    def run_optimization_experiment(self,
                                  optimizer,
                                  objective: str,
                                  constraints: Dict[str, Any],
                                  baseline_prices: np.ndarray) -> Dict[str, Any]:
        """
        Run price optimization experiment with tracking
        
        Args:
            optimizer: Price optimizer object
            objective: Optimization objective
            constraints: Optimization constraints
            baseline_prices: Baseline prices for comparison
            
        Returns:
            Optimization results
        """
        run_name = f"optimization_{objective}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        tags = {
            "experiment_type": "optimization",
            "objective": objective
        }
        
        run_id = self.tracker.start_run(run_name=run_name, tags=tags)
        
        try:
            # Log optimization parameters
            params = {
                "objective": objective,
                **constraints
            }
            self.tracker.log_params(params)
            
            # Run optimization
            logger.info(f"Running {objective} optimization...")
            optimized_prices = optimizer.optimize(objective=objective, **constraints)
            
            # Calculate results
            optimized_results = optimizer.evaluate_prices(optimized_prices)
            baseline_results = optimizer.evaluate_prices(baseline_prices)
            
            # Log results
            self.tracker.log_optimization_results(
                optimized_results,
                baseline_results
            )
            
            # Log price plans
            price_plan_df = pd.DataFrame({
                "baseline_price": baseline_prices,
                "optimized_price": optimized_prices,
                "price_change_pct": (optimized_prices / baseline_prices - 1) * 100
            })
            price_plan_df.to_csv("price_plan.csv", index=False)
            self.tracker.log_artifact("price_plan.csv")
            
            # End run
            self.tracker.end_run(status="FINISHED")
            
            return {
                "run_id": run_id,
                "optimized_prices": optimized_prices,
                "optimized_results": optimized_results,
                "baseline_results": baseline_results,
                "lift": {
                    key: (optimized_results[key] / baseline_results[key] - 1) * 100
                    for key in ["revenue", "units", "margin"]
                    if key in optimized_results and key in baseline_results
                }
            }
            
        except Exception as e:
            logger.error(f"Optimization experiment failed: {e}")
            self.tracker.end_run(status="FAILED")
            raise


# Utility functions for MLflow integration

def setup_mlflow_config(config_path: str = "config/mlflow.yaml") -> Dict[str, Any]:
    """Load MLflow configuration from YAML file"""
    config_file = Path(config_path)
    
    if config_file.exists():
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
    else:
        # Default configuration
        config = {
            "tracking_uri": "mlruns",
            "experiment_name": "demand-forecasting",
            "registry_uri": None,
            "artifact_location": None
        }
        
        # Create config directory and save default config
        config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(config_file, 'w') as f:
            yaml.dump(config, f)
    
    return config


def create_experiment_name(base_name: str = "demand-forecasting",
                          suffix: Optional[str] = None) -> str:
    """Create experiment name with optional suffix"""
    if suffix:
        return f"{base_name}-{suffix}"
    return base_name


def log_code_version(repo_path: str = "."):
    """Log git commit hash and diff as code version"""
    try:
        import subprocess
        
        # Get current commit hash
        commit_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_path
        ).decode().strip()
        
        mlflow.set_tag("git_commit", commit_hash)
        
        # Check for uncommitted changes
        diff = subprocess.check_output(
            ["git", "diff", "--stat"],
            cwd=repo_path
        ).decode().strip()
        
        if diff:
            mlflow.set_tag("has_uncommitted_changes", "true")
            mlflow.log_text(diff, "git_diff.txt")
        else:
            mlflow.set_tag("has_uncommitted_changes", "false")
            
    except Exception as e:
        logger.warning(f"Could not log git version: {e}")


if __name__ == "__main__":
    # Example usage
    print("MLflow Tracking Module")
    print("=" * 50)
    
    # Initialize tracker
    tracker = MLflowTracker(
        tracking_uri="mlruns",
        experiment_name="demand-forecasting-demo"
    )
    
    # Start a demo run
    tracker.start_run(run_name="demo_run")
    
    # Log some parameters
    tracker.log_params({
        "model_type": "ensemble",
        "n_estimators": 100,
        "learning_rate": 0.1
    })
    
    # Log some metrics
    tracker.log_metrics({
        "rmse": 8.45,
        "mape": 0.183,
        "r2": 0.71
    })
    
    # End run
    tracker.end_run()
    
    print("\nDemo run completed!")
    print(f"Check MLflow UI: mlflow ui --backend-store-uri {tracker.tracking_uri}")