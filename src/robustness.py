"""Robustness and stress testing for demand forecasting models."""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RobustnessAnalyzer:
    """Analyze model robustness and perform stress tests."""
    
    def __init__(self, model, feature_engineer):
        """Initialize robustness analyzer.
        
        Args:
            model: Trained forecasting model
            feature_engineer: Feature engineering pipeline
        """
        self.model = model
        self.feature_engineer = feature_engineer
        
    def stress_test_competitor_undercut(self, 
                                       test_data: pd.DataFrame, 
                                       undercut_pct: float = 5) -> Dict[str, float]:
        """Test model response to competitor price cuts.
        
        Args:
            test_data: Test dataset
            undercut_pct: Percentage price undercut by competitor
            
        Returns:
            Dictionary with impact metrics
        """
        # Create stressed scenario
        stressed_data = test_data.copy()
        
        # Reduce competitor prices
        stressed_data['competitor_price'] *= (1 - undercut_pct/100)
        
        # Recalculate price-related features
        stressed_data['price_gap'] = stressed_data['final_price'] - stressed_data['competitor_price']
        stressed_data['price_vs_comp'] = stressed_data['final_price'] / stressed_data['competitor_price']
        stressed_data['undercut_flag'] = (stressed_data['final_price'] > stressed_data['competitor_price']).astype(int)
        
        # Create features for both scenarios
        baseline_features = self.feature_engineer.create_features(test_data)
        stressed_features = self.feature_engineer.create_features(stressed_data)
        
        # Prepare for modeling
        X_baseline, _ = self.feature_engineer.prepare_for_modeling(baseline_features)
        X_stressed, _ = self.feature_engineer.prepare_for_modeling(stressed_features)
        
        # Make predictions
        baseline_pred = self.model.predict(X_baseline)
        stressed_pred = self.model.predict(X_stressed)
        
        # Calculate impacts
        demand_drop_pct = ((baseline_pred - stressed_pred) / baseline_pred * 100)
        
        # Calculate revenue impact
        baseline_revenue = (baseline_pred * test_data['final_price'].values[:len(baseline_pred)]).sum()
        stressed_revenue = (stressed_pred * test_data['final_price'].values[:len(stressed_pred)]).sum()
        revenue_drop_pct = (baseline_revenue - stressed_revenue) / baseline_revenue * 100
        
        return {
            'undercut_pct': undercut_pct,
            'avg_demand_drop': np.mean(demand_drop_pct),
            'max_demand_drop': np.max(demand_drop_pct),
            'min_demand_drop': np.min(demand_drop_pct),
            'affected_products_pct': (demand_drop_pct > 5).sum() / len(demand_drop_pct) * 100,
            'revenue_drop_pct': revenue_drop_pct
        }
    
    def stress_test_promotion_saturation(self, test_data: pd.DataFrame) -> Dict[str, float]:
        """Test impact of all products on promotion.
        
        Args:
            test_data: Test dataset
            
        Returns:
            Dictionary with impact metrics
        """
        # Create scenario with all items on promotion
        promo_data = test_data.copy()
        promo_data['promo_flag'] = 1
        promo_data['promo_depth'] = 0.2  # 20% discount
        promo_data['final_price'] = promo_data['base_price'] * 0.8
        
        # Create features
        baseline_features = self.feature_engineer.create_features(test_data)
        promo_features = self.feature_engineer.create_features(promo_data)
        
        # Prepare for modeling
        X_baseline, _ = self.feature_engineer.prepare_for_modeling(baseline_features)
        X_promo, _ = self.feature_engineer.prepare_for_modeling(promo_features)
        
        # Predictions
        baseline_pred = self.model.predict(X_baseline)
        promo_pred = self.model.predict(X_promo)
        
        # Calculate impacts
        demand_lift_pct = ((promo_pred - baseline_pred) / baseline_pred * 100)
        
        # Revenue calculation (with lower prices)
        baseline_revenue = (baseline_pred * test_data['final_price'].values[:len(baseline_pred)]).sum()
        promo_revenue = (promo_pred * promo_data['final_price'].values[:len(promo_pred)]).sum()
        revenue_change_pct = (promo_revenue - baseline_revenue) / baseline_revenue * 100
        
        return {
            'avg_demand_lift': np.mean(demand_lift_pct),
            'max_demand_lift': np.max(demand_lift_pct),
            'revenue_change_pct': revenue_change_pct,
            'cannibalization_risk': 'High' if np.mean(demand_lift_pct) < 20 else 'Low'
        }
    
    def stress_test_no_promotions(self, test_data: pd.DataFrame) -> Dict[str, float]:
        """Test impact of removing all promotions.
        
        Args:
            test_data: Test dataset
            
        Returns:
            Dictionary with impact metrics
        """
        # Create scenario without promotions
        no_promo_data = test_data.copy()
        no_promo_data['promo_flag'] = 0
        no_promo_data['promo_depth'] = 0
        no_promo_data['final_price'] = no_promo_data['base_price']
        
        # Create features
        baseline_features = self.feature_engineer.create_features(test_data)
        no_promo_features = self.feature_engineer.create_features(no_promo_data)
        
        # Prepare for modeling
        X_baseline, _ = self.feature_engineer.prepare_for_modeling(baseline_features)
        X_no_promo, _ = self.feature_engineer.prepare_for_modeling(no_promo_features)
        
        # Predictions
        baseline_pred = self.model.predict(X_baseline)
        no_promo_pred = self.model.predict(X_no_promo)
        
        # Calculate impacts
        demand_drop_pct = ((baseline_pred - no_promo_pred) / baseline_pred * 100)
        
        return {
            'avg_demand_drop': np.mean(demand_drop_pct),
            'promo_dependency': np.mean(demand_drop_pct),
            'affected_products_pct': (demand_drop_pct > 10).sum() / len(demand_drop_pct) * 100
        }
    
    def stress_test_price_boundaries(self, test_data: pd.DataFrame) -> Dict[str, Any]:
        """Test model behavior at price extremes.
        
        Args:
            test_data: Test dataset
            
        Returns:
            Dictionary with boundary test results
        """
        results = {}
        
        # Test minimum prices (30% discount)
        min_price_data = test_data.copy()
        min_price_data['final_price'] = min_price_data['base_price'] * 0.7
        
        # Test maximum prices (20% markup)
        max_price_data = test_data.copy()
        max_price_data['final_price'] = max_price_data['base_price'] * 1.2
        
        # Create features
        baseline_features = self.feature_engineer.create_features(test_data)
        min_price_features = self.feature_engineer.create_features(min_price_data)
        max_price_features = self.feature_engineer.create_features(max_price_data)
        
        # Prepare for modeling
        X_baseline, _ = self.feature_engineer.prepare_for_modeling(baseline_features)
        X_min, _ = self.feature_engineer.prepare_for_modeling(min_price_features)
        X_max, _ = self.feature_engineer.prepare_for_modeling(max_price_features)
        
        # Predictions
        baseline_pred = self.model.predict(X_baseline)
        min_pred = self.model.predict(X_min)
        max_pred = self.model.predict(X_max)
        
        # Calculate elasticities
        min_elasticity = ((min_pred - baseline_pred) / baseline_pred) / (-0.3)
        max_elasticity = ((max_pred - baseline_pred) / baseline_pred) / 0.2
        
        results['min_price_elasticity'] = np.mean(min_elasticity)
        results['max_price_elasticity'] = np.mean(max_elasticity)
        results['elasticity_stable'] = abs(results['min_price_elasticity'] - results['max_price_elasticity']) < 0.5
        
        return results
    
    def validate_causality(self, test_data: pd.DataFrame) -> Tuple[bool, Dict[str, bool]]:
        """Validate that model respects causal relationships.
        
        Args:
            test_data: Test dataset
            
        Returns:
            Tuple of (all_valid, individual_test_results)
        """
        tests = {}
        
        # Test 1: Higher prices should reduce demand
        price_up_data = test_data.copy()
        price_up_data['final_price'] *= 1.1
        
        baseline_features = self.feature_engineer.create_features(test_data)
        price_up_features = self.feature_engineer.create_features(price_up_data)
        
        X_baseline, _ = self.feature_engineer.prepare_for_modeling(baseline_features)
        X_price_up, _ = self.feature_engineer.prepare_for_modeling(price_up_features)
        
        baseline_pred = self.model.predict(X_baseline)
        price_up_pred = self.model.predict(X_price_up)
        
        tests['price_up_demand_down'] = np.mean(price_up_pred) < np.mean(baseline_pred)
        
        # Test 2: Promotions should increase demand
        promo_data = test_data.copy()
        promo_data['promo_flag'] = 1
        promo_features = self.feature_engineer.create_features(promo_data)
        X_promo, _ = self.feature_engineer.prepare_for_modeling(promo_features)
        promo_pred = self.model.predict(X_promo)
        
        tests['promo_increases_demand'] = np.mean(promo_pred) > np.mean(baseline_pred)
        
        # Test 3: Holidays should affect demand
        holiday_data = test_data.copy()
        holiday_data['holiday_flag'] = 1
        holiday_features = self.feature_engineer.create_features(holiday_data)
        X_holiday, _ = self.feature_engineer.prepare_for_modeling(holiday_features)
        holiday_pred = self.model.predict(X_holiday)
        
        tests['holiday_affects_demand'] = abs(np.mean(holiday_pred) - np.mean(baseline_pred)) > 0.01
        
        return all(tests.values()), tests
    
    def run_all_stress_tests(self, test_data: pd.DataFrame) -> Dict[str, Any]:
        """Run comprehensive stress testing suite.
        
        Args:
            test_data: Test dataset
            
        Returns:
            Dictionary with all test results
        """
        logger.info("Running comprehensive stress tests...")
        
        results = {
            'competitor_undercut_5pct': self.stress_test_competitor_undercut(test_data, 5),
            'competitor_undercut_10pct': self.stress_test_competitor_undercut(test_data, 10),
            'promotion_saturation': self.stress_test_promotion_saturation(test_data),
            'no_promotions': self.stress_test_no_promotions(test_data),
            'price_boundaries': self.stress_test_price_boundaries(test_data)
        }
        
        # Add causality validation
        causality_valid, causality_tests = self.validate_causality(test_data)
        results['causality_validation'] = {
            'all_tests_passed': causality_valid,
            'individual_tests': causality_tests
        }
        
        # Summary statistics
        results['summary'] = {
            'model_robust': causality_valid and results['price_boundaries']['elasticity_stable'],
            'main_vulnerability': self._identify_vulnerability(results),
            'recommended_safeguards': self._recommend_safeguards(results)
        }
        
        return results
    
    def _identify_vulnerability(self, results: Dict) -> str:
        """Identify main model vulnerability from test results."""
        vulnerabilities = []
        
        if results['competitor_undercut_10pct']['avg_demand_drop'] > 20:
            vulnerabilities.append("High sensitivity to competitor pricing")
        
        if not results['price_boundaries']['elasticity_stable']:
            vulnerabilities.append("Unstable elasticity at price extremes")
        
        if results['promotion_saturation']['cannibalization_risk'] == 'High':
            vulnerabilities.append("Promotion effectiveness diminishes with saturation")
        
        return vulnerabilities[0] if vulnerabilities else "No major vulnerabilities detected"
    
    def _recommend_safeguards(self, results: Dict) -> list:
        """Recommend safeguards based on test results."""
        safeguards = []
        
        if results['competitor_undercut_10pct']['avg_demand_drop'] > 15:
            safeguards.append("Implement competitive price monitoring and automatic response")
        
        if not results['price_boundaries']['elasticity_stable']:
            safeguards.append("Restrict pricing to ±20% of base price")
        
        if results['no_promotions']['promo_dependency'] > 20:
            safeguards.append("Diversify demand drivers beyond promotions")
        
        return safeguards if safeguards else ["Standard monitoring and validation procedures"]


def generate_stress_test_report(model, feature_engineer, test_data: pd.DataFrame) -> str:
    """Generate comprehensive stress test report.
    
    Args:
        model: Trained model
        feature_engineer: Feature engineering pipeline
        test_data: Test dataset
        
    Returns:
        Formatted report string
    """
    analyzer = RobustnessAnalyzer(model, feature_engineer)
    results = analyzer.run_all_stress_tests(test_data)
    
    report = """
=== ROBUSTNESS & STRESS TEST REPORT ===

1. COMPETITOR RESPONSE TESTS
-----------------------------
5% Undercut Impact:
  - Average Demand Drop: {:.1f}%
  - Revenue Impact: {:.1f}%
  - Affected Products: {:.1f}%

10% Undercut Impact:
  - Average Demand Drop: {:.1f}%
  - Revenue Impact: {:.1f}%
  - Affected Products: {:.1f}%

2. PROMOTION SCENARIOS
----------------------
All Products on Promotion:
  - Demand Lift: {:.1f}%
  - Revenue Change: {:.1f}%
  - Cannibalization Risk: {}

No Promotions:
  - Demand Drop: {:.1f}%
  - Promo Dependency: {:.1f}%

3. PRICE BOUNDARY TESTS
-----------------------
  - Min Price Elasticity: {:.2f}
  - Max Price Elasticity: {:.2f}
  - Elasticity Stable: {}

4. CAUSALITY VALIDATION
-----------------------
  - All Tests Passed: {}
  - Price → Demand: {}
  - Promo → Demand: {}
  - Holiday → Demand: {}

5. SUMMARY
----------
  - Model Robust: {}
  - Main Vulnerability: {}
  - Recommended Safeguards:
    {}
""".format(
        results['competitor_undercut_5pct']['avg_demand_drop'],
        results['competitor_undercut_5pct']['revenue_drop_pct'],
        results['competitor_undercut_5pct']['affected_products_pct'],
        results['competitor_undercut_10pct']['avg_demand_drop'],
        results['competitor_undercut_10pct']['revenue_drop_pct'],
        results['competitor_undercut_10pct']['affected_products_pct'],
        results['promotion_saturation']['avg_demand_lift'],
        results['promotion_saturation']['revenue_change_pct'],
        results['promotion_saturation']['cannibalization_risk'],
        results['no_promotions']['avg_demand_drop'],
        results['no_promotions']['promo_dependency'],
        results['price_boundaries']['min_price_elasticity'],
        results['price_boundaries']['max_price_elasticity'],
        results['price_boundaries']['elasticity_stable'],
        results['causality_validation']['all_tests_passed'],
        results['causality_validation']['individual_tests']['price_up_demand_down'],
        results['causality_validation']['individual_tests']['promo_increases_demand'],
        results['causality_validation']['individual_tests']['holiday_affects_demand'],
        results['summary']['model_robust'],
        results['summary']['main_vulnerability'],
        '\n    '.join(results['summary']['recommended_safeguards'])
    )
    
    return report