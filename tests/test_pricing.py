"""
Unit tests for pricing optimization module.

Tests monotonic demand with respect to price, constraint validation, and KPI calculations.
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pricing.optimizer import PricingOptimizer, PricingConstraints


class TestPricingOptimizer:
    """Test suite for pricing optimization."""
    
    @pytest.fixture
    def optimizer(self):
        """Create PricingOptimizer instance."""
        return PricingOptimizer(demand_model=None, elasticity=-1.5)
    
    @pytest.fixture
    def constraints(self):
        """Create default constraints."""
        return PricingConstraints(
            min_price_ratio=0.7,
            max_price_ratio=1.2,
            max_promo_days_per_month=10,
            max_discount_depth=0.5,
            min_margin_ratio=0.1
        )
    
    def test_monotonic_demand_price_relationship(self, optimizer):
        """Test that demand decreases monotonically with price increase."""
        base_price = 10.0
        base_demand = 100.0
        
        # Test prices from 70% to 130% of base price
        prices = np.linspace(base_price * 0.7, base_price * 1.3, 20)
        demands = []
        
        for price in prices:
            demand = optimizer.simulate_demand(price, base_price, base_demand)
            demands.append(demand)
        
        # Check monotonic decrease (allowing small tolerance for numerical issues)
        for i in range(1, len(demands)):
            assert demands[i] <= demands[i-1] + 0.01, \
                f"Demand should decrease with price: {demands[i]} > {demands[i-1]}"
        
        # Check elasticity is applied correctly
        # With elasticity = -1.5, 10% price increase should decrease demand by ~15%
        price_110 = base_price * 1.1
        demand_110 = optimizer.simulate_demand(price_110, base_price, base_demand)
        expected_demand = base_demand * (1.1 ** -1.5)
        
        assert abs(demand_110 - expected_demand) < 1.0, \
            f"Elasticity not applied correctly: got {demand_110}, expected {expected_demand}"
    
    def test_promotion_effect(self, optimizer):
        """Test that promotions increase demand."""
        base_price = 10.0
        base_demand = 100.0
        
        # Without promotion
        features_no_promo = {'promo_flag': 0}
        demand_no_promo = optimizer.simulate_demand(
            base_price, base_price, base_demand, features_no_promo
        )
        
        # With promotion
        features_promo = {'promo_flag': 1}
        demand_promo = optimizer.simulate_demand(
            base_price, base_price, base_demand, features_promo
        )
        
        # Promotion should increase demand (30% boost in implementation)
        assert demand_promo > demand_no_promo, "Promotion should increase demand"
        assert demand_promo / demand_no_promo > 1.2, "Promotion should have significant effect"
    
    def test_holiday_effect(self, optimizer):
        """Test that holidays increase demand."""
        base_price = 10.0
        base_demand = 100.0
        
        # Without holiday
        features_no_holiday = {'holiday_flag': 0}
        demand_no_holiday = optimizer.simulate_demand(
            base_price, base_price, base_demand, features_no_holiday
        )
        
        # With holiday
        features_holiday = {'holiday_flag': 1}
        demand_holiday = optimizer.simulate_demand(
            base_price, base_price, base_demand, features_holiday
        )
        
        # Holiday should increase demand (20% boost in implementation)
        assert demand_holiday > demand_no_holiday, "Holiday should increase demand"
        assert demand_holiday / demand_no_holiday > 1.15, "Holiday should have significant effect"
    
    def test_weather_effect(self, optimizer):
        """Test weather impact on demand."""
        base_price = 10.0
        base_demand = 100.0
        
        # Bad weather
        features_bad = {'weather_index': 0.0}
        demand_bad = optimizer.simulate_demand(
            base_price, base_price, base_demand, features_bad
        )
        
        # Good weather
        features_good = {'weather_index': 1.0}
        demand_good = optimizer.simulate_demand(
            base_price, base_price, base_demand, features_good
        )
        
        # Good weather should increase demand
        assert demand_good > demand_bad, "Good weather should increase demand"
    
    def test_kpi_calculations(self, optimizer):
        """Test KPI calculation accuracy."""
        price = 15.0
        demand = 100.0
        cost = 9.0
        
        kpis = optimizer.calculate_kpis(price, demand, cost)
        
        # Check all KPIs are present
        assert 'price' in kpis
        assert 'demand' in kpis
        assert 'revenue' in kpis
        assert 'margin' in kpis
        assert 'margin_ratio' in kpis
        assert 'units_sold' in kpis
        
        # Check calculations
        assert kpis['price'] == price
        assert kpis['demand'] == demand
        assert kpis['revenue'] == price * demand
        assert kpis['margin'] == (price - cost) * demand
        assert abs(kpis['margin_ratio'] - (price - cost) / price) < 0.001
        assert kpis['units_sold'] == demand
    
    def test_revenue_optimization(self, optimizer):
        """Test revenue optimization."""
        base_price = 15.0
        base_demand = 50.0
        cost = 9.0
        
        result = optimizer.optimize_revenue(base_price, base_demand, cost)
        
        # Check result structure
        assert 'optimal_price' in result
        assert 'optimal_kpis' in result
        assert 'baseline_kpis' in result
        assert 'improvements' in result
        assert 'optimization_success' in result
        
        # Optimal price should be within constraints
        assert result['optimal_price'] >= base_price * 0.7
        assert result['optimal_price'] <= base_price * 1.2
        
        # Revenue should be improved (or at least not worse)
        assert result['optimal_kpis']['revenue'] >= result['baseline_kpis']['revenue'] - 1.0
    
    def test_units_target_optimization(self, optimizer):
        """Test units target optimization."""
        base_price = 15.0
        base_demand = 50.0
        cost = 9.0
        units_target = 60.0
        
        result = optimizer.optimize_units_target(
            base_price, base_demand, cost, units_target
        )
        
        # Check result structure
        assert 'optimal_price' in result
        assert 'optimal_kpis' in result
        assert 'target_achievement' in result
        
        # Should try to get close to target
        units_achieved = result['optimal_kpis']['units_sold']
        assert units_achieved > base_demand  # Should increase from baseline
        
        # Price should be lower to increase units (given negative elasticity)
        assert result['optimal_price'] <= base_price
    
    def test_multi_objective_optimization(self, optimizer):
        """Test multi-objective optimization."""
        base_price = 15.0
        base_demand = 50.0
        cost = 9.0
        weights = {'revenue': 0.5, 'units': 0.3, 'margin': 0.2}
        
        result = optimizer.multi_objective_optimization(
            base_price, base_demand, cost, weights
        )
        
        # Check result structure
        assert 'optimal_price' in result
        assert 'optimal_kpis' in result
        assert 'weights_used' in result
        
        # Weights should be normalized
        total_weight = sum(result['weights_used'].values())
        assert abs(total_weight - 1.0) < 0.001
        
        # Optimal price should be within constraints
        assert result['optimal_price'] >= base_price * 0.7
        assert result['optimal_price'] <= base_price * 1.2
    
    def test_constraint_validation(self, optimizer):
        """Test constraint validation."""
        optimizer.constraints = PricingConstraints(
            min_price_ratio=0.8,
            max_price_ratio=1.1,
            max_promo_days_per_month=5
        )
        
        # Create price plan that violates constraints
        price_plan = pd.DataFrame({
            'sku_id': ['SKU001', 'SKU002', 'SKU003'],
            'base_price': [10.0, 20.0, 30.0],
            'optimal_price': [7.0, 22.5, 27.0],  # First two violate bounds
            'promo_flag': [1, 1, 0]
        })
        
        is_valid, violations = optimizer.validate_constraints(price_plan)
        
        assert not is_valid, "Should detect constraint violations"
        assert len(violations) >= 2, "Should find at least 2 violations"
        
        # Check specific violations are detected
        violation_texts = ' '.join(violations)
        assert 'SKU001' in violation_texts  # Price too low (7/10 = 0.7 < 0.8)
        assert 'SKU002' in violation_texts  # Price too high (22.5/20 = 1.125 > 1.1)
    
    def test_price_plan_optimization(self, optimizer):
        """Test full price plan optimization."""
        # Create sample data
        data = pd.DataFrame({
            'store_id': ['S01', 'S01', 'S02'],
            'sku_id': ['SKU001', 'SKU002', 'SKU001'],
            'base_price': [10.0, 15.0, 10.0],
            'units_sold': [50, 30, 40],
            'final_price': [10.0, 15.0, 10.0]
        })
        
        price_plan = optimizer.optimize_price_plan(data, horizon=7, objective='revenue')
        
        # Check result structure
        assert isinstance(price_plan, pd.DataFrame)
        assert 'store_id' in price_plan.columns
        assert 'sku_id' in price_plan.columns
        assert 'optimal_price' in price_plan.columns
        assert 'expected_revenue' in price_plan.columns
        
        # Should have entries for each store-SKU combo for each day
        expected_rows = 3 * 7  # 3 combinations × 7 days
        assert len(price_plan) == expected_rows
        
        # All optimal prices should be within constraints
        for _, row in price_plan.iterrows():
            ratio = row['optimal_price'] / row['base_price']
            assert ratio >= optimizer.constraints.min_price_ratio
            assert ratio <= optimizer.constraints.max_price_ratio
    
    def test_inventory_constraints(self):
        """Test inventory constraint application."""
        inventory = {'SKU001': 50}
        constraints = PricingConstraints(inventory_constraints=inventory)
        optimizer = PricingOptimizer(demand_model=None, elasticity=-1.5, constraints=constraints)
        
        base_price = 10.0
        base_demand = 100.0  # Higher than inventory
        features = {'sku_id': 'SKU001'}
        
        # Demand should be capped at inventory
        demand = optimizer.simulate_demand(base_price, base_price, base_demand, features)
        assert demand <= 50, "Demand should be capped by inventory"
    
    def test_negative_demand_prevention(self, optimizer):
        """Test that demand is never negative."""
        base_price = 10.0
        base_demand = 10.0
        
        # Very high price that would create negative demand with simple linear model
        high_price = base_price * 10
        demand = optimizer.simulate_demand(high_price, base_price, base_demand)
        
        assert demand >= 0, "Demand should never be negative"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])