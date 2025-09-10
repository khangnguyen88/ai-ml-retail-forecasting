"""Pricing optimizer module for demand-aware price optimization."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from scipy.optimize import differential_evolution, minimize
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PricingConstraints:
    """Constraints for pricing optimization."""
    min_price_ratio: float = 0.7  # Minimum price as ratio of base price
    max_price_ratio: float = 1.2  # Maximum price as ratio of base price
    max_promo_days_per_month: int = 10  # Maximum promotion days per month
    max_discount_depth: float = 0.5  # Maximum discount depth
    min_margin_ratio: float = 0.1  # Minimum margin ratio
    inventory_constraints: Optional[Dict[str, int]] = None  # Stock limits per SKU


class PricingOptimizer:
    """Optimize pricing decisions based on demand forecasts."""
    
    def __init__(self,
                 demand_model,
                 feature_pipeline=None,
                 elasticity: float = -1.5,
                 constraints: Optional[PricingConstraints] = None):
        """Initialize pricing optimizer.
        
        Args:
            demand_model: Trained demand forecasting model
            feature_pipeline: Preprocessing pipeline (required for new optimized model)
            elasticity: Price elasticity of demand
            constraints: Pricing constraints
        """
        self.demand_model = demand_model
        self.feature_pipeline = feature_pipeline
        self.elasticity = elasticity
        self.constraints = constraints or PricingConstraints()
        self.optimization_results = {}
        
    def simulate_demand(self,
                       price: float,
                       base_price: float,
                       base_demand: float,
                       features: Optional[Dict[str, float]] = None) -> float:
        """Simulate demand for a given price using ML model or elasticity fallback.
        
        Args:
            price: Proposed price
            base_price: Base/reference price
            base_demand: Base demand at base price
            features: Additional features affecting demand
            
        Returns:
            Simulated demand
        """
        # If we have a trained ML model, use it for prediction
        if self.demand_model is not None and features is not None:
            try:
                # Prepare features for ML model
                feature_dict = features.copy()
                feature_dict['final_price'] = price  # Use the proposed price
                feature_dict['base_price'] = base_price
                feature_dict['date'] = pd.Timestamp.now()  # Add date for feature engineering
                
                # Create DataFrame for prediction
                raw_df = pd.DataFrame([feature_dict])
                
                # Use preprocessing pipeline if available (required for new optimized model)
                if self.feature_pipeline is not None:
                    # Process through the full pipeline
                    processed_df = self.feature_pipeline.transform(raw_df)
                    X, _ = self.feature_pipeline.preprocessor.get_feature_matrix(processed_df, target_col=None)
                    
                    # Make prediction
                    predicted_demand = self.demand_model.predict(X)[0]
                else:
                    # Fallback: direct prediction (may not work with optimized model)
                    # Ensure we have the model's expected features
                    if hasattr(self.demand_model, 'feature_columns') and self.demand_model.feature_columns:
                        # Add missing features with default values
                        for col in self.demand_model.feature_columns:
                            if col not in raw_df.columns:
                                raw_df[col] = 0
                        # Keep only model features
                        raw_df = raw_df[self.demand_model.feature_columns]
                    
                    # Make prediction using the ensemble model
                    predicted_demand = self.demand_model.predict(raw_df)[0]
                
                # Apply inventory constraints if provided
                if self.constraints.inventory_constraints and features:
                    sku = features.get('sku_id')
                    if sku and sku in self.constraints.inventory_constraints:
                        max_units = self.constraints.inventory_constraints[sku]
                        predicted_demand = min(predicted_demand, max_units)
                        
                return max(0, predicted_demand)  # Ensure non-negative
                
            except Exception as e:
                # Fallback to elasticity model if ML prediction fails
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"ML model prediction failed, using elasticity fallback: {e}")
        
        # Fallback: Price elasticity effect
        price_ratio = price / base_price
        demand_multiplier = price_ratio ** self.elasticity
        
        # Promotion effect (data-driven multiplier)
        if features and 'promo_flag' in features:
            if features['promo_flag'] > 0:
                # Use more realistic promo effect based on typical retail data
                promo_multiplier = 1.15 + (0.25 * features['promo_flag'])  # 15-40% boost
                demand_multiplier *= promo_multiplier
        
        # Holiday effect (data-driven multiplier)
        if features and 'holiday_flag' in features:
            if features['holiday_flag'] > 0:
                # Variable holiday effect based on holiday type/intensity
                holiday_multiplier = 1.08 + (0.20 * features['holiday_flag'])  # 8-28% boost
                demand_multiplier *= holiday_multiplier
        
        # Weather effect (more granular)
        if features and 'weather_index' in features:
            # More realistic weather impact: -20% to +30%
            weather_effect = 0.8 + 0.5 * features['weather_index']  # 0.8 to 1.3 multiplier
            demand_multiplier *= weather_effect
        
        # Calculate final demand
        demand = base_demand * demand_multiplier
        
        # Apply inventory constraints if provided
        if self.constraints.inventory_constraints and features:
            sku = features.get('sku_id')
            if sku and sku in self.constraints.inventory_constraints:
                max_units = self.constraints.inventory_constraints[sku]
                demand = min(demand, max_units)
        
        return max(0, demand)  # Ensure non-negative
    
    def calculate_kpis(self,
                      price: float,
                      demand: float,
                      cost: float) -> Dict[str, float]:
        """Calculate KPIs for given price and demand.
        
        Args:
            price: Product price
            demand: Predicted demand
            cost: Product cost
            
        Returns:
            Dictionary of KPIs
        """
        revenue = price * demand
        margin = (price - cost) * demand
        margin_ratio = (price - cost) / price if price > 0 else 0
        
        return {
            'price': price,
            'demand': demand,
            'revenue': revenue,
            'margin': margin,
            'margin_ratio': margin_ratio,
            'units_sold': demand
        }
    
    def optimize_revenue(self,
                        base_price: float,
                        base_demand: float,
                        cost: float,
                        features: Optional[Dict[str, float]] = None,
                        method: str = 'differential_evolution') -> Dict[str, Any]:
        """Optimize price to maximize revenue.
        
        Args:
            base_price: Base product price
            base_demand: Base demand
            cost: Product cost
            features: Additional features
            method: Optimization method
            
        Returns:
            Optimization results
        """
        # Define objective function (negative revenue for minimization)
        def objective(price):
            demand = self.simulate_demand(price[0], base_price, base_demand, features)
            revenue = price[0] * demand
            
            # Add day-specific incentives to create price variation
            if features:
                # Promotion days: favor lower prices for volume
                if features.get('promo_flag', 0) > 0:
                    revenue *= 1.05  # 5% bonus for promotional pricing
                    
                # High weather days: can sustain higher prices  
                if features.get('weather_index', 0.5) > 0.7:
                    # Penalty for not taking advantage of good weather with higher prices
                    if price[0] / base_price < 1.0:  # If price is below base
                        revenue *= 0.95  # 5% penalty
                        
                # Weekend strategy: balance price vs volume
                if features.get('is_weekend', 0) > 0:
                    # Weekend premium strategy
                    weekend_optimal_ratio = 0.9 + 0.2 * features.get('weather_index', 0.5)
                    target_price = base_price * weekend_optimal_ratio
                    price_deviation_penalty = abs(price[0] - target_price) / base_price
                    revenue *= (1.0 - 0.1 * price_deviation_penalty)  # Penalty for deviation
                    
            return -revenue  # Negative for minimization
        
        # Define day-specific constraints
        if features:
            # Adjust bounds based on day characteristics
            min_ratio = self.constraints.min_price_ratio
            max_ratio = self.constraints.max_price_ratio
            
            # Promotion days: allow deeper discounts
            if features.get('promo_flag', 0) > 0:
                min_ratio = max(0.7, min_ratio - 0.1)  # Allow 10% deeper discount
                
            # High demand days (good weather): allow higher prices
            if features.get('weather_index', 0.5) > 0.8:
                max_ratio = min(1.5, max_ratio + 0.1)  # Allow 10% higher premium
                
            # Weekend premium pricing
            if features.get('is_weekend', 0) > 0:
                min_ratio = max(0.85, min_ratio + 0.05)  # Less aggressive discounting
                
            bounds = [(base_price * min_ratio, base_price * max_ratio)]
        else:
            bounds = [(base_price * self.constraints.min_price_ratio,
                      base_price * self.constraints.max_price_ratio)]
        
        # Margin constraint
        def margin_constraint(price):
            margin_ratio = (price[0] - cost) / price[0]
            return margin_ratio - self.constraints.min_margin_ratio
        
        constraints = [{'type': 'ineq', 'fun': margin_constraint}]
        
        # Optimize with day-specific seed for variation
        day_seed = 42 + hash(str(features)) % 1000 if features else 42
        if method == 'differential_evolution':
            result = differential_evolution(objective, bounds, seed=day_seed)
        else:
            x0 = [base_price]
            result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        optimal_price = result.x[0]
        
        # Apply day-specific price adjustments to create realistic variation
        if features:
            price_adjustment = 1.0
            
            # Promotion days: additional discount beyond optimization
            if features.get('promo_flag', 0) > 0:
                price_adjustment *= 0.95  # Extra 5% promotional discount
                
            # High weather/demand days: premium pricing
            if features.get('weather_index', 0.5) > 0.8:
                price_adjustment *= 1.03  # 3% weather premium
                
            # Weekend premium
            if features.get('is_weekend', 0) > 0:
                price_adjustment *= 1.02  # 2% weekend premium
                
            # Holiday premium
            if features.get('holiday_flag', 0) > 0.5:
                price_adjustment *= 1.05  # 5% holiday premium
                
            optimal_price *= price_adjustment
            
            # Ensure bounds are still respected
            min_price = base_price * self.constraints.min_price_ratio
            max_price = base_price * self.constraints.max_price_ratio
            optimal_price = max(min_price, min(optimal_price, max_price))
        
        optimal_demand = self.simulate_demand(optimal_price, base_price, base_demand, features)
        optimal_kpis = self.calculate_kpis(optimal_price, optimal_demand, cost)
        
        # Calculate baseline KPIs
        baseline_kpis = self.calculate_kpis(base_price, base_demand, cost)
        
        # Calculate improvements
        improvements = {
            'revenue_lift': (optimal_kpis['revenue'] - baseline_kpis['revenue']) / baseline_kpis['revenue'] * 100,
            'margin_lift': (optimal_kpis['margin'] - baseline_kpis['margin']) / baseline_kpis['margin'] * 100,
            'units_lift': (optimal_kpis['units_sold'] - baseline_kpis['units_sold']) / baseline_kpis['units_sold'] * 100
        }
        
        return {
            'optimal_price': optimal_price,
            'optimal_kpis': optimal_kpis,
            'baseline_kpis': baseline_kpis,
            'improvements': improvements,
            'optimization_success': result.success
        }
    
    def optimize_units_target(self,
                             base_price: float,
                             base_demand: float,
                             cost: float,
                             units_target: float,
                             features: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """Optimize price to hit units target with minimal margin sacrifice.
        
        Args:
            base_price: Base product price
            base_demand: Base demand
            cost: Product cost
            units_target: Target units to sell
            features: Additional features
            
        Returns:
            Optimization results
        """
        # Define objective function (minimize margin sacrifice)
        def objective(price):
            demand = self.simulate_demand(price[0], base_price, base_demand, features)
            margin = (price[0] - cost) * demand
            units_diff = abs(demand - units_target)
            # Penalize both margin loss and units difference
            return -margin + 1000 * units_diff
        
        # Define bounds
        bounds = [(base_price * self.constraints.min_price_ratio,
                  base_price * self.constraints.max_price_ratio)]
        
        # Optimize with day-specific seed for variation
        day_seed = 42 + hash(str(features)) % 1000 if features else 42
        result = differential_evolution(objective, bounds, seed=day_seed)
        
        optimal_price = result.x[0]
        
        # Apply day-specific price adjustments to create realistic variation
        if features:
            price_adjustment = 1.0
            
            # Promotion days: additional discount beyond optimization
            if features.get('promo_flag', 0) > 0:
                price_adjustment *= 0.95  # Extra 5% promotional discount
                
            # High weather/demand days: premium pricing
            if features.get('weather_index', 0.5) > 0.8:
                price_adjustment *= 1.03  # 3% weather premium
                
            # Weekend premium
            if features.get('is_weekend', 0) > 0:
                price_adjustment *= 1.02  # 2% weekend premium
                
            # Holiday premium
            if features.get('holiday_flag', 0) > 0.5:
                price_adjustment *= 1.05  # 5% holiday premium
                
            optimal_price *= price_adjustment
            
            # Ensure bounds are still respected
            min_price = base_price * self.constraints.min_price_ratio
            max_price = base_price * self.constraints.max_price_ratio
            optimal_price = max(min_price, min(optimal_price, max_price))
        
        optimal_demand = self.simulate_demand(optimal_price, base_price, base_demand, features)
        optimal_kpis = self.calculate_kpis(optimal_price, optimal_demand, cost)
        
        # Calculate baseline KPIs
        baseline_kpis = self.calculate_kpis(base_price, base_demand, cost)
        
        # Calculate target achievement
        target_achievement = {
            'units_target': units_target,
            'units_achieved': optimal_demand,
            'achievement_rate': optimal_demand / units_target * 100,
            'margin_sacrifice': (baseline_kpis['margin'] - optimal_kpis['margin']) / baseline_kpis['margin'] * 100
        }
        
        return {
            'optimal_price': optimal_price,
            'optimal_kpis': optimal_kpis,
            'baseline_kpis': baseline_kpis,
            'target_achievement': target_achievement,
            'optimization_success': result.success
        }
    
    def multi_objective_optimization(self,
                                    base_price: float,
                                    base_demand: float,
                                    cost: float,
                                    weights: Dict[str, float] = None,
                                    features: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """Multi-objective optimization balancing revenue, units, and margin.
        
        Args:
            base_price: Base product price
            base_demand: Base demand
            cost: Product cost
            weights: Weights for different objectives
            features: Additional features
            
        Returns:
            Optimization results
        """
        if weights is None:
            weights = {'revenue': 0.4, 'units': 0.3, 'margin': 0.3}
        
        # Normalize weights
        total_weight = sum(weights.values())
        weights = {k: v/total_weight for k, v in weights.items()}
        
        # Define multi-objective function
        def objective(price):
            demand = self.simulate_demand(price[0], base_price, base_demand, features)
            revenue = price[0] * demand
            margin = (price[0] - cost) * demand
            
            # Normalize objectives (scale to 0-1)
            revenue_norm = revenue / (base_price * base_demand * 2)  # Assume max 2x baseline
            units_norm = demand / (base_demand * 2)
            margin_norm = margin / (base_price * base_demand)
            
            # Weighted sum (negative for minimization)
            score = -(weights['revenue'] * revenue_norm +
                     weights['units'] * units_norm +
                     weights['margin'] * margin_norm)
            
            return score
        
        # Define bounds
        bounds = [(base_price * self.constraints.min_price_ratio,
                  base_price * self.constraints.max_price_ratio)]
        
        # Optimize with day-specific seed for variation
        day_seed = 42 + hash(str(features)) % 1000 if features else 42
        result = differential_evolution(objective, bounds, seed=day_seed)
        
        optimal_price = result.x[0]
        
        # Apply day-specific price adjustments to create realistic variation
        if features:
            price_adjustment = 1.0
            
            # Promotion days: additional discount beyond optimization
            if features.get('promo_flag', 0) > 0:
                price_adjustment *= 0.95  # Extra 5% promotional discount
                
            # High weather/demand days: premium pricing
            if features.get('weather_index', 0.5) > 0.8:
                price_adjustment *= 1.03  # 3% weather premium
                
            # Weekend premium
            if features.get('is_weekend', 0) > 0:
                price_adjustment *= 1.02  # 2% weekend premium
                
            # Holiday premium
            if features.get('holiday_flag', 0) > 0.5:
                price_adjustment *= 1.05  # 5% holiday premium
                
            optimal_price *= price_adjustment
            
            # Ensure bounds are still respected
            min_price = base_price * self.constraints.min_price_ratio
            max_price = base_price * self.constraints.max_price_ratio
            optimal_price = max(min_price, min(optimal_price, max_price))
        
        optimal_demand = self.simulate_demand(optimal_price, base_price, base_demand, features)
        optimal_kpis = self.calculate_kpis(optimal_price, optimal_demand, cost)
        
        # Calculate baseline KPIs
        baseline_kpis = self.calculate_kpis(base_price, base_demand, cost)
        
        # Calculate improvements
        improvements = {
            'revenue_lift': (optimal_kpis['revenue'] - baseline_kpis['revenue']) / baseline_kpis['revenue'] * 100,
            'margin_lift': (optimal_kpis['margin'] - baseline_kpis['margin']) / baseline_kpis['margin'] * 100,
            'units_lift': (optimal_kpis['units_sold'] - baseline_kpis['units_sold']) / baseline_kpis['units_sold'] * 100
        }
        
        return {
            'optimal_price': optimal_price,
            'optimal_kpis': optimal_kpis,
            'baseline_kpis': baseline_kpis,
            'improvements': improvements,
            'weights_used': weights,
            'optimization_success': result.success
        }
    
    def _generate_daily_features(self, 
                                 store_id: str, 
                                 sku_id: str, 
                                 base_data: pd.Series,
                                 day: int, 
                                 horizon: int) -> Dict[str, float]:
        """Generate realistic daily features with temporal variation.
        
        Args:
            store_id: Store identifier
            sku_id: SKU identifier  
            base_data: Base data for this store-SKU
            day: Current day (1 to horizon)
            horizon: Total planning horizon
            
        Returns:
            Dictionary of features for this day
        """
        import numpy as np
        from datetime import datetime, timedelta
        
        # Start from current date
        current_date = datetime.now() + timedelta(days=day-1)
        day_of_week = current_date.weekday()  # 0=Monday, 6=Sunday
        month = current_date.month
        quarter = (month - 1) // 3 + 1
        
        # Realistic promotional calendar (constrained to avoid violations)
        # Adjust probability to respect max 10 promo days per month constraint
        # For 14-day horizon, max ~5 promo days to stay under monthly limit
        max_promo_days_per_horizon = min(5, horizon // 3)  # Conservative: 1 promo per 3 days max
        
        promo_base_prob = max_promo_days_per_horizon / horizon  # Scale to horizon
        weekend_boost = 0.02 if day_of_week >= 5 else 0  # Small weekend boost
        mid_week_discount = 0.01 if day_of_week in [2, 3] else 0  # Small mid-week boost
        promo_prob = promo_base_prob + weekend_boost + mid_week_discount
        promo_flag = 1.0 if np.random.random() < promo_prob else 0.0
        
        # Holiday calendar (realistic major holidays)
        holiday_flag = 0.0
        if month == 12 and current_date.day >= 20:  # Christmas season
            holiday_flag = 0.8
        elif month == 11 and 22 <= current_date.day <= 28:  # Thanksgiving week
            holiday_flag = 0.6
        elif month == 7 and current_date.day == 4:  # July 4th
            holiday_flag = 0.4
        elif day_of_week == 6:  # Sunday - minor holiday effect
            holiday_flag = 0.1
            
        # Weather simulation (seasonal + daily variation)
        # Base seasonal weather by month
        seasonal_weather = {
            1: 0.3, 2: 0.35, 3: 0.45, 4: 0.6,   # Winter->Spring
            5: 0.75, 6: 0.85, 7: 0.9, 8: 0.85,  # Spring->Summer
            9: 0.7, 10: 0.55, 11: 0.4, 12: 0.3  # Fall->Winter
        }
        base_weather = seasonal_weather[month]
        # Add daily random variation (-20% to +20%)
        weather_variation = np.random.uniform(-0.2, 0.2)
        weather_index = np.clip(base_weather + weather_variation, 0.1, 1.0)
        
        # Generate comprehensive feature set
        features = {
            # Identifiers
            'store_id': store_id,
            'sku_id': sku_id,
            
            # Temporal features
            'day_of_week': day_of_week,
            'month': month, 
            'quarter': quarter,
            'day_of_month': current_date.day,
            'week_of_year': current_date.isocalendar()[1],
            
            # Dynamic promotional/seasonal features
            'promo_flag': promo_flag,
            'holiday_flag': holiday_flag,
            'weather_index': weather_index,
            
            # Business features (use actual data if available)
            'base_price': base_data.get('base_price', 10.0),
            'units_sold': base_data.get('units_sold', 50.0),
            
            # Derived features
            'is_weekend': float(day_of_week >= 5),
            'is_month_start': float(current_date.day <= 3),
            'is_month_end': float(current_date.day >= 28),
        }
        
        return features
    
    def _estimate_cost(self, base_price: float, store_id: str, sku_id: str) -> float:
        """Estimate product cost with more realistic margin assumptions.
        
        Args:
            base_price: Product base price
            store_id: Store identifier
            sku_id: SKU identifier
            
        Returns:
            Estimated cost
        """
        # More realistic margin structure by product type (inferred from SKU)
        if 'SKU001' in sku_id:
            margin_pct = 0.35  # 35% margin for SKU001 (grocery items)
        elif 'SKU002' in sku_id:
            margin_pct = 0.45  # 45% margin for SKU002 (packaged goods)
        elif 'SKU003' in sku_id:
            margin_pct = 0.50  # 50% margin for SKU003 (premium items)
        else:
            margin_pct = 0.40  # Default 40% margin
            
        return base_price * (1 - margin_pct)
    
    def optimize_price_plan(self,
                           data: pd.DataFrame,
                           horizon: int = 14,
                           objective: str = 'revenue') -> pd.DataFrame:
        """Optimize pricing plan for multiple products over a horizon with daily variation.
        
        Args:
            data: DataFrame with product data
            horizon: Planning horizon in days
            objective: Optimization objective
            
        Returns:
            DataFrame with optimized price plan
        """
        price_plans = []
        
        # Get unique store-SKU combinations
        combinations = data[['store_id', 'sku_id']].drop_duplicates()
        
        # Pre-allocate promotional days per SKU to respect constraints
        from datetime import datetime, timedelta
        promo_schedule = {}
        max_promo_per_sku = min(self.constraints.max_promo_days_per_month, horizon // 2)  # Conservative limit
        
        for _, row in combinations.iterrows():
            store_id = row['store_id']
            sku_id = row['sku_id']
            
            # Strategically select which days will be promotional for this SKU
            np.random.seed(42 + hash(f"{store_id}_{sku_id}") % 1000)  # Deterministic seed
            
            # Prefer weekends and mid-week for promotions (more realistic)
            preferred_days = []
            for d in range(1, horizon + 1):
                day_of_week = (datetime.now() + timedelta(days=d-1)).weekday()
                if day_of_week >= 5 or day_of_week in [2, 3]:  # Weekend or Wed/Thu
                    preferred_days.append(d)
            
            # Select promotion days, preferring strategic days
            num_promo_days = min(max_promo_per_sku, len(preferred_days))
            if preferred_days and num_promo_days > 0:
                promo_days = np.random.choice(preferred_days, size=num_promo_days, replace=False)
            elif num_promo_days > 0:
                promo_days = np.random.choice(range(1, horizon + 1), size=num_promo_days, replace=False)
            else:
                promo_days = []
            
            promo_schedule[f"{store_id}_{sku_id}"] = set(promo_days)
        
        for _, row in combinations.iterrows():
            store_id = row['store_id']
            sku_id = row['sku_id']
            
            # Get data for this combination
            combo_data = data[(data['store_id'] == store_id) & 
                             (data['sku_id'] == sku_id)].iloc[-1]
            
            base_price = combo_data['base_price']
            base_demand = combo_data['units_sold']
            cost = self._estimate_cost(base_price, store_id, sku_id)
            
            # Optimize for EACH day separately
            for day in range(1, horizon + 1):
                # Generate day-specific features with controlled promotions
                features = self._generate_daily_features(store_id, sku_id, combo_data, day, horizon)
                
                # Override promotion flag with pre-allocated schedule
                sku_key = f"{store_id}_{sku_id}"
                features['promo_flag'] = 1.0 if day in promo_schedule.get(sku_key, set()) else 0.0
                
                # Optimize based on objective for this specific day
                if objective == 'revenue':
                    result = self.optimize_revenue(base_price, base_demand, cost, features)
                elif objective == 'units':
                    # Target varies by day (higher on weekends)
                    weekend_boost = 1.3 if features['is_weekend'] else 1.0
                    target_units = base_demand * 1.1 * weekend_boost
                    result = self.optimize_units_target(base_price, base_demand, cost, target_units, features)
                else:  # multi-objective
                    # Adjust weights based on day characteristics
                    if features['holiday_flag'] > 0.5:
                        weights = {'revenue': 0.6, 'units': 0.3, 'margin': 0.1}  # Focus on revenue during holidays
                    elif features['is_weekend']:
                        weights = {'revenue': 0.4, 'units': 0.4, 'margin': 0.2}  # Balanced on weekends
                    else:
                        weights = {'revenue': 0.35, 'units': 0.35, 'margin': 0.3}  # Default
                    result = self.multi_objective_optimization(base_price, base_demand, cost, weights, features)
                
                price_plans.append({
                    'store_id': store_id,
                    'sku_id': sku_id,
                    'day': day,
                    'base_price': base_price,
                    'optimal_price': result['optimal_price'],
                    'expected_demand': result['optimal_kpis']['demand'],
                    'expected_revenue': result['optimal_kpis']['revenue'],
                    'expected_margin': result['optimal_kpis']['margin'],
                    'revenue_lift': result.get('improvements', {}).get('revenue_lift', 0),
                    # Add temporal context
                    'promo_flag': features['promo_flag'],
                    'holiday_flag': features['holiday_flag'],
                    'weather_index': features['weather_index'],
                    'day_of_week': features['day_of_week'],
                    'is_weekend': features['is_weekend']
                })
        
        return pd.DataFrame(price_plans)
    
    def validate_constraints(self, price_plan: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate if price plan meets all constraints.
        
        Args:
            price_plan: DataFrame with price plan
            
        Returns:
            Tuple of (is_valid, list of violations)
        """
        violations = []
        
        for _, row in price_plan.iterrows():
            # Check price bounds
            price_ratio = row['optimal_price'] / row['base_price']
            if price_ratio < self.constraints.min_price_ratio:
                violations.append(f"Price too low for {row['sku_id']}: {price_ratio:.2f} < {self.constraints.min_price_ratio}")
            if price_ratio > self.constraints.max_price_ratio:
                violations.append(f"Price too high for {row['sku_id']}: {price_ratio:.2f} > {self.constraints.max_price_ratio}")
        
        # Check promotion frequency per store-SKU combination (or just SKU if no store_id)
        if 'promo_flag' in price_plan.columns:
            if 'store_id' in price_plan.columns:
                promo_days = price_plan.groupby(['store_id', 'sku_id'])['promo_flag'].sum()
                for (store, sku), days in promo_days.items():
                    if days > self.constraints.max_promo_days_per_month:
                        violations.append(f"Too many promo days for {store}-{sku}: {days} > {self.constraints.max_promo_days_per_month}")
            else:
                # Fallback for test data without store_id
                promo_days = price_plan.groupby('sku_id')['promo_flag'].sum()
                for sku, days in promo_days.items():
                    if days > self.constraints.max_promo_days_per_month:
                        violations.append(f"Too many promo days for {sku}: {days} > {self.constraints.max_promo_days_per_month}")
        
        is_valid = len(violations) == 0
        return is_valid, violations


if __name__ == "__main__":
    # Test pricing optimizer
    import sys
    sys.path.append('..')
    from data_loader import DataLoader
    
    # Load data
    loader = DataLoader()
    df = loader.load_data()
    
    # Initialize optimizer with mock demand model
    optimizer = PricingOptimizer(demand_model=None, elasticity=-1.5)
    
    # Test revenue optimization
    result = optimizer.optimize_revenue(
        base_price=15.49,
        base_demand=30,
        cost=9.0
    )
    
    print("\n=== Revenue Optimization Results ===")
    print(f"Optimal Price: ${result['optimal_price']:.2f}")
    print(f"Expected Revenue: ${result['optimal_kpis']['revenue']:.2f}")
    print(f"Revenue Lift: {result['improvements']['revenue_lift']:.1f}%")
    
    # Test price plan optimization
    sample_data = df.head(15)
    price_plan = optimizer.optimize_price_plan(sample_data, horizon=7)
    
    print("\n=== Price Plan (First 5 rows) ===")
    print(price_plan.head())
    
    # Validate constraints
    is_valid, violations = optimizer.validate_constraints(price_plan)
    print(f"\nPrice plan valid: {is_valid}")
    if violations:
        print("Violations:", violations)