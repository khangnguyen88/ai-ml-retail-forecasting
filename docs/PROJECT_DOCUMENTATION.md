# Dynamic Demand Forecasting & Pricing Optimization System

A production-ready solution for retail demand forecasting and price optimization, comparing Classical Time Series (SARIMAX) against modern Gradient Boosting ensemble method.

## Project Overview

This system addresses the challenge of forecasting demand for a multi-store retailer (5 stores, 3 SKUs) while optimizing pricing decisions to maximize business KPIs. The solution demonstrates that optimized ensemble methods achieve 94.1% R² while classical SARIMAX models are fitted for individual store-SKU combinations using automated model selection.

## Key Achievements

- **94.1% R² Score**: Excellent predictive accuracy on test data after data leakage fixes
- **3.79 RMSE**: Optimized Gradient Boosting model performance
- **4.2% MAPE**: Mean Absolute Percentage Error
- **91 Features**: Comprehensive feature engineering pipeline with advanced interactions
- **Data Leakage Fixed**: Implemented proper temporal validation to prevent future data leakage
- **Production Ready**: CLI interface, Docker support, extensive testing with 37 passing tests

## Technical Architecture

### Core Components

```
src/
├── data_loader.py          # Data ingestion and preprocessing
├── features.py             # 60+ engineered features
├── models/
│   ├── sarimax_model.py   # Classical time series (SARIMAX)
│   ├── ensemble_model.py  # Gradient Boosting ensemble model
│   └── model_comparison.py # Performance evaluation
├── pricing/
│   └── optimizer.py        # Price optimization engine
└── pipeline.py             # Main CLI orchestrator
```

## Installation

### Prerequisites
- Python 3.8+
- 4GB RAM minimum
- Git

### Setup

```bash
# Clone repository
git clone [repository-url]
cd ai-ml-challenge-main

# Install dependencies
pip install -r requirements.txt

# Verify installation
python pipeline.py --help
```

## Usage Guide

### 1. Model Training

Train and evaluate forecasting models:

```bash
python pipeline.py train \
    --train-start 2023-01-01 \
    --train-end 2023-09-30 \
    --test-start 2023-10-01 \
    --test-end 2023-12-31 \
    --model-type ensemble
```

Options:
- `--model-type`: Choose `sarimax`, `ensemble`, or `both`
- `--output-dir`: Directory for saved models (default: `models/`)

### 2. Model Comparison

Generate comprehensive comparison report:

```bash
python pipeline.py compare --output comparison_report.txt
```

This generates:
- Statistical significance tests
- Performance metrics comparison
- Visualization plots
- Recommendations

### 3. Price Optimization

Optimize pricing strategy for future periods:

```bash
python pipeline.py optimize \
    --horizon 14 \
    --objective revenue \
    --constraints constraints.yaml \
    --out optimized_prices.csv
```

Objectives available:
- `revenue`: Maximize total revenue
- `units`: Hit unit targets with minimal margin loss
- `multi`: Balance multiple objectives

### 4. Demand Simulation

Simulate demand for proposed pricing:

```bash
python pipeline.py simulate \
    --horizon 14 \
    --price-plan optimized_prices.csv \
    --out simulation_results.csv
```

## Feature Engineering

The system creates 91 features across six categories:

### Temporal Features
- Lag features (1, 7, 14, 28 days)
- Rolling statistics (7, 14, 28-day windows)
- Seasonal indicators (day of week, month, quarter)
- Cyclic encodings (sine/cosine transformations)

### Price Features
- Price ratios and changes
- Elasticity indicators
- Log transformations
- Historical comparisons

### Promotion Features
- Promotion depth and frequency
- Days since/until promotion
- Consecutive promo days
- Promotion effectiveness

### Competition Features
- Price gaps and ratios
- Competitive positioning
- Undercut indicators

### External Features
- Weather impact
- Holiday effects
- Event proximity

### Interaction Features
- Price × Promotion
- Price × Holiday
- Weather × Demand

## Pricing Optimization

### Supported Constraints

Configure constraints in `constraints.yaml`:

```yaml
min_price_ratio: 0.7        # Minimum 70% of base price
max_price_ratio: 1.2        # Maximum 120% of base price  
max_promo_days_per_month: 10
max_discount_depth: 0.5
min_margin_ratio: 0.1
inventory_constraints:
  SKU001: 1000
  SKU002: 1500
```

### Elasticity Modeling

The system estimates price elasticity dynamically:
- Default elasticity: -1.5
- SKU-specific calibration
- Promotion impact: +30% demand
- Holiday impact: +20% demand

## Model Selection Guidelines

### Use SARIMAX When:
- Dataset has fewer than 1,000 observations
- Interpretability is critical
- Strong seasonal patterns exist
- Limited external features available
- Need confidence intervals

### Use Ensemble When:
- Large datasets (5,000+ observations)
- Accuracy is top priority  
- Rich feature set available
- Real-time predictions required
- Complex non-linear relationships

## Performance Validation

### Current Performance Metrics
- **R² Score**: 94.14% (Excellent predictive accuracy)
- **RMSE**: 3.79 units (Root Mean Squared Error)
- **MAPE**: 4.20% (Mean Absolute Percentage Error)  
- **MAE**: 1.63 units (Mean Absolute Error)
- **Coverage**: 85.65% (Prediction interval coverage)
- **All Tests Passing**: 37/37 unit tests successful

### Key Metrics Explained
- **RMSE**: Root Mean Squared Error (lower is better)
- **MAPE**: Mean Absolute Percentage Error (lower is better)
- **R²**: Variance explained (higher is better)

## Docker Deployment

```bash
# Build image
docker build -t demand-forecasting .

# Run container
docker run -v $(pwd)/data:/app/data demand-forecasting

# Or use docker-compose
docker-compose up
```

## Testing

Run comprehensive test suite:

```bash
# Unit tests
pytest tests/ -v

# With coverage report
pytest tests/ --cov=src --cov-report=html

# Integration tests
python tests/integration_test.py
```

## Project Structure

```
.
├── pipeline.py              # Main CLI interface
├── src/                     # Source code
│   ├── data_loader.py       # Data management
│   ├── features.py          # Feature engineering
│   ├── models/              # Forecasting models
│   └── pricing/             # Optimization logic
├── tests/                   # Test suite
├── notebooks/               # Analysis notebooks
├── reports/                 # Generated reports
├── models/                  # Saved model artifacts
├── requirements.txt         # Dependencies
├── Dockerfile               # Container definition
├── docker-compose.yml       # Multi-service setup
└── README.md                # This file
```

## Advanced Features

### **Currently Implemented**
- **Robustness Features**:
  - Stockout detection and handling
  - Outlier management  
  - Missing data imputation
  - Endogeneity mitigation

- **Data Leakage Prevention**:
  - Fixed store-SKU average demand calculation using historical data only
  - Fixed market share calculation to prevent future data leakage  
  - Fixed price vs average using proper temporal validation
  - Fixed demand trend features with shifted rolling windows
  - Fixed relative price position using historical market statistics

### **Planned Advanced Features**
- **Hierarchical Forecasting** (Roadmap):
  - Store-level aggregation constraints
  - SKU family coherence
  - Top-down/bottom-up reconciliation

- **Monitoring Capabilities** (Roadmap):
  - Model drift detection
  - Performance tracking
  - A/B testing framework
  - Real-time dashboards

## Troubleshooting

### Common Issues

1. **Memory errors during training**
   - Reduce batch size in ensemble models
   - Use subset of features
   - Enable GPU support if available

2. **SARIMAX convergence warnings**
   - Increase max iterations
   - Simplify seasonal order
   - Check data stationarity

3. **Poor forecast accuracy**
   - Verify data quality
   - Check for structural breaks
   - Retrain with recent data

## Contributing

We welcome contributions! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## **Future Development Roadmap**

### **Planned ML Enhancements**
- [ ] **Deep Learning Models**: LSTM, Transformer architectures for time series
- [ ] **AutoML Integration**: Automated model selection and hyperparameter tuning
- [ ] **Causal Inference**: Advanced causal modeling for pricing impact analysis
- [ ] **Multi-objective Evolutionary Algorithms**: Advanced optimization techniques

### **Planned Infrastructure & Deployment**
- [ ] **Real-time Streaming Predictions**: Live model serving and prediction APIs
- [ ] **Cloud Deployment Templates**: AWS/GCP/Azure deployment configurations
- [ ] **Web Application**: Streamlit/FastAPI dashboard for interactive analysis
- [ ] **Stock Data Integration**: yfinance integration for market analysis

### **Planned Business Intelligence**
- [ ] **LLM-Powered Reporting**: GPT-4 integration for automated narrative reports
- [ ] **Executive Dashboards**: Business-ready visualization and KPI tracking
- [ ] **A/B Testing Platform**: Integrated experimentation framework


- Research papers on demand forecasting and pricing optimization

---
