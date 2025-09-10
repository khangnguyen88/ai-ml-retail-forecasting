# Quick Start Guide - Simplified Project Structure

## IMPORTANT: Model Compatibility Update
**If you have existing models from earlier versions, you'll need to retrain them due to changes in the save/load mechanism. The new version uses pickle instead of joblib for better compatibility.**

## Core Scripts (All You Need!)

After consolidation, the project now has just **3 main scripts**:

### 1. `pipeline.py` - Pricing & Analysis Interface
Handles pricing optimization, demand simulation, and model comparison.

```bash
# Compare models
python pipeline.py compare --output report.txt

# Optimize pricing
python pipeline.py optimize --horizon 14 --objective revenue

# Simulate demand
python pipeline.py simulate --horizon 14 --price-plan plan.csv
```

### 2. `train_with_pipeline.py` - Training (Recommended)
Modern training with optimized hyperparameters - NO WARNINGS!

```bash
# Quick training with optimized parameters (~5 minutes) - BEST: R² = 0.923
python train_with_pipeline.py --quick

# Standard training (~10 minutes)
python train_with_pipeline.py --max-sarimax 5

# With grid search (~30 minutes) - May have lower performance
python train_with_pipeline.py --optimize
```

### 3. `predict_with_pipeline.py` - Predictions
Make predictions with proper preprocessing - NO WARNINGS!

```bash
# Test with demo data
python predict_with_pipeline.py --demo

# Predict on new CSV file
python predict_with_pipeline.py --input data.csv --output predictions.csv
```

---

## Recommended Workflow

### Step 1: Install Dependencies
```bash
# Use existing virtual environment
source venv/bin/activate
pip install -r requirements.txt
```

**Note:** This project already includes a `venv/` directory with pre-installed dependencies. Simply activate it instead of creating a new virtual environment.

### Step 2: Train Model
```bash
# For best performance (RECOMMENDED - R² = 0.923)
python train_with_pipeline.py --quick

# For grid search (may have lower performance)
python train_with_pipeline.py --optimize
```

### Step 3: Make Predictions
```bash
# Test it works
python predict_with_pipeline.py --demo

# Use on real data
python predict_with_pipeline.py --input new_data.csv --output results.csv
```

### Step 4: Explore Data Analysis (Optional)
```bash
# Activate virtual environment
source venv/bin/activate

# Start Jupyter Notebook
jupyter notebook

# Navigate to notebooks/eda_analysis.ipynb in the browser interface
```

---

## Output Files

After training, you'll have:
- `models/gb_model_pipeline.pkl` - Optimized Gradient Boosting model (R² = 0.923)
- `models/pipeline.pkl` - Preprocessing pipeline with 88 features
- `models/feature_importance.csv` - Feature rankings
- `models/training_metadata_pipeline.json` - Training info and metrics

---

## Additional Features

### Jupyter Notebook Analysis
- `notebooks/eda_analysis.ipynb` - Interactive data exploration
- Visualizations and statistical analysis
- Feature correlation analysis
- Model interpretation insights

## Benefits of This Structure

1. **Simpler** - Just 3 scripts to understand
2. **No Warnings** - Preprocessing pipeline handles all features
3. **Consistent** - Same preprocessing for train and predict
4. **Production Ready** - Pipeline can be deployed as-is
5. **Interactive Analysis** - Jupyter notebook for deep data exploration

---
