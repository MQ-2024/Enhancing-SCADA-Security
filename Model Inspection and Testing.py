import pickle
import pandas as pd
import numpy as np
import os

# File paths
scaler_path = r"D:\for_AI\tighter1\scaler.pkl"
iso_forest_path = r"D:\for_AI\tighter1\iso_forest.pkl"

# Check file sizes
print("Scaler file size (bytes):", os.path.getsize(scaler_path))
print("Iso Forest file size (bytes):", os.path.getsize(iso_forest_path))

# Load and inspect scaler.pkl
try:
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    print("Scaler type:", type(scaler))
    if hasattr(scaler, 'mean_'):
        print("Scaler means:", scaler.mean_)
        print("Scaler variances:", scaler.var_)
    elif isinstance(scaler, np.ndarray):
        print("Scaler shape:", scaler.shape)
        print("Scaler sample (first 5):", scaler[:5])
except Exception as e:
    print("Error loading scaler.pkl:", e)

# Load and inspect iso_forest.pkl
try:
    with open(iso_forest_path, 'rb') as f:
        iso_forest = pickle.load(f)
    print("Model type:", type(iso_forest))
    if hasattr(iso_forest, 'n_estimators'):
        print("Number of estimators:", iso_forest.n_estimators)
        print("Contamination:", iso_forest.contamination)
except Exception as e:
    print("Error loading iso_forest.pkl:", e)

# Test with sample data if both load successfully
if 'scaler' in locals() and 'iso_forest' in locals() and hasattr(scaler, 'transform'):
    # Sample SCADA data
    sample = pd.DataFrame({
        'WindSpeed': [5.0, 60.0],  # 60 m/s is anomalous
        'RotorSpeed': [65.0, 20.0],
        'GeneratorSpeed': [780.0, 240.0],
        'PowerOutput': [3.3, 10.0]  # 10 kW exceeds 6 kW limit
    })
    scaled_sample = scaler.transform(sample)
    predictions = iso_forest.predict(scaled_sample)
    scores = iso_forest.decision_function(scaled_sample)
    print("Scaled sample:\n", scaled_sample)
    print("Predictions (1 = normal, -1 = anomaly):", predictions)
    print("Anomaly scores:", scores)