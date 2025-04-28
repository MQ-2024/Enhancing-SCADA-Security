import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
import time
import pickle
import os

# Configuration
raw_input_file = r'D:\datasets\Aventa_AV7_IET_OST_SCADA.csv'
cleaned_output_file = r'D:\datasets\Aventa_AV7_IET_OST_SCADA_cleanedtest_ml.csv'
summary_output_file = r'D:\datasets\normal_data_summary_ml.txt'
scaler_file = r'D:\for_AI\tighter1\scaler_ml.pkl'
iso_forest_file = r'D:\for_AI\tighter1\iso_forest_ml.pkl'

os.environ["LOKY_MAX_CPU_COUNT"] = "4"

start_time = time.time()
print("Fully ML Preprocessing: Detecting Rare Entities (Chunked)...")

# Chunk Settings
chunk_size = 5000000  # 5M rows per chunk
chunks = []

# Feature Engineering and Imputation (Chunked)
features = ['WindSpeed', 'RotorSpeed', 'GeneratorSpeed', 'PowerOutput', 'GeneratorTemperature', 'StatusAnlage']
features_extended = features + ['Temp_Rate', 'Speed_Rate', 'Power_Rate']

print("Processing data in chunks...")
first_chunk = True
for chunk in pd.read_csv(raw_input_file, chunksize=chunk_size):
    chunk['Datetime'] = pd.to_datetime(chunk['Datetime'], errors='coerce')
    chunk = chunk.dropna(subset=['Datetime']).sort_values('Datetime').reset_index(drop=True)
    
    # Feature Engineering
    chunk['Time_Diff'] = chunk['Datetime'].diff().dt.total_seconds().fillna(0)
    chunk['Delta_GeneratorTemperature'] = chunk['GeneratorTemperature'].diff().fillna(0)
    chunk['Delta_GeneratorSpeed'] = chunk['GeneratorSpeed'].diff().fillna(0)
    chunk['Delta_PowerOutput'] = chunk['PowerOutput'].diff().fillna(0)
    chunk['Temp_Rate'] = chunk['Delta_GeneratorTemperature'] / chunk['Time_Diff'].replace(0, 1)
    chunk['Speed_Rate'] = chunk['Delta_GeneratorSpeed'] / chunk['Time_Diff'].replace(0, 1)
    chunk['Power_Rate'] = chunk['Delta_PowerOutput'] / chunk['Time_Diff'].replace(0, 1)
    
    # Impute Missing Values
    imputer = SimpleImputer(strategy='median')
    chunk_imputed = pd.DataFrame(imputer.fit_transform(chunk[features_extended]), columns=features_extended, index=chunk.index)
    chunk_ml = pd.concat([chunk[['Datetime']], chunk_imputed], axis=1)
    
    # Scale Features with Emphasis
    scaler = StandardScaler()
    scaled_chunk = scaler.fit_transform(chunk_ml[features_extended])
    scaled_chunk[:, 4] *= 5  # GeneratorTemperature
    scaled_chunk[:, 3] *= 5  # PowerOutput
    scaled_chunk[:, 6] *= 3  # Temp_Rate
    scaled_chunk[:, 8] *= 3  # Power_Rate
    scaled_chunk[:, 5] *= 2  # StatusAnlage
    
    # Outlier Detection
    iso_forest = IsolationForest(contamination=0.01, n_estimators=300, random_state=42, n_jobs=-1)
    core_mask = iso_forest.fit_predict(scaled_chunk) == 1
    cleaned_chunk = chunk_ml[core_mask].copy()
    
    chunks.append(cleaned_chunk)
    print(f"Processed chunk: {len(cleaned_chunk)} rows remaining in {time.time() - start_time:.2f}s")
    
    if first_chunk:
        first_chunk = False
        global_scaler = scaler  # Save scaler from first chunk
        global_iso_forest = iso_forest  # Save model from first chunk

# Combine Chunks
print("Combining chunks...")
cleaned_data = pd.concat(chunks, axis=0).reset_index(drop=True)
print(f"Total cleaned rows: {len(cleaned_data)} in {time.time() - start_time:.2f}s")

# Clustering (Sampled)
print("Clustering (sampling 20% of data)...")
sample_size = int(len(cleaned_data) * 0.2)
sampled_data = cleaned_data.sample(n=sample_size, random_state=42)
sampled_scaled = global_scaler.transform(sampled_data[features_extended])
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
cluster_labels_sample = kmeans.fit_predict(sampled_scaled)
print(f"Sample clustering done in {time.time() - start_time:.2f}s")

print("Extrapolating clusters...")
full_scaled = global_scaler.transform(cleaned_data[features_extended])
cluster_labels_full = kmeans.predict(full_scaled)
cleaned_data['Cluster'] = cluster_labels_full
print(f"Clustering completed in {time.time() - start_time:.2f}s")

# Analyze Clusters
print("Analyzing clusters...")
idle_mask_ml = cleaned_data['Cluster'] == kmeans.cluster_centers_.argmin(axis=0)[0]
active_mask_ml = ~idle_mask_ml
idle_rows = len(cleaned_data[idle_mask_ml])
active_rows = len(cleaned_data[active_mask_ml])

max_values = cleaned_data[features[:-1]].max()
min_values = cleaned_data[features[:-1]].min()
active_data = cleaned_data[active_mask_ml]
wind_to_rotor = (active_data['RotorSpeed'] / active_data['WindSpeed'].replace(0, np.nan)).mean()
rotor_to_gen = (active_data['GeneratorSpeed'] / active_data['RotorSpeed'].replace(0, np.nan)).mean()
wind_to_power = (active_data['PowerOutput'] / active_data['WindSpeed'].replace(0, np.nan)).mean()
wind_to_rotor_std = (active_data['RotorSpeed'] / active_data['WindSpeed'].replace(0, np.nan)).std()
rotor_to_gen_std = (active_data['GeneratorSpeed'] / active_data['RotorSpeed'].replace(0, np.nan)).std()
wind_to_power_std = (active_data['PowerOutput'] / active_data['WindSpeed'].replace(0, np.nan)).std()
print(f"Analysis done in {time.time() - start_time:.2f}s")

# Check Feb 2022 Faults
feb_faults = cleaned_data[
    (cleaned_data['Datetime'].between('2022-02-18 10:28:20', '2022-02-18 10:28:40')) |
    (cleaned_data['Datetime'].between('2022-02-24 09:42:40', '2022-02-24 09:51:50'))
]
print(f"Feb 2022 fault rows remaining: {len(feb_faults)}")
if not feb_faults.empty:
    print("Sample of remaining fault rows:\n", feb_faults[['Datetime', 'GeneratorTemperature', 'StatusAnlage', 'Temp_Rate']].head())

# Save Data
print("Saving cleaned dataset...")
cleaned_data.to_csv(cleaned_output_file, index=False)
print(f"Cleaned dataset saved to {cleaned_output_file} in {time.time() - start_time:.2f}s")

# Save Models
print("Saving ML models...")
with open(scaler_file, 'wb') as f:
    pickle.dump(global_scaler, f)
with open(iso_forest_file, 'wb') as f:
    pickle.dump(global_iso_forest, f)
print(f"Scaler and Isolation Forest saved in {time.time() - start_time:.2f}s")

# Summary
summary = f"""Fully ML Preprocessed Dataset Summary ({raw_input_file})
Total Rows (Raw): {initial_rows}
Total Rows (Cleaned): {len(cleaned_data)}
Active Rows (Clustered): {active_rows}
Idle Rows (Clustered): {idle_rows}
Max Values:
  WindSpeed: {max_values['WindSpeed']:.2f} m/s
  RotorSpeed: {max_values['RotorSpeed']:.2f} RPM
  GeneratorSpeed: {max_values['GeneratorSpeed']:.2f} RPM
  PowerOutput: {max_values['PowerOutput']:.2f} kW
  GeneratorTemperature: {max_values['GeneratorTemperature']:.2f}°C
Min Values:
  WindSpeed: {min_values['WindSpeed']:.2f} m/s
  RotorSpeed: {min_values['RotorSpeed']:.2f} RPM
  GeneratorSpeed: {min_values['GeneratorSpeed']:.2f} RPM
  PowerOutput: {min_values['PowerOutput']:.2f} kW
  GeneratorTemperature: {min_values['GeneratorTemperature']:.2f}°C
Relationships (Active Cluster):
  RotorSpeed = {wind_to_rotor:.2f} * WindSpeed (std = {wind_to_rotor_std:.2f})
  GeneratorSpeed = {rotor_to_gen:.2f} * RotorSpeed (std = {rotor_to_gen_std:.2f})
  PowerOutput = {wind_to_power:.2f} * WindSpeed (std = {wind_to_power_std:.2f})
Runtime: {time.time() - start_time:.2f}s
"""
with open(summary_output_file, 'w') as f:
    f.write(summary)
print(summary)
print(f"Saved summary to {summary_output_file}")
print(f"Total runtime: {time.time() - start_time:.2f}s")