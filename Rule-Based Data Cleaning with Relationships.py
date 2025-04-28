import pandas as pd
import time

# Configuration
raw_input_file = r'D:\datasets\Aventa_AV7_IET_OST_SCADA.csv'
cleaned_output_file = r'D:\datasets\Aventa_AV7_IET_OST_SCADA_cleanedtest_less_aggressive.csv'
summary_output_file = r'D:\datasets\normal_data_summary_less_aggressive.txt'

thresholds = {
    'PowerOutput': (-0.5, 7.5),
    'RotorSpeed': (0, 84),
    'GeneratorSpeed': (0, 1000),
    'WindSpeed': (0, 50),
    'GeneratorTemperature': (-25, 150)
}

relationships = {
    'wind_to_rotor': 12.99  # RotorSpeed = 12.99 * WindSpeed (std = 2.48)
}
tolerance = 4.0  # ±9.92

start_time = time.time()
print("Processing Dataset with Less Aggressive Active Relationships...")

# Load and Clean
print("Loading and cleaning raw CSV...")
data = pd.read_csv(raw_input_file)
print(f"Loaded {len(data)} rows in {time.time() - start_time:.2f}s")

data['Datetime'] = pd.to_datetime(data['Datetime'], errors='coerce')
initial_rows = len(data)
data = data.dropna(subset=['Datetime'])
print(f"Dropped {initial_rows - len(data)} rows with invalid Datetime")

# Apply Cleaning
threshold_mask = True
for col, (min_val, max_val) in thresholds.items():
    if col in data.columns:
        threshold_mask &= (data[col] >= min_val) & (data[col] <= max_val)

fault_mask = ~((data['StatusAnlage'] == 13.0) & 
               (((data['Datetime'] >= '2022-02-18 10:28:20') & (data['Datetime'] <= '2022-02-18 10:28:40')) |
                ((data['Datetime'] >= '2022-02-24 09:42:40') & (data['Datetime'] <= '2022-02-24 09:51:50'))))

relation_mask = (
    ((data['WindSpeed'] <= 2) & (data['RotorSpeed'] >= 0)) | 
    ((data['WindSpeed'] > 2) & 
     (data['RotorSpeed'] >= (relationships['wind_to_rotor'] - tolerance * 2.48) * data['WindSpeed']) & 
     (data['RotorSpeed'] <= (relationships['wind_to_rotor'] + tolerance * 2.48) * data['WindSpeed']))
)

data = data[threshold_mask & fault_mask & relation_mask]
cleaned_rows = len(data)
print(f"Cleaned to {cleaned_rows} rows (removed {initial_rows - cleaned_rows}) in {time.time() - start_time:.2f}s")

# Save Cleaned Data
data.to_csv(cleaned_output_file, index=False)
print(f"Cleaned dataset saved to {cleaned_output_file}")

# Analyze
print("Analyzing full dataset...")
idle_mask = (data['RotorSpeed'] == 0) & (data['GeneratorSpeed'] == 0) & (data['PowerOutput'] <= 0)
idle_rows = len(data[idle_mask])
active_rows = len(data[(data['WindSpeed'] > 2) & (data['PowerOutput'] > 0)])

max_values = data[['WindSpeed', 'RotorSpeed', 'GeneratorSpeed', 'PowerOutput', 'GeneratorTemperature']].max()
min_values = data[['WindSpeed', 'RotorSpeed', 'GeneratorSpeed', 'PowerOutput', 'GeneratorTemperature']].min()
mean_values = data[['WindSpeed', 'RotorSpeed', 'GeneratorSpeed', 'PowerOutput', 'GeneratorTemperature']].mean()

active_data = data[(data['WindSpeed'] > 2) & (data['PowerOutput'] > 0)]
wind_to_rotor = (active_data['RotorSpeed'] / active_data['WindSpeed']).mean()
rotor_to_gen = (active_data['GeneratorSpeed'] / active_data['RotorSpeed']).mean()
wind_to_power = (active_data['PowerOutput'] / active_data['WindSpeed']).mean()
wind_to_rotor_std = (active_data['RotorSpeed'] / active_data['WindSpeed']).std()
rotor_to_gen_std = (active_data['GeneratorSpeed'] / active_data['RotorSpeed']).std()
wind_to_power_std = (active_data['PowerOutput'] / active_data['WindSpeed']).std()

# Summary
summary = f"""Normal Dataset Summary with Less Aggressive Active Relationships ({raw_input_file})
Total Rows (Raw): {initial_rows}
Total Rows (Cleaned): {cleaned_rows}
Active Rows (WindSpeed > 2, PowerOutput > 0): {active_rows}
Idle Rows (RotorSpeed = 0, GeneratorSpeed = 0, PowerOutput <= 0): {idle_rows}
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
Mean Values:
  WindSpeed: {mean_values['WindSpeed']:.2f} m/s
  RotorSpeed: {mean_values['RotorSpeed']:.2f} RPM
  GeneratorSpeed: {mean_values['GeneratorSpeed']:.2f} RPM
  PowerOutput: {mean_values['PowerOutput']:.2f} kW
  GeneratorTemperature: {mean_values['GeneratorTemperature']:.2f}°C
Relationships (Active States - Post-Cleaning):
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