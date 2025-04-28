import pandas as pd

# Configuration
raw_input_file = r'D:\datasets\Aventa_AV7_IET_OST_SCADA.csv'

# Load dataset
data = pd.read_csv(raw_input_file)
data['Datetime'] = pd.to_datetime(data['Datetime'], errors='coerce')
data = data.dropna(subset=['Datetime'])

# Detect min/max values
print("Min values in dataset:")
print(data[['WindSpeed', 'RotorSpeed', 'GeneratorSpeed', 'PowerOutput', 'GeneratorTemperature']].min().to_string())
print("\nMax values in dataset:")
print(data[['WindSpeed', 'RotorSpeed', 'GeneratorSpeed', 'PowerOutput', 'GeneratorTemperature']].max().to_string())