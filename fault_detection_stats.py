import pandas as pd
import time

raw_input_file = r'D:\datasets\Aventa_AV7_IET_OST_SCADA.csv'

start_time = time.time()
print("Detecting Faults via Stats...")

# Load dataset
data = pd.read_csv(raw_input_file)
data['Datetime'] = pd.to_datetime(data['Datetime'], errors='coerce')
data = data.dropna(subset=['Datetime'])

# Calculate temperature anomalies
temp_extremes = data[(data['GeneratorTemperature'] < -25) | (data['GeneratorTemperature'] > 150)]
print(f"Rows with extreme temperatures: {len(temp_extremes)}")
print(temp_extremes[['Datetime', 'GeneratorTemperature', 'StatusAnlage']].head().to_string())

# Check Feb 2022 anomalies
feb_2022 = data[
    (data['Datetime'].dt.year == 2022) & 
    (data['Datetime'].dt.month == 2) & 
    (data['StatusAnlage'] == 13.0)
]
print(f"\nFeb 2022 fault rows: {len(feb_2022)}")
print(feb_2022[['Datetime', 'GeneratorTemperature', 'StatusAnlage']].head().to_string())
print(f"Completed in {time.time() - start_time:.2f}s")