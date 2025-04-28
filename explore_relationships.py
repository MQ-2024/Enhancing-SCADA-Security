import pandas as pd
import time

# Configuration
raw_input_file = r'D:\datasets\Aventa_AV7_IET_OST_SCADA.csv'

start_time = time.time()
print("Exploring Relationships...")

# Load and clean dataset
data = pd.read_csv(raw_input_file)
data['Datetime'] = pd.to_datetime(data['Datetime'], errors='coerce')
data = data.dropna(subset=['Datetime'])

# Define active state
active_data = data[(data['WindSpeed'] > 2) & (data['PowerOutput'] > 0)]
print(f"Active rows: {len(active_data)}")

# Calculate relationships
wind_to_rotor = (active_data['RotorSpeed'] / active_data['WindSpeed']).mean()
rotor_to_gen = (active_data['GeneratorSpeed'] / active_data['RotorSpeed']).mean()
wind_to_power = (active_data['PowerOutput'] / active_data['WindSpeed']).mean()
wind_to_rotor_std = (active_data['RotorSpeed'] / active_data['WindSpeed']).std()
rotor_to_gen_std = (active_data['GeneratorSpeed'] / active_data['RotorSpeed']).std()
wind_to_power_std = (active_data['PowerOutput'] / active_data['WindSpeed']).std()

# Output results
print(f"RotorSpeed = {wind_to_rotor:.2f} * WindSpeed (std = {wind_to_rotor_std:.2f})")
print(f"GeneratorSpeed = {rotor_to_gen:.2f} * RotorSpeed (std = {rotor_to_gen_std:.2f})")
print(f"PowerOutput = {wind_to_power:.2f} * WindSpeed (std = {wind_to_power_std:.2f})")
print(f"Completed in {time.time() - start_time:.2f}s")