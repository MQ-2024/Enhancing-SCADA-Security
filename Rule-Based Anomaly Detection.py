import pandas as pd
import time

start_time = time.time()
print("Starting (Testing Normal Dataset with Relationships)...")

# Load the normal dataset - swap path as needed
print("Loading CSV...")
data = pd.read_csv(r'D:\for_AI\Aventa_AV7_IET_OST_SCADA.csv')  # Replace with your normal file
print(f"CSV loaded in {time.time() - start_time:.2f} seconds")

# Quick inspection of max values
print("Max values in dataset:")
print(data[['GeneratorTemperature', 'GeneratorSpeed', 'PowerOutput', 'WindSpeed', 'RotorSpeed']].max().to_string())

# Prepare the data
print("Preparing data...")
data['Datetime'] = pd.to_datetime(data['Datetime'], format='%Y-%m-%d %H:%M:%S', errors='coerce')  # Adjust format if needed
data = data.dropna(subset=['Datetime']).sort_values('Datetime').reset_index(drop=True)
print(f"Total rows: {len(data)}")

# Calculate time differences and rates
data['Time_Diff'] = data['Datetime'].diff().dt.total_seconds().fillna(0)
data['Delta_GeneratorTemperature'] = data['GeneratorTemperature'].diff().fillna(0)
data['Delta_GeneratorSpeed'] = data['GeneratorSpeed'].diff().fillna(0)
data['Temp_Rate'] = data['Delta_GeneratorTemperature'] / data['Time_Diff'].replace(0, 1)
data['Speed_Rate'] = data['Delta_GeneratorSpeed'] / data['Time_Diff'].replace(0, 1)

# Normal limits
temp_rate_max = 19.5  # °C/s
speed_rate_max = 2.00  # RPM/s
print(f"Using Temp_Rate limit: {temp_rate_max}°C/s, Speed_Rate limit: {speed_rate_max} RPM/s")

# Flag rate anomalies
data['Temp_Anomaly'] = data['Temp_Rate'].abs() > temp_rate_max
data['Speed_Anomaly'] = data['Speed_Rate'].abs() > speed_rate_max

# Relationship ratios (from tampered tuning, may adjust for normal)
wind_to_power_avg = 0.24  # May tweak after units
wind_to_rotor_avg = 7.00
rotor_to_gen_avg = 6.06
gen_to_power_avg = 0.04  # May tweak
tolerance = 0.5  # ±50%

# Calculate ratios
data['Wind_to_Power_Ratio'] = data['PowerOutput'] / data['WindSpeed'].replace(0, 1)
data['Wind_to_Rotor_Ratio'] = data['RotorSpeed'] / data['WindSpeed'].replace(0, 1)
data['Rotor_to_Gen_Ratio'] = data['GeneratorSpeed'] / data['RotorSpeed'].replace(0, 1)
data['Gen_to_Power_Ratio'] = data['PowerOutput'] / data['GeneratorSpeed'].replace(0, 1)

# Flag relationship anomalies (skip if both 0)
data['Wind_Power_Anomaly'] = (~data['Wind_to_Power_Ratio'].between(
    wind_to_power_avg * (1 - tolerance), wind_to_power_avg * (1 + tolerance)) & 
    ~((data['PowerOutput'] == 0) & (data['WindSpeed'] == 0)))
data['Wind_Rotor_Anomaly'] = (~data['Wind_to_Rotor_Ratio'].between(
    wind_to_rotor_avg * (1 - tolerance), wind_to_rotor_avg * (1 + tolerance)) & 
    ~((data['RotorSpeed'] == 0) & (data['WindSpeed'] == 0)))
data['Rotor_Gen_Anomaly'] = (~data['Rotor_to_Gen_Ratio'].between(
    rotor_to_gen_avg * (1 - tolerance), rotor_to_gen_avg * (1 + tolerance)) & 
    ~((data['GeneratorSpeed'] == 0) & (data['RotorSpeed'] == 0)))
data['Gen_Power_Anomaly'] = (~data['Gen_to_Power_Ratio'].between(
    gen_to_power_avg * (1 - tolerance), gen_to_power_avg * (1 + tolerance)) & 
    ~((data['PowerOutput'] == 0) & (data['GeneratorSpeed'] == 0)))

# Combine anomalies
data['Anomaly'] = (data['Temp_Anomaly'] | data['Speed_Anomaly'] | 
                   data['Wind_Power_Anomaly'] | data['Wind_Rotor_Anomaly'] | 
                   data['Rotor_Gen_Anomaly'] | data['Gen_Power_Anomaly'])

# Report anomalies
anomalies = data[data['Anomaly']]
print(f"\nDetected {len(anomalies)} anomalies (rate or relationship):")
if not anomalies.empty:
    # Overall top 10
    top_anomalies = anomalies.sort_values(by=['Temp_Rate', 'Speed_Rate'], key=abs, ascending=False)
    print("Top 10 anomalies by rate magnitude (overall):")
    print(top_anomalies[['Datetime', 'Time_Diff', 'GeneratorTemperature', 'GeneratorSpeed', 
                         'PowerOutput', 'WindSpeed', 'RotorSpeed', 
                         'Temp_Rate', 'Speed_Rate', 
                         'Wind_to_Power_Ratio', 'Wind_Rotor_Anomaly']].head(10).to_string())
    # Tampered region (5000-5999, optional for normal data)
    tampered_anomalies = anomalies[(anomalies.index >= 5000) & (anomalies.index < 6000)]
    if not tampered_anomalies.empty:
        print("\nTop 10 anomalies in rows 5000-5999:")
        print(tampered_anomalies[['Datetime', 'Time_Diff', 'GeneratorTemperature', 'GeneratorSpeed', 
                                  'PowerOutput', 'WindSpeed', 'RotorSpeed', 
                                  'Temp_Rate', 'Speed_Rate', 
                                  'Wind_to_Power_Ratio', 'Wind_Rotor_Anomaly']].head(10).to_string())
    # High WindSpeed, low RotorSpeed check
    tampered_like = anomalies[(data['WindSpeed'] > 15) & (data['RotorSpeed'] < 1)]
    if not tampered_like.empty:
        print("\nTop 10 anomalies with WindSpeed > 15 m/s and RotorSpeed < 1 RPM:")
        print(tampered_like[['Datetime', 'Time_Diff', 'GeneratorTemperature', 'GeneratorSpeed', 
                             'PowerOutput', 'WindSpeed', 'RotorSpeed', 
                             'Temp_Rate', 'Speed_Rate', 
                             'Wind_to_Power_Ratio', 'Wind_Rotor_Anomaly']].head(10).to_string())
    extreme_temp = data[data['GeneratorTemperature'] > 100]
    if not extreme_temp.empty:
        print("\nRows with GeneratorTemperature > 100°C:")
        print(extreme_temp[['Datetime', 'Time_Diff', 'GeneratorTemperature', 'GeneratorSpeed', 
                            'PowerOutput', 'WindSpeed', 'RotorSpeed', 
                            'Temp_Rate', 'Speed_Rate']].to_string())
else:
    print("No anomalies detected.")

# Save the analyzed dataset
print("Saving analyzed dataset...")
data.to_csv(r'D:\for_AI\normal_with_anomalies.csv', index=False)
print(f"Dataset saved to 'D:\\for_AI\\normal_with_anomalies.csv' in {time.time() - start_time:.2f} seconds")
print(f"Debug: Raw path = {repr(r'D:\for_AI\\normal_with_anomalies.csv')}")

print(f"Total runtime: {time.time() - start_time:.2f} seconds")