import pandas as pd

# File path
file_path = 'D:/datasets/Aventa_AV7_IET_OST_SCADA.csv'

# Initialize variables
max_temp = float('-inf')
max_temp_row = None

# Process in chunks
chunk_size = 1000000
print("Loading dataset and finding maximum GeneratorTemperature...")

# Read CSV in chunks
chunks = pd.read_csv(file_path, chunksize=chunk_size)

# Iterate through chunks
for chunk in chunks:
    temp_column = 'GeneratorTemperature'
    time_column = 'Datetime'
    status_column = 'StatusAnlage'
    power_column = 'PowerOutput'
    wind_column = 'WindSpeed'
    
    # Check columns
    missing_cols = [col for col in [temp_column, time_column, status_column, power_column, wind_column] 
                   if col not in chunk.columns]
    if missing_cols:
        print(f"Error: {missing_cols} not found. Available columns: {list(chunk.columns)}")
        break
    
    # Find max temp in chunk
    chunk_max_idx = chunk[temp_column].idxmax()
    if pd.notna(chunk_max_idx):  # Ensure there’s a valid max
        chunk_max_temp = chunk.loc[chunk_max_idx, temp_column]
        if chunk_max_temp > max_temp:
            max_temp = chunk_max_temp
            max_temp_row = chunk.loc[chunk_max_idx, 
                                   [time_column, temp_column, status_column, power_column, wind_column]].to_dict()

# Output result
if max_temp_row:
    print(f"\nMaximum GeneratorTemperature Found:")
    print("-" * 80)
    print(f"Datetime: {max_temp_row['Datetime']}")
    print(f"GeneratorTemperature: {max_temp_row['GeneratorTemperature']}°C")
    print(f"Status: {max_temp_row['StatusAnlage']}")
    print(f"PowerOutput: {max_temp_row['PowerOutput']} kW")
    print(f"WindSpeed: {max_temp_row['WindSpeed']} m/s")
    print("-" * 80)
    
    # Save to CSV for reference
    pd.DataFrame([max_temp_row]).to_csv('max_temp_row.csv', index=False)
    print("Result saved to 'max_temp_row.csv'")
else:
    print("No valid GeneratorTemperature data found.")