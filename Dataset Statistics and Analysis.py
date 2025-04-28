import pandas as pd

# Load your CSV file with the specified path
data = pd.read_csv(r'D:\for_AI\Aventa_AV7_IET_OST_SCADA.csv')

# List of columns to analyze
columns = ['Datetime', 'RotorSpeed', 'GeneratorTemperature', 'WindSpeed', 'PowerOutput', 
           'SpeiseSpannung', 'StatusAnlage', 'MaxWindHeute', 'offsetWindDirection', 'PitchDeg']

# Dictionary to store results
stats = {}

# Calculate min, max, mean for numerical columns, and unique values for categorical
for col in columns:
    if col == 'Datetime':
        # For Datetime, show range and count
        stats[col] = {
            'Min': data[col].min(),
            'Max': data[col].max(),
            'Count': data[col].count()
        }
    elif col == 'StatusAnlage':
        # For categorical, show unique values and their counts
        stats[col] = {
            'Unique Values': data[col].value_counts().to_dict(),
            'Count': data[col].count()
        }
    else:
        # For numerical columns, calculate min, max, mean
        stats[col] = {
            'Min': data[col].min(),
            'Max': data[col].max(),
            'Mean': data[col].mean(),
            'Count': data[col].count()
        }

# Print results
print("Dataset Statistics:")
print(f"Total Rows: {len(data)}")
for col, values in stats.items():
    print(f"\nColumn: {col}")
    for key, value in values.items():
        print(f"{key}: {value}")

# Save detailed summary to the specified output path
output_path = r'D:\for_AI\data_summary.csv'
data.describe().to_csv(output_path)
print(f"\nDetailed summary saved to '{output_path}'")