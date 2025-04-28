import pandas as pd
import time
import pymysql
from sqlalchemy import create_engine
from datetime import datetime

# Configuration
db_name = "scada_db"
db_user = "root"
db_password = "scada123"
db_host = "192.168.3.45"
db_table = "scada_data_nodredAI"

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

# Required columns based on table description
required_columns = [
    'timestamp', 'RotorSpeed', 'GeneratorSpeed', 'GeneratorTemperature',
    'WindSpeed', 'PowerOutput', 'offsetWindDirection', 'SpeiseSpannung',
    'PitchDeg', 'StatusAnlage', 'MaxWindHeute'
]

def log_message(message):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}")

start_time = time.time()
log_message("Processing Dataset with Unrealistic Data Flagging...")

# Step 1: Test Raw MySQL Connection
log_message(f"Testing raw MySQL connection to '{db_host}' (port 3306)...")
try:
    connection = pymysql.connect(
        host=db_host,
        port=3306,
        user=db_user,
        password=db_password,
        database=db_name,
        connect_timeout=5
    )
    log_message("Successfully connected to MySQL server")
    
    # Verify table exists and has data
    with connection.cursor() as cursor:
        cursor.execute(f"SHOW TABLES LIKE '{db_table}'")
        if not cursor.fetchone():
            log_message(f"Error: Table '{db_table}' does not exist in database '{db_name}'")
            raise Exception(f"Table '{db_table}' not found")
        log_message(f"Table '{db_table}' found")
        
        # Check column names
        cursor.execute(f"DESCRIBE {db_table}")
        db_columns = [row[0] for row in cursor.fetchall()]
        log_message(f"Table columns: {db_columns}")
        missing_columns = [col for col in required_columns if col not in db_columns]
        if missing_columns:
            log_message(f"Error: Missing required columns in table '{db_table}': {missing_columns}")
            raise Exception(f"Missing columns: {missing_columns}")
        
        # Check if table has data
        cursor.execute(f"SELECT COUNT(*) FROM {db_table}")
        row_count = cursor.fetchone()[0]
        if row_count == 0:
            log_message(f"Error: Table '{db_table}' is empty")
            raise Exception(f"Table '{db_table}' is empty")
        log_message(f"Table '{db_table}' contains {row_count} rows")
    
    connection.close()
except Exception as e:
    log_message(f"Error: Failed to connect to MySQL server on '{db_host}' - {e}")
    raise

# Step 2: Load Data with SQLAlchemy
log_message(f"Loading data from table '{db_table}' in database '{db_name}'...")
try:
    engine = create_engine(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}")
    data = pd.read_sql(f"SELECT * FROM {db_table}", engine)
    log_message(f"Successfully loaded {len(data)} rows in {time.time() - start_time:.2f}s")
except Exception as e:
    log_message(f"Error: Failed to load data from table '{db_table}' - {e}")
    raise

# Step 3: Verify Data Loaded
if 'data' not in locals() or data is None:
    log_message("Error: Data loading failed, 'data' variable not defined")
    raise Exception("Data loading failed")

# Step 4: Log Timestamp Range
log_message(f"Timestamp range: {data['timestamp'].min()} to {data['timestamp'].max()}")

# Step 5: Rename timestamp to Datetime
log_message("Renaming 'timestamp' column to 'Datetime'...")
data = data.rename(columns={'timestamp': 'Datetime'})
data['Datetime'] = pd.to_datetime(data['Datetime'], errors='coerce')

# Step 6: Flag Unrealistic Data (Without Dropping)
log_message("Flagging unrealistic data...")
initial_rows = len(data)

# Flag invalid Datetime
data['invalid_datetime'] = data['Datetime'].isna()
invalid_datetime_rows = data['invalid_datetime'].sum()
if invalid_datetime_rows > 0:
    log_message(f"Alert: {invalid_datetime_rows} rows have invalid Datetime values")
    invalid_rows = data[data['invalid_datetime']].index.tolist()
    log_message(f"Row numbers with invalid Datetime: {invalid_rows[:10]} (showing first 10 of {len(invalid_rows)})")

# Flag threshold violations
threshold_mask = True
threshold_violations = pd.Series(False, index=data.index)
for col, (min_val, max_val) in thresholds.items():
    if col in data.columns:
        violation = (data[col] < min_val) | (data[col] > max_val)
        threshold_violations |= violation
        if violation.sum() > 0:
            log_message(f"Alert: {violation.sum()} rows violate {col} threshold ({min_val} to {max_val})")
            violation_rows = data[violation].index.tolist()
            log_message(f"Row numbers violating {col}: {violation_rows[:10]} (showing first 10 of {len(violation_rows)})")
data['threshold_violation'] = threshold_violations
threshold_violation_rows = data['threshold_violation'].sum()
log_message(f"Total rows with threshold violations: {threshold_violation_rows}")

# Flag fault periods
fault_mask = ~((data['StatusAnlage'] == 13.0) & 
               (((data['Datetime'] >= '2022-02-18 10:28:20') & (data['Datetime'] <= '2022-02-18 10:28:40')) |
                ((data['Datetime'] >= '2022-02-24 09:42:40') & (data['Datetime'] <= '2022-02-24 09:51:50'))))
data['fault_period'] = ~fault_mask
fault_period_rows = data['fault_period'].sum()
if fault_period_rows > 0:
    log_message(f"Alert: {fault_period_rows} rows fall in fault periods")
    fault_rows = data[data['fault_period']].index.tolist()
    log_message(f"Row numbers in fault periods: {fault_rows[:10]} (showing first 10 of {len(fault_rows)})")

# Flag relationship violations
relation_mask = (
    ((data['WindSpeed'] <= 2) & (data['RotorSpeed'] >= 0)) | 
    ((data['WindSpeed'] > 2) & 
     (data['RotorSpeed'] >= (relationships['wind_to_rotor'] - tolerance * 2.48) * data['WindSpeed']) & 
     (data['RotorSpeed'] <= (relationships['wind_to_rotor'] + tolerance * 2.48) * data['WindSpeed']))
)
data['relation_violation'] = ~relation_mask
relation_violation_rows = data['relation_violation'].sum()
if relation_violation_rows > 0:
    log_message(f"Alert: {relation_violation_rows} rows violate WindSpeed-RotorSpeed relationship")
    relation_rows = data[data['relation_violation']].index.tolist()
    log_message(f"Row numbers violating relationship: {relation_rows[:10]} (showing first 10 of {len(relation_rows)})")

# Combine flags into is_unrealistic
data['is_unrealistic'] = (data['invalid_datetime'] | data['threshold_violation'] | 
                         data['fault_period'] | data['relation_violation'])
unrealistic_rows = data['is_unrealistic'].sum()
log_message(f"Total unrealistic rows flagged: {unrealistic_rows}")

# Log sample of unrealistic rows (first 5 for brevity)
if unrealistic_rows > 0:
    log_message("Sample of unrealistic rows (first 5):")
    sample_unrealistic = data[data['is_unrealistic']].head(5)[['Datetime', 'WindSpeed', 'RotorSpeed', 'PowerOutput', 
                                                             'invalid_datetime', 'threshold_violation', 
                                                             'fault_period', 'relation_violation']]
    sample_unrealistic['row_number'] = sample_unrealistic.index
    log_message(sample_unrealistic.to_string(index=False))

# Step 7: Analyze
log_message("Analyzing dataset...")
idle_mask = (data['RotorSpeed'] == 0) & (data['GeneratorSpeed'] == 0) & (data['PowerOutput'] <= 0)
idle_rows = len(data[idle_mask])
active_rows = len(data[(data['WindSpeed'] > 1) & (data['PowerOutput'] > 0)])

max_values = data[['WindSpeed', 'RotorSpeed', 'GeneratorSpeed', 'PowerOutput', 'GeneratorTemperature']].max()
min_values = data[['WindSpeed', 'RotorSpeed', 'GeneratorSpeed', 'PowerOutput', 'GeneratorTemperature']].min()
mean_values = data[['WindSpeed', 'RotorSpeed', 'GeneratorSpeed', 'PowerOutput', 'GeneratorTemperature']].mean()

active_data = data[(data['WindSpeed'] > 1) & (data['PowerOutput'] > 0)]
if len(active_data) == 0:
    log_message("Warning: No active rows (WindSpeed > 1, PowerOutput > 0) found")
    wind_to_rotor = wind_to_rotor_std = rotor_to_gen = rotor_to_gen_std = wind_to_power = wind_to_power_std = float('nan')
else:
    wind_to_rotor = (active_data['RotorSpeed'] / active_data['WindSpeed']).mean()
    rotor_to_gen = (active_data['GeneratorSpeed'] / active_data['RotorSpeed']).mean()
    wind_to_power = (active_data['PowerOutput'] / active_data['WindSpeed']).mean()
    wind_to_rotor_std = (active_data['RotorSpeed'] / active_data['WindSpeed']).std()
    rotor_to_gen_std = (active_data['GeneratorSpeed'] / active_data['RotorSpeed']).std()
    wind_to_power_std = (active_data['PowerOutput'] / active_data['WindSpeed']).std()

# Step 8: Summary
summary = f"""Dataset Summary with Unrealistic Data Flagging (Table: {db_table})
Total Rows: {len(data)}
Unrealistic Rows (Flagged): {unrealistic_rows}
Active Rows (WindSpeed > 1, PowerOutput > 0): {active_rows}
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
Relationships (Active States):
  RotorSpeed = {wind_to_rotor:.2f} * WindSpeed (std = {wind_to_rotor_std:.2f})
  GeneratorSpeed = {rotor_to_gen:.2f} * RotorSpeed (std = {rotor_to_gen_std:.2f})
  PowerOutput = {wind_to_power:.2f} * WindSpeed (std = {wind_to_power_std:.2f})
Runtime: {time.time() - start_time:.2f}s
"""
log_message(summary)
log_message(f"Total runtime: {time.time() - start_time:.2f}s")

# Save unrealistic rows to CSV for SQL import
data[data['is_unrealistic']].to_csv('unrealistic_rows.csv', index=True, index_label='row_number')
log_message("Saved unrealistic rows to 'unrealistic_rows.csv' for SQL import")