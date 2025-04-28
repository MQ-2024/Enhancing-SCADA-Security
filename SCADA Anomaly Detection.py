import pandas as pd
import time
import pymysql
from sqlalchemy import create_engine
from datetime import datetime
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import plotly.express as px
import paho.mqtt.client as mqtt
import logging
import json
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Configuration
db_name = "scada_db"
db_user = "root"
db_password = "scada123"
db_host = "192.168.3.45"
db_table = "scada_data_nodered"
mqtt_broker = "localhost"
mqtt_topic_alerts = "scada/alerts"
mqtt_topic_data = "scada/data"

thresholds = {
    'PowerOutput': (-0.5, 7.5),
    'RotorSpeed': (0, 84),
    'GeneratorSpeed': (0, 1000),
    'WindSpeed': (0, 50),
    'GeneratorTemperature': (-25, 150)
}

relationships = {'wind_to_rotor': 12.99}
tolerance = 4.0
expected_interval = 5
delay_threshold = 15
rapid_threshold = 3

features = ['WindSpeed', 'RotorSpeed', 'GeneratorSpeed', 'PowerOutput', 'GeneratorTemperature']

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')
logger = logging.getLogger(__name__)

def connect_db():
    try:
        engine = create_engine(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}")
        data = pd.read_sql(f"SELECT * FROM {db_table}", engine)
        logger.info(f"Loaded {len(data)} rows")
        return data
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise

def preprocess_data(data):
    data = data.rename(columns={'timestamp': 'Datetime'})
    data['Datetime'] = pd.to_datetime(data['Datetime'], errors='coerce')
    data = data.dropna(subset=['Datetime'] + features)
    logger.info(f"Rows after preprocessing: {len(data)}")
    return data

def temporal_anomaly_detection(data):
    data = data.sort_values('Datetime')
    data['time_diff'] = data['Datetime'].diff().dt.total_seconds()
    data['delay_anomaly'] = data['time_diff'] > delay_threshold
    data['rapid_anomaly'] = (data['time_diff'] < rapid_threshold) & (data['time_diff'] > 0)
    return data

def rule_based_flagging(data):
    data['invalid_datetime'] = data['Datetime'].isna()
    data['threshold_violation'] = False
    data['violated_rule'] = np.nan  # New column to store specific rule violations
    for col, (min_val, max_val) in thresholds.items():
        if col in data.columns:
            violation_mask = (data[col] < min_val) | (data[col] > max_val)
            data['threshold_violation'] |= violation_mask
            # Store specific violation details
            data.loc[violation_mask & data['violated_rule'].isna(), 'violated_rule'] = f"{col} < {min_val} or > {max_val}"
            data.loc[violation_mask & ~data['violated_rule'].isna(), 'violated_rule'] += f"; {col} < {min_val} or > {max_val}"

    data['fault_period'] = ((data['StatusAnlage'] == 13.0) & 
                           (((data['Datetime'] >= '2022-02-18 10:28:20') & (data['Datetime'] <= '2022-02-18 10:28:40')) |
                            ((data['Datetime'] >= '2022-02-24 09:42:40') & (data['Datetime'] <= '2022-02-24 09:51:50'))))
    data.loc[data['fault_period'] & data['violated_rule'].isna(), 'violated_rule'] = "Fault Period"
    data.loc[data['fault_period'] & ~data['violated_rule'].isna(), 'violated_rule'] += "; Fault Period"

    relation_mask = (
        ((data['WindSpeed'] <= 2) & (data['RotorSpeed'] >= 0)) | 
        ((data['WindSpeed'] > 2) & 
         (data['RotorSpeed'] >= (relationships['wind_to_rotor'] - tolerance * 2.48) * data['WindSpeed']) & 
         (data['RotorSpeed'] <= (relationships['wind_to_rotor'] + tolerance * 2.48) * data['WindSpeed']))
    )
    data['relation_violation'] = ~relation_mask
    # Not adding relation_violation to is_unrealistic yet, as per previous script

    data['is_unrealistic'] = (data['invalid_datetime'] | data['threshold_violation'] | 
                             data['fault_period'] | 
                             data['delay_anomaly'] | data['rapid_anomaly'])

    # Add rule-based violation details for delay and rapid anomalies
    data.loc[data['delay_anomaly'] & data['violated_rule'].isna(), 'violated_rule'] = f"Delay > {delay_threshold}s"
    data.loc[data['delay_anomaly'] & ~data['violated_rule'].isna(), 'violated_rule'] += f"; Delay > {delay_threshold}s"
    data.loc[data['rapid_anomaly'] & data['violated_rule'].isna(), 'violated_rule'] = f"Rapid < {rapid_threshold}s"
    data.loc[data['rapid_anomaly'] & ~data['violated_rule'].isna(), 'violated_rule'] += f"; Rapid < {rapid_threshold}s"
    data.loc[data['invalid_datetime'] & data['violated_rule'].isna(), 'violated_rule'] = "Invalid Datetime"
    data.loc[data['invalid_datetime'] & ~data['violated_rule'].isna(), 'violated_rule'] += "; Invalid Datetime"

    return data

def train_isolation_forest(data):
    scaler = StandardScaler()
    X = scaler.fit_transform(data[features])
    iso_forest = IsolationForest(contamination=0.002, random_state=42)
    data['ml_anomaly'] = iso_forest.fit_predict(X) == -1
    return data, iso_forest, scaler

def train_autoencoder(data, epochs=10, sample_size=100000):
    if len(data) > sample_size:
        data = data.sample(n=sample_size, random_state=42)
        logger.info(f"Sampled {sample_size} rows for Autoencoder training")
    
    scaler = StandardScaler()
    X = scaler.fit_transform(data[features])
    
    model = Sequential([
        Dense(16, activation='relu', input_shape=(len(features),)),
        Dense(8, activation='relu'),
        Dense(16, activation='relu'),
        Dense(len(features), activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')
    
    model.fit(X, X, epochs=epochs, batch_size=256, verbose=1)
    
    reconstructions = model.predict(X)
    mse = np.mean(np.power(X - reconstructions, 2), axis=1)
    threshold = np.percentile(mse, 99.8)
    data['autoencoder_anomaly'] = mse > threshold
    return data, model, scaler

def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode())
        df = pd.DataFrame([payload])
        df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
        userdata['buffer'].append(df)
        
        if len(userdata['buffer']) >= 10 or (time.time() - userdata['last_process']) > 30:
            data = pd.concat(userdata['buffer'], ignore_index=True)
            data = preprocess_data(data)
            data = temporal_anomaly_detection(data)
            data = rule_based_flagging(data)
            data, _, _ = train_isolation_forest(data)
            data, _, _ = train_autoencoder(data, epochs=5)
            
            anomalies = data[data['is_unrealistic'] | data['ml_anomaly'] | data['autoencoder_anomaly']]
            for _, row in anomalies.iterrows():
                alert = {
                    'Datetime': row['Datetime'].isoformat(),
                    'Type': row['Anomaly_Types'],
                    'Details': row[features].to_dict()
                }
                client.publish(mqtt_topic_alerts, json.dumps(alert))
            
            userdata['buffer'] = []
            userdata['last_process'] = time.time()
            
            anomalies.to_csv('anomalies_realtime.csv', mode='a', index=True, index_label='row_number')
    except Exception as e:
        logger.error(f"Error processing MQTT message: {e}")

def setup_mqtt():
    client = mqtt.Client(userdata={'buffer': [], 'last_process': time.time()}, callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
    client.on_message = on_message
    client.connect(mqtt_broker)
    client.subscribe(mqtt_topic_data)
    return client

def visualize_anomalies(data):
    sample_data = data.sample(min(10000, len(data)))
    fig = px.scatter(sample_data, x='WindSpeed', y='RotorSpeed', color='is_unrealistic',
                     title='Anomalies: WindSpeed vs RotorSpeed', hover_data=['Datetime', 'time_diff'])
    fig.update_layout(showlegend=True)
    fig.update_xaxes(title='WindSpeed (m/s)')
    fig.update_yaxes(title='RotorSpeed (RPM)')
    fig.write_html('anomalies_scatter.html')
    logger.info("Saved scatter plot to 'anomalies_scatter.html'")

    plt.figure(figsize=(12, 6))
    plt.plot(data['Datetime'], data['time_diff'], label='Time Difference (s)')
    plt.axhline(y=delay_threshold, color='r', linestyle='--', label='Delay Threshold (15s)')
    plt.axhline(y=rapid_threshold, color='g', linestyle='--', label='Rapid Threshold (3s)')
    anomaly_data = data[data['delay_anomaly'] | data['rapid_anomaly']]
    plt.scatter(anomaly_data['Datetime'], anomaly_data['time_diff'], c='red', label='Temporal Anomalies')
    plt.xlabel('Datetime')
    plt.ylabel('Time Difference (s)')
    plt.title('Temporal Anomalies in Reading Intervals')
    plt.legend()
    plt.savefig('temporal_anomalies.png')
    plt.close()
    logger.info("Saved time-series plot to 'temporal_anomalies.png'")

# Inspect data to diagnose anomaly flagging
data = connect_db()
data = preprocess_data(data)
data = temporal_anomaly_detection(data)

print(data[['time_diff', 'PowerOutput', 'RotorSpeed', 'WindSpeed', 'StatusAnlage']].describe())
print("\nStatusAnlage Value Counts:")
print(data['StatusAnlage'].value_counts())
print(f"\nRows with WindSpeed <= 2: {(data['WindSpeed'] <= 2).sum()}")

def main(batch_mode=True):
    start_time = time.time()
    logger.info("Starting Anomaly Detection...")

    if batch_mode:
        data = connect_db()
        data = preprocess_data(data)
        data = temporal_anomaly_detection(data)
        logger.info(f"Delay Anomalies (>15s): {data['delay_anomaly'].sum()}")
        logger.info(f"Rapid Anomalies (<3s): {data['rapid_anomaly'].sum()}")
        data = rule_based_flagging(data)
        logger.info(f"Invalid Datetime: {data['invalid_datetime'].sum()}")
        logger.info(f"Threshold Violations: {data['threshold_violation'].sum()}")
        logger.info(f"Fault Periods: {data['fault_period'].sum()}")
        logger.info(f"Relation Violations: {data['relation_violation'].sum()}")
        logger.info(f"Total Rule-Based Anomalies: {data['is_unrealistic'].sum()}")
        data, iso_forest, scaler = train_isolation_forest(data)
        logger.info(f"ML Anomalies (Isolation Forest): {data['ml_anomaly'].sum()}")
        data, autoencoder, ae_scaler = train_autoencoder(data, epochs=10)
        logger.info(f"Autoencoder Anomalies: {data['autoencoder_anomaly'].sum()}")

        # Identify anomaly types for each row using vectorized operations
        data['Anomaly_Types'] = ''
        data.loc[data['is_unrealistic'], 'Anomaly_Types'] = 'Rule-Based Anomaly'
        data.loc[data['ml_anomaly'], 'Anomaly_Types'] = np.where(
            data.loc[data['ml_anomaly'], 'Anomaly_Types'].eq(''),
            'ML Anomaly',
            data.loc[data['ml_anomaly'], 'Anomaly_Types'] + '; ML Anomaly'
        )
        data.loc[data['autoencoder_anomaly'], 'Anomaly_Types'] = np.where(
            data.loc[data['autoencoder_anomaly'], 'Anomaly_Types'].eq(''),
            'Autoencoder Anomaly',
            data.loc[data['autoencoder_anomaly'], 'Anomaly_Types'] + '; Autoencoder Anomaly'
        )

        anomalies = data[data['is_unrealistic'] | data['ml_anomaly'] | data['autoencoder_anomaly']].copy()
        anomalies.to_csv('anomalies_batch.csv', index=True, index_label='row_number')
        logger.info(f"Saved {len(anomalies)} anomalies to 'anomalies_batch.csv'")

        # Log detailed breakdown of anomaly types (top 5 per category)
        logger.info("\nDetailed Anomaly Breakdown (Top 5 per Type):")
        
        # Rule-Based Anomalies
        rule_based_anomalies = anomalies[anomalies['is_unrealistic']].head(5)
        if not rule_based_anomalies.empty:
            logger.info("\nRule-Based Anomalies:")
            for idx, row in rule_based_anomalies.iterrows():
                details = f"Row {idx}: Datetime={row['Datetime']}, Violated Rule={row['violated_rule']}, "
                details += ", ".join(f"{feat}={row[feat]:.2f}" for feat in features)
                logger.info(details)

        # ML Anomalies (Isolation Forest)
        ml_anomalies = anomalies[anomalies['ml_anomaly']].head(5)
        if not ml_anomalies.empty:
            logger.info("\nML Anomalies (Isolation Forest):")
            for idx, row in ml_anomalies.iterrows():
                details = f"Row {idx}: Datetime={row['Datetime']}, "
                details += ", ".join(f"{feat}={row[feat]:.2f}" for feat in features)
                logger.info(details)

        # Autoencoder Anomalies
        autoencoder_anomalies = anomalies[anomalies['autoencoder_anomaly']].head(5)
        if not autoencoder_anomalies.empty:
            logger.info("\nAutoencoder Anomalies:")
            for idx, row in autoencoder_anomalies.iterrows():
                details = f"Row {idx}: Datetime={row['Datetime']}, "
                details += ", ".join(f"{feat}={row[feat]:.2f}" for feat in features)
                logger.info(details)

        visualize_anomalies(data)

        client = setup_mqtt()
        for _, row in anomalies.iterrows():
            alert = {
                'Datetime': row['Datetime'].isoformat(),
                'Type': row['Anomaly_Types'],
                'Details': row[features].to_dict()
            }
            client.publish(mqtt_topic_alerts, json.dumps(alert))
        client.disconnect()
    else:
        client = setup_mqtt()
        client.loop_forever()

    summary = f"""Anomaly Detection Summary
Total Rows: {len(data) if batch_mode else 'N/A (Real-Time)'}
Rule-Based Anomalies: {data['is_unrealistic'].sum()}
ML Anomalies (Isolation Forest): {data['ml_anomaly'].sum()}
Autoencoder Anomalies: {data['autoencoder_anomaly'].sum()}
Delay Anomalies (>15s): {data['delay_anomaly'].sum()}
Rapid Anomalies (<3s): {data['rapid_anomaly'].sum()}
Runtime: {time.time() - start_time:.2f}s
"""
    logger.info(summary)
    return data

if __name__ == "__main__":
    main(batch_mode=True)