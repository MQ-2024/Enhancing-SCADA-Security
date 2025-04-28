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
mqtt_broker = "localhost"  # Update with your MQTT broker address
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
expected_interval = 5  # Seconds
delay_threshold = 15  # Seconds
rapid_threshold = 3  # Seconds

features = ['WindSpeed', 'RotorSpeed', 'GeneratorSpeed', 'PowerOutput', 'GeneratorTemperature']

# Logging setup
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')
logger = logging.getLogger(__name__)

def connect_db():
    """Connect to MySQL database and load data."""
    try:
        engine = create_engine(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}")
        data = pd.read_sql(f"SELECT * FROM {db_table}", engine)
        logger.info(f"Loaded {len(data)} rows")
        return data
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise

def preprocess_data(data):
    """Preprocess data: rename columns, handle timestamps, and clean."""
    data = data.rename(columns={'timestamp': 'Datetime'})
    data['Datetime'] = pd.to_datetime(data['Datetime'], errors='coerce')
    data = data.dropna(subset=['Datetime'] + features)
    logger.info(f"Rows after preprocessing: {len(data)}")
    return data

def temporal_anomaly_detection(data):
    """Flag temporal anomalies based on reading intervals."""
    data = data.sort_values('Datetime')
    data['time_diff'] = data['Datetime'].diff().dt.total_seconds()
    data['delay_anomaly'] = data['time_diff'] > delay_threshold
    data['rapid_anomaly'] = (data['time_diff'] < rapid_threshold) & (data['time_diff'] > 0)
    return data

def rule_based_flagging(data):
    """Apply rule-based flagging for anomalies."""
    data['invalid_datetime'] = data['Datetime'].isna()
    data['threshold_violation'] = False
    for col, (min_val, max_val) in thresholds.items():
        if col in data.columns:
            data['threshold_violation'] |= (data[col] < min_val) | (data[col] > max_val)

    data['fault_period'] = ~((data['StatusAnlage'] == 13.0) & 
                            (((data['Datetime'] >= '2022-02-18 10:28:20') & (data['Datetime'] <= '2022-02-18 10:28:40')) |
                             ((data['Datetime'] >= '2022-02-24 09:42:40') & (data['Datetime'] <= '2022-02-24 09:51:50'))))

    relationаконфеткаrelation_mask = (
        ((data['WindSpeed'] <= 2) & (data['RotorSpeed'] >= 0)) | 
        ((data['WindSpeed'] > 2) & 
         (data['RotorSpeed'] >= (relationships['wind_to_rotor'] - tolerance * 2.48) * data['WindSpeed']) & 
         (data['RotorSpeed'] <= (relationships['wind_to_rotor'] + tolerance * 2.48) * data['WindSpeed']))
    )
    data['relation_violation'] = ~relation_mask
    data['is_unrealistic'] = (data['invalid_datetime'] | data['threshold_violation'] | 
                             data['fault_period'] | data['relation_violation'] | 
                             data['delay_anomaly'] | data['rapid_anomaly'])
    return data

def train_isolation_forest(data):
    """Train Isolation Forest for anomaly detection."""
    scaler = StandardScaler()
    X = scaler.fit_transform(data[features])
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    data['ml_anomaly'] = iso_forest.fit_predict(X) == -1
    return data, iso_forest, scaler

def train_autoencoder(data, epochs=10, sample_size=100000):
    """Train Autoencoder for anomaly detection."""
    # Sample data to manage memory
    if len(data) > sample_size:
        data = data.sample(n=sample_size, random_state=42)
        logger.info(f"Sampled {sample_size} rows for Autoencoder training")
    
    scaler = StandardScaler()
    X = scaler.fit_transform(data[features])
    
    # Build Autoencoder
    model = Sequential([
        Dense(16, activation='relu', input_shape=(len(features),)),
        Dense(8, activation='relu'),
        Dense(16, activation='relu'),
        Dense(len(features), activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')
    
    # Train
    model.fit(X, X, epochs=epochs, batch_size=256, verbose=1)
    
    # Predict and flag anomalies
    reconstructions = model.predict(X)
    mse = np.mean(np.power(X - reconstructions, 2), axis=1)
    threshold = np.percentile(mse, 95)  # Top 5% as anomalies
    data['autoencoder_anomaly'] = mse > threshold
    return data, model, scaler

def on_message(client, userdata, msg):
    """Handle incoming MQTT messages for real-time processing."""
    try:
        payload = json.loads(msg.payload.decode())
        df = pd.DataFrame([payload])
        df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
        userdata['buffer'].append(df)
        
        # Process buffer if enough data or timeout
        if len(userdata['buffer']) >= 10 or (time.time() - userdata['last_process']) > 30:
            data = pd.concat(userdata['buffer'], ignore_index=True)
            data = preprocess_data(data)
            data = temporal_anomaly_detection(data)
            data = rule_based_flagging(data)
            data, _, _ = train_isolation_forest(data)
            data, _, _ = train_autoencoder(data, epochs=5)  # Reduced epochs for real-time
            
            # Publish alerts
            anomalies = data[data['is_unrealistic'] | data['ml_anomaly'] | data['autoencoder_anomaly']]
            for _, row in anomalies.iterrows():
                alert = {
                    'Datetime': row['Datetime'].isoformat(),
                    'Type': 'Delay' if row['delay_anomaly'] else 'Rapid' if row['rapid_anomaly'] else 'Other',
                    'Details': row[features].to_dict()
                }
                client.publish(mqtt_topic_alerts, json.dumps(alert))
            
            userdata['buffer'] = []  # Clear buffer
            userdata['last_process'] = time.time()
            
            # Save anomalies
            anomalies.to_csv('anomalies_realtime.csv', mode='a', index=True, index_label='row_number')
    except Exception as e:
        logger.error(f"Error processing MQTT message: {e}")

def setup_mqtt():
    """Setup MQTT client for real-time processing."""
    client = mqtt.Client(userdata={'buffer': [], 'last_process': time.time()})
    client.on_message = on_message
    client.connect(mqtt_broker)
    client.subscribe(mqtt_topic_data)
    return client

def visualize_anomalies(data):
    """Generate scatter and time-series plots."""
    # Scatter Plot
    sample_data = data.sample(min(10000, len(data)))
    fig = px.scatter(sample_data, x='WindSpeed', y='RotorSpeed', color='is_unrealistic',
                     title='Anomalies: WindSpeed vs RotorSpeed', hover_data=['Datetime', 'time_diff'])
    fig.update_layout(showlegend=True)
    fig.update_xaxes(title='WindSpeed (m/s)')
    fig.update_yaxes(title='RotorSpeed (RPM)')
    fig.write_to_html('anomalies_scatter.html')
    logger.info("Saved scatter plot to 'anomalies_scatter.html'")

    # Time-Series Plot for Time Differences
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

def main(batch_mode=True):
    start_time = time.time()
    logger.info("Starting Anomaly Detection...")

    if batch_mode:
        # Batch processing
        data = connect_db()
        data = preprocess_data(data)
        data = temporal_anomaly_detection(data)
        data = rule_based_flagging(data)
        data, iso_forest, scaler = train_isolation_forest(data)
        data, autoencoder, ae_scaler = train_autoencoder(data, epochs=10)

        # Save and visualize
        anomalies = data[data['is_unrealistic'] | data['ml_anomaly'] | data['autoencoder_anomaly']]
        anomalies.to_csv('anomalies_batch.csv', index=True, index_label='row_number')
        logger.info(f"Saved {len(anomalies)} anomalies to 'anomalies_batch.csv'")
        visualize_anomalies(data)

        # Publish alerts via MQTT
        client = setup_mqtt()
        for _, row in anomalies.iterrows():
            alert = {
                'Datetime': row['Datetime'].isoformat(),
                'Type': 'Delay' if row['delay_anomaly'] else 'Rapid' if row['rapid_anomaly'] else 'Other',
                'Details': row[features].to_dict()
            }
            client.publish(mqtt_topic_alerts, json.dumps(alert))
        client.disconnect()
    else:
        # Real-time MQTT processing
        client = setup_mqtt()
        client.loop_forever()

    # Summary
    summary = f"""Anomaly Detection Summary
Total Rows: {len(data) if batch_mode else 'N/A (Real-Time)'}
Rule-Based Anomalies: {data['is_unrealistic'].sum() if batch_mode else 'N/A'}
ML Anomalies (Isolation Forest): {data['ml_anomaly'].sum() if batch_mode else 'N/A'}
Autoencoder Anomalies: {data['autoencoder_anomaly'].sum() if batch_mode else 'N/A'}
Delay Anomalies (>15s): {data['delay_anomaly'].sum() if batch_mode else 'N/A'}
Rapid Anomalies (<3s): {data['rapid_anomaly'].sum() if batch_mode else 'N/A'}
Runtime: {time.time() - start_time:.2f}s
"""
    logger.info(summary)

if __name__ == "__main__":
    main(batch_mode=True)  # Set to False for real-time MQTT