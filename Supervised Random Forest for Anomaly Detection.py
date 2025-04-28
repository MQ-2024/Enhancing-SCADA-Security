import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import pickle
import time
import pymysql
from sqlalchemy import create_engine
from datetime import datetime
import logging
import json
import paho.mqtt.client as mqtt

# Configuration (from your existing code)
db_name = "scada_db"
db_user = "root"
db_password = "scada123"
db_host = "192.168.3.45"
db_table = "scada_data_nodered"
mqtt_broker = "localhost"
mqtt_topic_alerts = "scada/alerts"
mqtt_topic_data = "scada/data"

features = ['WindSpeed', 'RotorSpeed', 'GeneratorSpeed', 'PowerOutput', 'GeneratorTemperature']
model_path = r"D:\for_AI\random_forest_model.pkl"
scaler_path = r"D:\for_AI\rf_scaler.pkl"

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')
logger = logging.getLogger(__name__)

# Database connection (from your code)
def connect_db():
    try:
        engine = create_engine(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}")
        data = pd.read_sql(f"SELECT * FROM {db_table}", engine)
        logger.info(f"Loaded {len(data)} rows")
        return data
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise

# Preprocessing (adapted from your code)
def preprocess_data(data):
    data = data.rename(columns={'timestamp': 'Datetime'})
    data['Datetime'] = pd.to_datetime(data['Datetime'], errors='coerce')
    data = data.dropna(subset=['Datetime'] + features)
    logger.info(f"Rows after preprocessing: {len(data)}")
    return data

# Train supervised Random Forest model
def train_supervised_model(data):
    # Create pseudo-labels (assuming you have rule-based or unsupervised flags)
    # Example: Label as anomalous if any existing flag is True
    data['label'] = ((data['is_unrealistic'] == True) | 
                     (data['ml_anomaly'] == True) | 
                     (data['autoencoder_anomaly'] == True)).astype(int)
    
    # Features and labels
    X = data[features]
    y = data['label']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest
    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    logger.info("Classification Report:\n" + classification_report(y_test, y_pred))
    logger.info("Confusion Matrix:\n" + str(confusion_matrix(y_test, y_pred)))
    logger.info(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")
    
    # Save model and scaler
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    return model, scaler

# Predict anomalies using trained model
def predict_anomalies(data, model, scaler):
    X = data[features]
    X_scaled = scaler.transform(X)
    data['rf_anomaly'] = model.predict(X_scaled)
    data['rf_anomaly_prob'] = model.predict_proba(X_scaled)[:, 1]
    return data

# Modified MQTT on_message function (integrates with your existing code)
def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode())
        df = pd.DataFrame([payload])
        df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
        userdata['buffer'].append(df)
        
        if len(userdata['buffer']) >= 10 or (time.time() - userdata['last_process']) > 30:
            data = pd.concat(userdata['buffer'], ignore_index=True)
            data = preprocess_data(data)
            data = temporal_anomaly_detection(data)  # From your original code
            data = rule_based_flagging(data)  # From your original code
            data, _, _ = train_isolation_forest(data)  # From your original code
            data, _, _ = train_autoencoder(data, epochs=5)  # From your original code
            
            # Load supervised model and scaler
            with open(model_path, 'rb') as f:
                rf_model = pickle.load(f)
            with open(scaler_path, 'rb') as f:
                rf_scaler = pickle.load(f)
            
            # Predict with supervised model
            data = predict_anomalies(data, rf_model, rf_scaler)
            
            # Combine anomalies
            anomalies = data[data['is_unrealistic'] | data['ml_anomaly'] | 
                            data['autoencoder_anomaly'] | (data['rf_anomaly'] == 1)]
            for _, row in anomalies.iterrows():
                alert = {
                    'Datetime': row['Datetime'].isoformat(),
                    'Type': ('Supervised' if row['rf_anomaly'] == 1 else 
                             'Delay' if row['delay_anomaly'] else 
                             'Rapid' if row['rapid_anomaly'] else 'Other'),
                    'Details': row[features].to_dict(),
                    'Probability': row['rf_anomaly_prob'] if row['rf_anomaly'] == 1 else None
                }
                client.publish(mqtt_topic_alerts, json.dumps(alert))
            
            userdata['buffer'] = []
            userdata['last_process'] = time.time()
            
            anomalies.to_csv('anomalies_realtime.csv', mode='a', index=True, index_label='row_number')
    except Exception as e:
        logger.error(f"Error processing MQTT message: {e}")

# MQTT setup (from your code)
def setup_mqtt():
    client = mqtt.Client(userdata={'buffer': [], 'last_process': time.time()}, 
                        callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
    client.on_message = on_message
    client.connect(mqtt_broker)
    client.subscribe(mqtt_topic_data)
    return client

# Main execution
def main():
    start_time = time.time()
    logger.info("Starting Supervised Anomaly Detection...")
    
    # Load and preprocess data (assuming it has existing anomaly flags)
    data = connect_db()
    data = preprocess_data(data)
    data = temporal_anomaly_detection(data)
    data = rule_based_flagging(data)
    data, _, _ = train_isolation_forest(data)
    data, _, _ = train_autoencoder(data, epochs=10)
    
    # Train supervised model
    rf_model, rf_scaler = train_supervised_model(data)
    
    # Predict on full dataset
    data = predict_anomalies(data, rf_model, rf_scaler)
    
    # Save results
    anomalies = data[data['is_unrealistic'] | data['ml_anomaly'] | 
                    data['autoencoder_anomaly'] | (data['rf_anomaly'] == 1)]
    anomalies.to_csv('anomalies_batch_supervised.csv', index=True, index_label='row_number')
    logger.info(f"Saved {len(anomalies)} anomalies to 'anomalies_batch_supervised.csv'")
    
    # Real-time processing
    client = setup_mqtt()
    client.loop_forever()

if __name__ == "__main__":
    main()