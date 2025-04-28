# Cell 1: Imports and Configuration
import pandas as pd
import time  # Ensure time is imported
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
%matplotlib inline

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

# Cell 2: Database and Preprocessing Functions
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

# Cell 3: Rule-Based Flagging Function
def rule_based_flagging(data):
    data['invalid_datetime'] = data['Datetime'].isna()
    data['threshold_violation'] = False
    for col, (min_val, max_val) in thresholds.items():
        if col in data.columns:
            data['threshold_violation'] |= (data[col] < min_val) | (data[col] > max_val)

    data['fault_period'] = ((data['StatusAnlage'] == 13.0) & 
                           (((data['Datetime'] >= '2022-02-18 10:28:20') & (data['Datetime'] <= '2022-02-18 10:28:40')) |
                            ((data['Datetime'] >= '2022-02-24 09:42:40') & (data['Datetime'] <= '2022-02-24 09:51:50'))))

    relation_mask = (
        ((data['WindSpeed'] <= 2) & (data['RotorSpeed'] >= 0)) | 
        ((data['WindSpeed'] > 2) & 
         (data['RotorSpeed'] >= (relationships['wind_to_rotor'] - tolerance * 2.48) * data['WindSpeed']) & 
         (data['RotorSpeed'] <= (relationships['wind_to_rotor'] + tolerance * 2.48) * data['WindSpeed']))
    )
    data['relation_violation'] = ~relation_mask
    data['is_unrealistic'] = (data['invalid_datetime'] | data['threshold_violation'] | 
                             data['fault_period'] | 
                             data['delay_anomaly'] | data['rapid_anomaly'])
    return data

# Cell 4: Machine Learning Functions
def train_isolation_forest(data):
    scaler = StandardScaler()
    X = scaler.fit_transform(data[features])
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    data['ml_anomaly'] = iso_forest.fit_predict(X) == -1
    return data, iso_forest, scaler

def train_autoencoder(data, epochs=10, sample_size=100000):
    if len(data) > sample_size:
        data = data.sample(n=sample_size, random_state=42)
        logger.info(f"Sampled {sample_size} rows for Autoencoder training")
    
    scaler = StandardScaler()
    X = scaler.fit_transform(data[features])
    
    model = Sequential