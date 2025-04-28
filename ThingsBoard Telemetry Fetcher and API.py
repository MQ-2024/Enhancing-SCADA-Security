from flask import Flask, jsonify, request
import mysql.connector
import requests
import time
import threading
import pandas as pd

app = Flask(__name__)

# Configuration
MYSQL_CONFIG = {
    'host': '192.168.116.147',
    'user': 'scada_user',
    'password': 'scada123',
    'database': 'scada_db'
}
DEVICE_ID = "6fbcea80-0472-11f0-b38a-b70fb17fbe02"
THINGSBOARD_LOGIN_URL = "http://localhost:8080/api/auth/login"
THINGSBOARD_URL = f"http://localhost:8080/api/plugins/telemetry/DEVICE/{DEVICE_ID}/values/timeseries"
FETCH_INTERVAL = 10

SENSOR_MAPPING = {
    "RotorSpeed": "sensor_1",
    "GeneratorSpeed": "sensor_2",
    "GeneratorTemperature": "sensor_3",
    "WindSpeed": "sensor_4",
    "PowerOutput": "sensor_5",
    "offsetWindDirection": "sensor_6",
    "SpeiseSpannung": "sensor_7",
    "PitchDeg": "sensor_8",
    "StatusAnlage": "sensor_9",
    "MaxWindHeute": "sensor_10"
}

# Global token variable
bearer_token = None

def get_new_token():
    global bearer_token
    try:
        login_data = {
            "username": "tenant@thingsboard.org",
            "password": "tenant"  # Confirmed working from your curl output
        }
        response = requests.post(THINGSBOARD_LOGIN_URL, json=login_data)
        response.raise_for_status()
        bearer_token = response.json()["token"]
        print(f"[{time.ctime()}] New token obtained: {bearer_token[:20]}...")
    except requests.exceptions.RequestException as e:
        print(f"[{time.ctime()}] Failed to get token: {e}")
        bearer_token = None

def fetch_and_store_telemetry():
    global bearer_token
    while True:
        try:
            if not bearer_token:
                get_new_token()
            if not bearer_token:  # If token fetch failed, skip this iteration
                time.sleep(FETCH_INTERVAL)
                continue
            
            headers = {"Authorization": f"Bearer {bearer_token}"}
            response = requests.get(THINGSBOARD_URL, headers=headers)
            response.raise_for_status()
            data = response.json()

            if not data:
                print(f"[{time.ctime()}] No telemetry data received")
                time.sleep(FETCH_INTERVAL)
                continue

            conn = mysql.connector.connect(**MYSQL_CONFIG)
            cursor = conn.cursor()

            for field in SENSOR_MAPPING.keys():
                if field in data and data[field]:
                    value = data[field][0]["value"]
                    sensor_id = SENSOR_MAPPING[field]
                    try:
                        float_value = float(value)
                        cursor.execute('''
                            INSERT INTO telemetry_data (sensor_id, field, value)
                            VALUES (%s, %s, %s)
                        ''', (sensor_id, field, float_value))
                    except ValueError:
                        print(f"[{time.ctime()}] Skipping non-numeric value for {field}: {value}")

            conn.commit()
            conn.close()
            print(f"[{time.ctime()}] Stored telemetry: {data}")

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:  # Token expired or invalid
                print(f"[{time.ctime()}] 401 Error, refreshing token...")
                get_new_token()
            else:
                print(f"[{time.ctime()}] Error fetching telemetry: {e}")
        except Exception as e:
            print(f"[{time.ctime()}] Unexpected error: {e}")
        
        time.sleep(FETCH_INTERVAL)

@app.route('/telemetry', methods=['GET'])
def get_telemetry():
    limit = request.args.get('limit', default=1000, type=int)
    conn = mysql.connector.connect(**MYSQL_CONFIG)
    df = pd.read_sql_query(f"SELECT * FROM telemetry_data ORDER BY timestamp DESC LIMIT {limit}", conn)
    conn.close()
    return jsonify(df.to_dict(orient="records")), 200

if __name__ == "__main__":
    print("Starting Telemetry API and ThingsBoard fetcher...")
    fetch_thread = threading.Thread(target=fetch_and_store_telemetry, daemon=True)
    fetch_thread.start()
    app.run(host="0.0.0.0", port=5000, debug=True)