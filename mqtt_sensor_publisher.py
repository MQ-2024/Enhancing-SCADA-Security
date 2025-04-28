import paho.mqtt.client as mqtt
import json
import time
import pandas as pd
import random

# Define the broker settings
BROKER = "192.168.116.198"
PORT = 1883
FILE_PATH = r'D:\12\Aventa_AV7_IET_OST_SCADA.csv'
TOPIC = "scada/windturbine"  # Single topic for all readings

# Sensor fields
SENSOR_FIELDS = [
    "RotorSpeed", "GeneratorSpeed", "GeneratorTemperature", "WindSpeed", "PowerOutput",
    "offsetWindDirection", "SpeiseSpannung", "PitchDeg", "StatusAnlage", "MaxWindHeute"
]

# Load the dataset and sample 0.1%
df = pd.read_csv(FILE_PATH)
sampled_df = df.sample(frac=0.001, random_state=42)
total_rows = len(df)
sampled_rows = len(sampled_df)
column_names = df.columns.tolist()

print(f"Loaded {sampled_rows} rows (0.1% of {total_rows}) into memory")
print(f"Column Names: {column_names}")

# Convert sampled data to a list of dictionaries
sampled_data = sampled_df.to_dict('records')

# Callback when the client connects to the broker
def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        print("Connected to MQTT broker")
    else:
        print(f"Connection failed with code {rc}")

# Initialize the MQTT client
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.on_connect = on_connect

# Connect to the broker
client.connect(BROKER, PORT, 60)
client.loop_start()

# Continuous publishing loop
try:
    while True:
        # Pick a random row from the sampled data
        row = random.choice(sampled_data)
        
        # Create a single message with all sensor readings
        message = {"timestamp": time.time()}
        for field in SENSOR_FIELDS:
            try:
                value = float(row[field]) if pd.notna(row[field]) else 0.0
                message[field] = value
            except (KeyError, ValueError):
                message[field] = random.uniform(0, 100)  # Fallback value
        
        # Publish the message
        payload = json.dumps(message)
        client.publish(TOPIC, payload, qos=2)
        print(f"\n--- Message Sent to {TOPIC} ---")
        print(f"Payload (JSON):")
        print(json.dumps(message, indent=4))
        print(f"Payload Size: {len(payload)} bytes")
        
        time.sleep(10)  # Wait 10 seconds before the next message

except KeyboardInterrupt:
    print("\nStopped by user")

finally:
    client.loop_stop()
    client.disconnect()
    print("Disconnected from broker")