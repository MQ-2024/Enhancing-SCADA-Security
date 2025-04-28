```python
import pandas as pd
import psycopg2

# Read Zeek conn.log
df = pd.read_csv("/opt/zeek/logs/current/conn.log", sep="\t", skiprows=8, names=[
    "ts", "uid", "id.orig_h", "id.orig_p", "id.resp_h", "id.resp_p", "proto",
    "service", "duration", "orig_bytes", "resp_bytes", "conn_state", "local_orig",
    "local_resp", "missed_bytes", "history", "orig_pkts", "orig_ip_bytes",
    "resp_pkts", "resp_ip_bytes"
])

# Filter for ThingsBoard traffic
ddos_traffic = df[df["id.resp_h"] == "192.168.3.35"]

# Connect to PostgreSQL and store
conn = psycopg2.connect(dbname="thingsboard", user="postgres", password="admin123", host="192.168.3.36")
cur = conn.cursor()
cur.execute("CREATE TABLE IF NOT EXISTS zeek_conn (ts FLOAT, orig_h TEXT, resp_h TEXT, packets BIGINT)")
for _, row in ddos_traffic.iterrows():
    cur.execute("INSERT INTO zeek_conn (ts, orig_h, resp_h, packets) VALUES (%s, %s, %s, %s)", 
                (row["ts"], row["id.orig_h"], row["id.resp_h"], row["orig_pkts"]))
conn.commit()
conn.close()
```