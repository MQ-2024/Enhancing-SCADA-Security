```python
import pandas as pd
import matplotlib.pyplot as plt

# Read Zeek conn.log
df = pd.read_csv("/opt/zeek/logs/current/conn.log", sep="\t", skiprows=8, names=[
    "ts", "uid", "id.orig_h", "id.orig_p", "id.resp_h", "id.resp_p", "proto",
    "service", "duration", "orig_bytes", "resp_bytes", "conn_state", "local_orig",
    "local_resp", "missed_bytes", "history", "orig_pkts", "orig_ip_bytes",
    "resp_pkts", "resp_ip_bytes"
])

# Filter for ThingsBoard traffic
ddos_traffic = df[df["id.resp_h"] == "192.168.3.35"]

# Print total packets
print(f"Total attack packets: {ddos_traffic['orig_pkts'].sum()}")

# Plot packets over time
ddos_traffic.groupby("ts")["orig_pkts"].sum().plot()
plt.xlabel("Timestamp")
plt.ylabel("Packets")
plt.title("DDoS Traffic to ThingsBoard")
plt.savefig("ddos_plot.png", dpi=300, bbox_inches="tight")
plt.show()
```