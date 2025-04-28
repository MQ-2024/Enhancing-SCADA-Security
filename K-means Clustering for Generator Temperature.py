import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.cluster import KMeans

# Paths
data_path = r"D:\datasets\Aventa_AV7_IET_OST_SCADA_cleanedtest_less_aggressive.csv"
output_dir = r"D:\datasets\im"

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Load dataset
df = pd.read_csv(data_path)

# Use 'GeneratorTemperature' column (adjust if name differs)
gen_temp_column = 'GeneratorTemperature'
temperatures_before = df[gen_temp_column].astype(float).dropna().values

# Apply cleaning (remove below 0.5°C)
MIN_THRESHOLD = 0.5
temperatures_after = temperatures_before[temperatures_before >= MIN_THRESHOLD]

# Function to plot clusters
def plot_clusters(temps, title, filename):
    X = temps.reshape(-1, 1)
    kmeans = KMeans(n_clusters=5, random_state=42)
    labels = kmeans.fit_predict(X)
    centroids = kmeans.cluster_centers_.flatten()
    cluster_sizes = np.bincount(labels)

    plt.figure(figsize=(8, 6))
    for cluster in range(5):
        cluster_points = temps[labels == cluster]
        cluster_size = cluster_sizes[cluster]
        plt.scatter(cluster_points, [np.log10(cluster_size)] * len(cluster_points), 
                    alpha=0.3, label=f'Cluster {cluster} (Size: {cluster_size})')
    
    plt.scatter(centroids, np.log10(cluster_sizes), c='black', marker='x', s=200, label='Centroids')
    plt.axvline(x=MIN_THRESHOLD, color='red', linestyle='--', label=f'Min Threshold (~{MIN_THRESHOLD}°C)')
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Cluster Size (LOG SCALE)')
    plt.title(title)
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.xlim(0, 80)
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()

# Generate plots
plot_clusters(temperatures_before, 'Generator Temperature Clusters (Before Cleaning, Log Scale)', 
              'generator_temp_clusters_before_cleaning.png')
plot_clusters(temperatures_after, 'Generator Temperature Clusters (After Cleaning, Log Scale)', 
              'generator_temp_clusters_after_cleaning.png')

print("Cluster plots saved to D:\\datasets\\im")