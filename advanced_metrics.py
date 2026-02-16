import pandas as pd
import numpy as np
import os

# Select tracking file
csv_files = [f for f in os.listdir() if f.endswith("_tracks.csv")]

print("\nAvailable Tracking Files:")
for i, file in enumerate(csv_files):
    print(f"{i + 1}. {file}")

choice = int(input("\nSelect file number: ")) - 1
selected_file = csv_files[choice]

df = pd.read_csv(selected_file)

FPS = 30

print("\n🔎 ANALYZING PLAYER PERFORMANCE...\n")

player_stats = []

for player_id in df["id"].unique():
    player_data = df[df["id"] == player_id]

    if len(player_data) < 5:
        continue

    coords = player_data[["x", "y"]].values

    distances = np.sqrt(np.sum(np.diff(coords, axis=0) ** 2, axis=1))
    total_distance_pixels = np.sum(distances)

    total_time = len(player_data) / FPS
    avg_speed_pixels = total_distance_pixels / total_time

    player_stats.append({
        "id": player_id,
        "frames": len(player_data),
        "distance_pixels": total_distance_pixels,
        "avg_speed_pixels_per_sec": avg_speed_pixels
    })

stats_df = pd.DataFrame(player_stats)

top_distance = stats_df.sort_values(by="distance_pixels", ascending=False).head(5)
top_speed = stats_df.sort_values(by="avg_speed_pixels_per_sec", ascending=False).head(5)

print("🏆 Top 5 Players by Distance Covered:")
print(top_distance)

print("\n⚡ Top 5 Players by Average Speed:")
print(top_speed)
