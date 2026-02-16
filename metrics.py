import pandas as pd
import numpy as np

df = pd.read_csv("player_tracks.csv")

counts = df["id"].value_counts()
top_players = counts.head(5).index

print("Top 5 Players by Tracking Duration:")
print(counts.head())

print("\nDistance Covered (in pixels):\n")

for player_id in top_players:
    player_data = df[df["id"] == player_id]

    coords = player_data[["x", "y"]].values

    distances = np.sqrt(np.sum(np.diff(coords, axis=0) ** 2, axis=1))
    total_distance = np.sum(distances)

    print(f"Player {player_id}: {total_distance:.2f} pixels")
