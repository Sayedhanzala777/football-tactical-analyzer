import cv2
import pandas as pd
import numpy as np
import os

# Select CSV
csv_files = [f for f in os.listdir() if f.endswith("_tracks.csv")]

if len(csv_files) == 0:
    print("No tracking CSV files found.")
    exit()

print("\nAvailable Tracking Files:")
for i, file in enumerate(csv_files):
    print(f"{i + 1}. {file}")

choice = int(input("\nSelect file number: ")) - 1
selected_file = csv_files[choice]

df = pd.read_csv(selected_file)

# pick player
counts = df["id"].value_counts()
player_id = counts.index[0]
player_data = df[df["id"] == player_id]

print(f"Drawing trail for player: {player_id}")

# load video
video_name = selected_file.replace("_tracks.csv", ".mp4")
if not os.path.exists(f"input_videos/{video_name}"):
    print("Video not found for this CSV.")
    exit()

cap = cv2.VideoCapture(f"input_videos/{video_name}")

history = []
idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # find matching frame data
    row = player_data[player_data["frame"] == idx]
    if not row.empty:
        x = int(row["x"].values[0])
        y = int(row["y"].values[0])
        history.append((x,y))

    # draw trail
    for i in range(1, len(history)):
        cv2.line(frame, history[i-1], history[i], (0,0,255), 2)

    # draw id
    if len(history) > 0:
        cv2.circle(frame, history[-1], 5, (0,255,0), -1)

    cv2.imshow("Movement Trail Overlay", frame)

    if cv2.waitKey(30) & 0xFF == ord("q"):
        break

    idx += 1

cap.release()
cv2.destroyAllWindows()
