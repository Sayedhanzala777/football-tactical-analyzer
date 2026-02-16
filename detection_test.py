import cv2
import numpy as np
import os
from ultralytics import YOLO

# -------- SELECT VIDEO --------
video_folder = "input_videos"
videos = [f for f in os.listdir(video_folder) if f.endswith(".mp4")]

if not videos:
    print("No videos found.")
    exit()

print("\nAvailable Videos:")
for i, video in enumerate(videos):
    print(f"{i + 1}. {video}")

choice = int(input("\nSelect video number: ")) - 1
video_name = videos[choice]
video_path = os.path.join(video_folder, video_name)

print(f"\nProcessing: {video_name}\n")

# -------- LOAD MODEL --------
model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(video_path)

FPS = 30

next_id = 0
objects = {}
history = {}
distance_travelled = {}
team_map = {}
team_possession = {"Red": 0, "Blue": 0}
total_frames = 0

def calculate_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def match_objects(detections, previous_objects, threshold=50):
    global next_id
    updated = {}
    for centroid in detections:
        matched = False
        for obj_id, old_centroid in previous_objects.items():
            if calculate_distance(centroid, old_centroid) < threshold:
                updated[obj_id] = centroid
                matched = True
                break
        if not matched:
            updated[next_id] = centroid
            history[next_id] = []
            distance_travelled[next_id] = 0
            next_id += 1
    return updated

while True:
    ret, frame = cap.read()
    if not ret:
        break

    total_frames += 1

    results = model(frame)

    detections = []
    ball_position = None
    player_boxes = {}

    for box, cls in zip(results[0].boxes.xyxy, results[0].boxes.cls):
        x1, y1, x2, y2 = map(int, box)
        class_id = int(cls)

        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        # BALL
        if class_id == 32:
            ball_position = (cx, cy)
            cv2.circle(frame, ball_position, 6, (0, 255, 255), -1)
            continue

        # PLAYER
        if class_id == 0:
            detections.append((cx, cy))
            player_boxes[(cx, cy)] = (x1, y1, x2, y2)

    objects = match_objects(detections, objects)

    # -------- DRAW PLAYERS --------
    for obj_id, centroid in objects.items():
        cx, cy = centroid
        x1, y1, x2, y2 = player_boxes.get(centroid, (cx-20, cy-40, cx+20, cy+40))

        if obj_id not in history:
            history[obj_id] = []
            distance_travelled[obj_id] = 0

        history[obj_id].append((cx, cy))

        # Distance + Speed
        if len(history[obj_id]) > 1:
            dist = calculate_distance(history[obj_id][-1], history[obj_id][-2])
            distance_travelled[obj_id] += dist
            current_speed = dist * FPS
        else:
            current_speed = 0

        # Team classification (once)
        if obj_id not in team_map:
            roi = frame[y1:y2, x1:x2]
            if roi.size > 0:
                avg_color = np.mean(roi.reshape(-1, 3), axis=0)
                if avg_color[2] > avg_color[1]:
                    team_map[obj_id] = "Red"
                else:
                    team_map[obj_id] = "Blue"
            else:
                team_map[obj_id] = "Blue"

        team = team_map[obj_id]
        color = (0, 0, 255) if team == "Red" else (255, 0, 0)

        # Draw trail
        for i in range(1, len(history[obj_id])):
            cv2.line(frame,
                     history[obj_id][i - 1],
                     history[obj_id][i],
                     color,
                     2)

        # Draw box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        rating = min(10, (distance_travelled[obj_id] / 800) + (current_speed / 150))

        # Overlay
        cv2.putText(frame, f"ID {obj_id} ({team})", (x1, y1 - 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.putText(frame, f"Dist: {int(distance_travelled[obj_id])}",
                    (x1, y1 - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        cv2.putText(frame, f"Speed: {int(current_speed)}",
                    (x1, y1 - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        cv2.putText(frame, f"Rating: {rating:.1f}",
                    (x1, y1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # -------- POSSESSION --------
    if ball_position:
        closest_player = None
        min_dist = 999999
        for pid, pos in objects.items():
            d = calculate_distance(pos, ball_position)
            if d < min_dist:
                min_dist = d
                closest_player = pid

        if closest_player is not None:
            team = team_map.get(closest_player, None)
            if team:
                team_possession[team] += 1

    red_pos = (team_possession["Red"] / total_frames) * 100 if total_frames else 0
    blue_pos = (team_possession["Blue"] / total_frames) * 100 if total_frames else 0

    cv2.putText(frame, f"Red Possession: {red_pos:.1f}%",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.putText(frame, f"Blue Possession: {blue_pos:.1f}%",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.imshow("🔥 Tactical Analyzer PRO", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
