import cv2
import numpy as np

video_path = "input_videos/match2.mp4"  # change if needed

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
cap.release()

points = []

def click_event(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])
        print(f"Point selected: {x}, {y}")
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Select 4 Pitch Corners", frame)

cv2.imshow("Select 4 Pitch Corners", frame)
cv2.setMouseCallback("Select 4 Pitch Corners", click_event)

print("Click 4 pitch corners in this order:")
print("Top-left → Top-right → Bottom-right → Bottom-left")

cv2.waitKey(0)
cv2.destroyAllWindows()

print("\nSelected Points:")
print(points)
