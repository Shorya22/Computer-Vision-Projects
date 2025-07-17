import os
import time
import threading
import requests
from dotenv import load_dotenv
from ultralytics import YOLO
import cv2
import numpy as np
import warnings
warnings.filterwarnings("ignore")


# ---------------------------
# Configs
# ---------------------------
FRAME_WIDTH = 1280
FRAME_HEIGHT = 704
SAVE_DIR = "violations"
TARGET_FPS = 25
COOLDOWN_SECONDS = 0.0
YOLO_CONFIDENCE = 0.75
RTSP_STREAM = "rtsp://admin:87654321@10.40.104.80:554/1/h264major"
MODEL_PATH = "best.pt"

# ---------------------------
# Setup
# ---------------------------
load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
os.makedirs(SAVE_DIR, exist_ok=True)
model = YOLO(MODEL_PATH).to("cuda")

def send_telegram_alert(image_path, caption="🚨 Helmet Violation Detected!"):
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
        with open(image_path, "rb") as photo:
            data = {"chat_id": CHAT_ID, "caption": caption}
            files = {"photo": photo}
            response = requests.post(url, data=data, files=files)
            print("📤 Telegram:", response.status_code, response.json())
    except Exception as e:
        print("❗ Telegram Error:", str(e))

def is_inside(inner, outer):
    return inner[0] >= outer[0] and inner[1] >= outer[1] and inner[2] <= outer[2] and inner[3] <= outer[3]

def alert_violation(image, path):
    cv2.imwrite(path, image)
    send_telegram_alert(path)
    print(f"🚨 Helmet Violation Detected: {path}")

# ---------------------------
# Test Telegram
# ---------------------------
print("🚀 Sending test message...")
requests.get(f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
             params={"chat_id": CHAT_ID, "text": "✅ Bot is connected and running!"})

# ---------------------------
# ROI Drawing
# ---------------------------
roi_points = []
drawing_done = False

def draw_roi(event, x, y, flags, param):
    global roi_points, drawing_done
    if event == cv2.EVENT_LBUTTONDOWN:
        roi_points.append((x, y))
    elif event == cv2.EVENT_RBUTTONDOWN:
        drawing_done = True
        print("✅ ROI completed:", roi_points)

cap = cv2.VideoCapture(RTSP_STREAM)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

time.sleep(1)  # Let camera warm up

ret, roi_frame = cap.read()
if not ret:
    print("❗ Error: Failed to capture frame for ROI drawing.")
    exit()

roi_frame = cv2.resize(roi_frame, (FRAME_WIDTH, FRAME_HEIGHT))
cv2.namedWindow("Draw ROI")
cv2.setMouseCallback("Draw ROI", draw_roi)

while True:
    preview = roi_frame.copy()
    for pt in roi_points:
        cv2.circle(preview, pt, 5, (0, 255, 0), -1)
    if len(roi_points) > 1:
        cv2.polylines(preview, [np.array(roi_points)], isClosed=True, color=(255, 0, 0), thickness=2)

    cv2.imshow("Draw ROI", preview)
    if drawing_done or cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyWindow("Draw ROI")
cap.release()

# ✅ Reopen fresh RTSP stream for inference
time.sleep(1)
cap = cv2.VideoCapture(RTSP_STREAM)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

# ---------------------------
# Main Loop with Manual Inference
# ---------------------------
last_alert_time = 0
last_time = 0
frame_interval = 1.0 / TARGET_FPS

print("🚦 Helmet Violation Detection Running...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❗ Failed to grab frame")
        continue

    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

    now = time.time()
    if now - last_time < frame_interval:
        continue
    last_time = now

    # Run YOLOv8 manually on resized frame
    results = model.predict(frame, imgsz=(FRAME_WIDTH, FRAME_HEIGHT), conf=YOLO_CONFIDENCE, verbose=True)[0]

    riders = []
    with_helmets = []
    without_helmets = []
    number_plates = []

    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        bbox = (x1, y1, x2, y2)

        if conf < YOLO_CONFIDENCE:
            continue

        if label == "rider":
            riders.append(bbox)
        elif label == "with helmet":
            with_helmets.append(bbox)
        elif label == "without helmet":
            without_helmets.append(bbox)
        elif label == "number plate":
            number_plates.append(bbox)

    for rider_box in riders:
        x1, y1, x2, y2 = rider_box
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        inside_roi = cv2.pointPolygonTest(np.array(roi_points), (cx, cy), False) >= 0
        if not inside_roi:
            continue

        has_with_helmet = any(is_inside(h, rider_box) for h in with_helmets)
        has_without_helmet = any(is_inside(h, rider_box) for h in without_helmets)
        has_plate = any(is_inside(p, rider_box) for p in number_plates)

        if has_without_helmet and not has_with_helmet and has_plate:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, "Helmet Violation", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if now - last_alert_time > COOLDOWN_SECONDS:
                last_alert_time = now
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                img_path = os.path.join(SAVE_DIR, f"violation_{timestamp}.jpg")
                threading.Thread(target=alert_violation, args=(frame.copy(), img_path)).start()

    # Draw ROI
    if len(roi_points) > 1:
        cv2.polylines(frame, [np.array(roi_points)], isClosed=True, color=(0, 255, 255), thickness=2)

    cv2.imshow("Helmet Violation Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
