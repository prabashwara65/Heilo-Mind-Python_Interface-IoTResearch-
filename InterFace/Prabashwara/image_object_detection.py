import cv2
import numpy as np
import serial
import time
from collections import deque
from threading import Thread
from queue import Queue, Empty

# ---------------- CONFIG ----------------
ARDUINO_PORT = '/dev/ttyACM0'
BAUD_RATE = 9600

PAN_MIN, PAN_MAX = 10, 80
TILT_MIN, TILT_MAX = 10, 80

SERVO_UPDATE_INTERVAL = 0.05
MAX_SERVO_STEP = 6
BRIGHTNESS_THRESHOLD = 240
INVERT_PAN = False
INVERT_TILT = True
POSITION_HISTORY = 3
DEBUG = True
# ---------------------------------------

# ---------------- GLOBAL VARIABLES ----------------
D10_GLOBAL = 0
D12_GLOBAL = 0

# ---------------- CONNECT TO ARDUINO ----------------
try:
    arduino = serial.Serial(ARDUINO_PORT, BAUD_RATE, timeout=0.1)
    time.sleep(2)
    print("✅ Arduino connected")
except:
    arduino = None
    print("⚠️ Arduino not connected (visual mode)")

# ---------------- QUEUE ----------------
servo_queue = Queue(maxsize=1)

# ---------------- SERIAL THREAD ----------------
def serial_thread():
    global D10_GLOBAL, D12_GLOBAL

    while True:

        # SEND SERVO COMMAND
        try:
            angles = servo_queue.get_nowait()

            if angles is None:
                break

            if arduino:
                cmd = f"{angles[0]},{angles[1]}\n"
                arduino.write(cmd.encode())

        except Empty:
            pass

        # READ DATA FROM ARDUINO
        if arduino and arduino.in_waiting:
            try:
                line = arduino.readline().decode(errors='ignore').strip()

                if line:
                    if DEBUG:
                        print("RAW:", line)

                    parts = line.split(',')

                    if len(parts) >= 4:
                        D10_GLOBAL = int(float(parts[1]))
                        D12_GLOBAL = int(float(parts[3]))

                        if DEBUG:
                            print(f"ARM → D10={D10_GLOBAL}  D12={D12_GLOBAL}")

            except:
                pass

        time.sleep(0.01)

# Start thread
thread = Thread(target=serial_thread, daemon=True)
thread.start()

# ---------------- HELPER FUNCTIONS ----------------
def map_value(val, old_min, old_max, new_min, new_max):
    return int((val - old_min) / (old_max - old_min) * (new_max - new_min) + new_min)

def smooth_move(current, target):
    smoothed = []
    for c, t in zip(current, target):
        diff = t - c
        step = np.clip(diff, -MAX_SERVO_STEP, MAX_SERVO_STEP)
        smoothed.append(c + step)
    return smoothed

def send_servo_latest(angles):
    if servo_queue.full():
        try:
            servo_queue.get_nowait()
        except Empty:
            pass
    servo_queue.put(angles)

def find_camera(max_index=3):
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"✅ Camera found at index {i}")
            return cap
        cap.release()
    return None

# ---------------- MAIN LOOP ----------------
def run():
    global D10_GLOBAL, D12_GLOBAL

    cap = find_camera()

    if cap is None:
        print("❌ No camera found")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    position_history = deque(maxlen=POSITION_HISTORY)

    current_servo = [45, 45]
    target_servo = [45, 45]

    last_servo_update = time.time()

    print("🔆 Light tracker running (Ctrl+C to stop)")

    try:
        while True:

            ret, frame = cap.read()

            if not ret:
                time.sleep(0.01)
                continue

            h, w, _ = frame.shape

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            _, mask = cv2.threshold(gray, BRIGHTNESS_THRESHOLD, 255, cv2.THRESH_BINARY)

            coords = cv2.findNonZero(mask)

            if coords is not None:

                cx = int(coords[:, :, 0].mean())
                cy = int(coords[:, :, 1].mean())

                position_history.append((cx, cy))

                avg_cx = int(sum(p[0] for p in position_history) / len(position_history))
                avg_cy = int(sum(p[1] for p in position_history) / len(position_history))

                pan = map_value(avg_cx, 0, w, PAN_MAX, PAN_MIN) if INVERT_PAN else map_value(avg_cx, 0, w, PAN_MIN, PAN_MAX)

                tilt = map_value(avg_cy, 0, h, TILT_MAX, TILT_MIN) if INVERT_TILT else map_value(avg_cy, 0, h, TILT_MIN, TILT_MAX)

                target_servo = [
                    np.clip(pan, PAN_MIN, PAN_MAX),
                    np.clip(tilt, TILT_MIN, TILT_MAX)
                ]

            now = time.time()

            if now - last_servo_update > SERVO_UPDATE_INTERVAL:

                current_servo = smooth_move(current_servo, target_servo)

                send_servo_latest(current_servo)

                last_servo_update = now

                if DEBUG:
                    light_pos = (avg_cx, avg_cy) if coords is not None else "None"
                    print(f"Camera Servo → H:{current_servo[0]} V:{current_servo[1]} | Light:{light_pos}")
                    print(f"Arm Servos → D10:{D10_GLOBAL}  D12:{D12_GLOBAL}")

            time.sleep(0.001)

    except KeyboardInterrupt:
        print("\n👋 Stopping tracker...")

    finally:
        cap.release()

        send_servo_latest([45, 45])

        servo_queue.put(None)

        thread.join()

        if arduino:
            arduino.close()

        print("✅ Tracker stopped")

# ---------------- START ----------------
if __name__ == "__main__":
    run()