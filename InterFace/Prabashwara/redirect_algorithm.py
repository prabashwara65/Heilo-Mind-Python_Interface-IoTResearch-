import time
import numpy as np
import serial


class ServoController:

    def __init__(self, arduino_port='/dev/ttyACM0', baud_rate=9600):
        self.port = arduino_port
        self.baud = baud_rate
        self.arduino = None

        # Servo limits
        self.D9_MIN = 10
        self.D9_MAX = 70

        # Other servos fixed (unchanged)
        self.D10_FIXED = 50
        self.D11_FIXED = 70
        self.D12_FIXED = 50

        # Movement range
        self.START = 50
        self.END = 10
        self.STEPS = 20


    def connect(self):
        try:
            self.arduino = serial.Serial(self.port, self.baud, timeout=2)
            time.sleep(3)
            print("✅ Arduino Connected")
            return True
        except Exception as e:
            print("❌ Connection Failed:", e)
            return False


    def send_command(self, d9):

        d9 = int(np.clip(d9, self.D9_MIN, self.D9_MAX))

        cmd = f"{d9} {self.D10_FIXED} {self.D11_FIXED} {self.D12_FIXED}\n"

        self.arduino.write(cmd.encode())
        self.arduino.flush()


    def rotate_d9(self):

        positions = np.linspace(self.START, self.END, self.STEPS)

        for pos in positions:
            angle = int(pos)
            print(f"D9 → {angle}")
            self.send_command(angle)
            time.sleep(0.5)


    def close(self):
        if self.arduino:
            self.arduino.close()
            print("🔌 Arduino Closed")



# ================= MAIN =================

if __name__ == "__main__":

    controller = ServoController('/dev/ttyACM0')

    if controller.connect():
        controller.rotate_d9()
        controller.close()