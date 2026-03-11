import serial
import time
import logging

class NBIoTClient:
    def __init__(self, port='/dev/ttyUSB0', baudrate=115200, apn='your_apn'):
        self.port = port
        self.baudrate = baudrate
        self.apn = apn
        self.ser = None
        self.connected = False
        logging.basicConfig(level=logging.INFO)

    def connect(self):
        """Open serial port and initialize modem."""
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=1)
            self._send_at('AT', 'OK')                 # basic test
            self._send_at('AT+CFUN=1', 'OK')          # full functionality
            self._send_at(f'AT+CGDCONT=1,"IP","{self.apn}"', 'OK')
            self._send_at('AT+CGATT=1', 'OK')         # attach to network
            self._send_at('AT+CSQ')                    # check signal
            self.connected = True
            logging.info("NB-IoT modem ready")
            return True
        except Exception as e:
            logging.error(f"Modem connection failed: {e}")
            return False

    def _send_at(self, command, expected='OK', timeout=5):
        """Send AT command and wait for expected response."""
        self.ser.write((command + '\r\n').encode())
        time.sleep(0.1)
        response = b''
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.ser.in_waiting:
                response += self.ser.read(self.ser.in_waiting)
                if expected.encode() in response:
                    return response.decode()
        raise TimeoutError(f"AT command '{command}' did not return '{expected}'")

    def mqtt_connect(self, broker, port=1883, client_id='solar_pi', username=None, password=None):
        """Establish MQTT connection via modem (AT+QMTCFG, AT+QMTOPEN, etc.)."""
        # This is a placeholder – actual implementation depends on BG96 firmware
        # Usually: configure SSL, open network, connect
        cmd = f'AT+QMTOPEN=0,"{broker}",{port}'
        self._send_at(cmd, '+QMTOPEN: 0,0')
        if username:
            self._send_at(f'AT+QMTCFG="auth",0,1,"{username}","{password}"', 'OK')
        self._send_at(f'AT+QMTCONN=0,"{client_id}"', '+QMTCONN: 0,0,0')
        logging.info("MQTT connected")
        return True

    def mqtt_publish(self, topic, payload, qos=1):
        """Publish a message."""
        # Escape quotes in payload if needed
        payload = payload.replace('"', '\\"')
        cmd = f'AT+QMTPUB=0,"{topic}",{qos},"{payload}"'
        self._send_at(cmd, '+QMTPUB: 0,0')
        logging.info(f"Published to {topic}")

    def mqtt_disconnect(self):
        self._send_at('AT+QMTDISC=0', '+QMTDISC: 0,0')

    def disconnect(self):
        if self.ser:
            self.ser.close()
        self.connected = False