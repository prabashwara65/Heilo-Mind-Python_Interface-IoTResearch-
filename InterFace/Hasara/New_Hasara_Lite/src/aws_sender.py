from pathlib import Path
import json
import ssl
import time
import paho.mqtt.client as mqtt

# --- CONFIGURATION ---
AWS_ENDPOINT = "ajmja1mzmi1j4-ats.iot.eu-north-1.amazonaws.com"
PORT = 8883
TOPIC = "hasara/sensor_data"

# Root folder = Main_Program
ROOT_DIR = Path(__file__).resolve().parent.parent.parent  # src/... -> Main_Program

CERT_PATH = ROOT_DIR / "certs/6c2a210110a2809a43a9da4b7f2c58bb1ae4fc5e4cc7d35a5f9747eb84709ce8-certificate.pem.crt"
KEY_PATH = ROOT_DIR / "certs/6c2a210110a2809a43a9da4b7f2c58bb1ae4fc5e4cc7d35a5f9747eb84709ce8-private.pem.key"
ROOT_CA_PATH = ROOT_DIR / "certs/AmazonRootCA1.pem"


def create_mqtt_client(client_id="hasara_client"):
    # Paho MQTT v2 expects an enum for callback_api_version.
    # Keep compatibility with both v1 and v2.
    try:
        client = mqtt.Client(
            client_id=client_id,
            callback_api_version=mqtt.CallbackAPIVersion.VERSION1
        )
    except (AttributeError, TypeError):
        client = mqtt.Client(client_id=client_id)

    # Configure TLS
    client.tls_set(ca_certs=str(ROOT_CA_PATH),
                   certfile=str(CERT_PATH),
                   keyfile=str(KEY_PATH),
                   cert_reqs=ssl.CERT_REQUIRED,
                   tls_version=ssl.PROTOCOL_TLSv1_2,
                   ciphers=None)

    client.tls_insecure_set(False)
    return client


def publish_data(sensor_data):
    client = create_mqtt_client()
    client.connect(AWS_ENDPOINT, PORT)
    client.loop_start()

    payload = json.dumps(sensor_data)
    print(f"Publishing to {TOPIC}: {payload}")
    client.publish(TOPIC, payload, qos=1)

    time.sleep(1)
    client.loop_stop()
    client.disconnect()
