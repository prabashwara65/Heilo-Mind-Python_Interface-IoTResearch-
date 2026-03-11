import json
import logging
from .nbiot_client import NBIoTClient

class FirebasePublisher:
    def __init__(self, nbiot_client, mqtt_broker='mqtt.googleapis.com', mqtt_port=8883,
                 project_id=None, registry_id=None, device_id=None, private_key=None):
        """
        For Google Cloud IoT Core (MQTT bridge). If you use a custom broker,
        adapt accordingly.
        """
        self.nbiot = nbiot_client
        self.broker = mqtt_broker
        self.port = mqtt_port
        self.project_id = project_id
        self.registry_id = registry_id
        self.device_id = device_id
        self.private_key = private_key   # for JWT, if needed
        self.connected = False

    def connect(self):
        # For GCP IoT Core, you need JWT authentication.
        # This example assumes a simple username/password MQTT.
        # You may need to implement JWT generation.
        return self.nbiot.mqtt_connect(self.broker, self.port, client_id=self.device_id)

    def publish_sensor_data(self, data, topic=None):
        """Publish a JSON payload to the given topic (default: /devices/{device_id}/events)."""
        if not topic and self.device_id:
            topic = f'/devices/{self.device_id}/events'
        payload = json.dumps(data)
        self.nbiot.mqtt_publish(topic, payload)

    def disconnect(self):
        self.nbiot.mqtt_disconnect()


# Alternative: Direct Firebase REST API (if NB-IoT supports HTTPS)
class FirebaseRESTPublisher:
    def __init__(self, database_url, secret=None):
        self.database_url = database_url.rstrip('/')
        self.secret = secret   # for legacy Firebase, use auth token

    def publish(self, path, data):
        """Send data via HTTPS POST (needs requests library)."""
        import requests
        url = f"{self.database_url}/{path}.json"
        if self.secret:
            url += f'?auth={self.secret}'
        response = requests.post(url, json=data)
        return response.ok